import os
from typing import Any, Dict, Optional, Tuple, List
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
import shap

app = FastAPI(
    title="API Solvabilité BTP",
    description=(
        "Consultation, scoring XGBoost temps réel et arbitrage d'octroi de"
        " crédit B2B"
    ),
    version="1.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# CHARGEMENT DE L'ARTEFACT XGBOOST ET DE L'EXPLAINER SHAP
# -----------------------------------------------------------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "modele_1_xgboost.joblib")
print("MODEL_PATH", MODEL_PATH)
model = None
explainer = None
feature_names = []

# Dictionnaire de traduction métier des variables pour l'affichage final
DICTIONNAIRE_LABELS = {
    "age_entreprise_ans": "Ancienneté de l'entreprise",
    "chiffre_affaires": "Chiffre d'affaires",
    "resultat_net": "Résultat net",
    "capitaux_propres": "Capitaux propres",
    "tresorerie": "Trésorerie disponible",
    "dettes_financieres": "Dettes financières",
    "total_actif": "Total de l'actif",
    "marge_nette": "Marge nette",
    "ratio_endettement": "Ratio d'endettement",
    "ratio_autonomie_financiere": "Autonomie financière",
    "tresorerie_jours_ca": "Couverture de trésorerie (jours)",
    "has_bilan_depose": "Transparence comptable (dépôt du bilan)",
    "anciennete_bilan_annees": "Fraîcheur des données comptables",
}

if os.path.exists(MODEL_PATH):
    try:
        artifact = joblib.load(MODEL_PATH)
        model = artifact["model"]
        feature_names = artifact["feature_names"]
        # Initialisation de l'explainer TreeSHAP
        explainer = shap.TreeExplainer(model)
        print(f"✅ Modèle XGBoost et TreeSHAP initialisés ({len(feature_names)} features).")
    except Exception as e:
        print(f"⚠️ Erreur lors du chargement de l'artefact : {e}")
else:
    print(f"⚠️ Artefact introuvable sur le chemin : {MODEL_PATH}")


def get_db_connection():
    """Établit la connexion avec la base de données PostgreSQL Render."""
    db_url = os.getenv(
        "DATABASE_URL",
        #"postgresql://scoring_b2b:INihq7lK3FeGWNEprxz8tURdmZWFl2qt@dpg-dai0psid0e5s7399fijg-a.frankfurt-postgres.render.com/scoring_b2b?sslmode=require",
        "postgresql://postgres.roxylxdgllxzetdlaptd:iM%40D2OO3lIn%402OO726@aws-0-eu-west-1.pooler.supabase.com:5432/postgres?sslmode=require",
    )
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)
    return psycopg2.connect(db_url)


# -----------------------------------------------------------------------------
# JOURNALISATION ASYNCHRONE DANS L'HISTORIQUE (POSTGRESQL)
# -----------------------------------------------------------------------------
def enregistrer_historique_decision(
    siren: str,
    nom_commercial: Optional[str],
    montant_demande: float,
    proba_defaut: Optional[float],
    statut_decision: str,
    plafond_accorde: float,
    motif_decision: str,
):
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            query = """
                INSERT INTO historique_decisions (
                    siren,
                    nom_commercial,
                    montant_demande,
                    proba_defaut,
                    statut_decision,
                    plafond_accorde,
                    motif_decision
                ) VALUES (%s, %s, %s, %s, %s, %s, %s);
            """
            cur.execute(
                query,
                (
                    siren,
                    nom_commercial or "Inconnu",
                    montant_demande,
                    round(proba_defaut, 4) if proba_defaut is not None else None,
                    statut_decision,
                    plafond_accorde,
                    motif_decision,
                ),
            )
            conn.commit()
    except Exception as exc:
        print(f"⚠️ Erreur insertion historique_decisions pour SIREN {siren} : {exc}")
    finally:
        if conn:
            conn.close()


# -----------------------------------------------------------------------------
# INFÉRENCE XGBOOST ET EXTRACTION DU TOP 3 DES FACTEURS EXPLICATIFS
# -----------------------------------------------------------------------------
def predire_risque_et_raisons(entreprise_dict: dict) -> Tuple[float, List[dict]]:
    """Prépare les features, calcule proba_defaut et extrait le top 3 des raisons (SHAP)."""
    if model is None:
        return 0.0, []

    df_row = pd.DataFrame([entreprise_dict])

    # 1. Ancienneté du bilan
    date_exec = entreprise_dict.get("date_exercice")
    has_bilan = int(entreprise_dict.get("has_bilan_depose") or 0)

    if has_bilan == 1 and pd.notna(date_exec):
        date_cloture = pd.to_datetime(date_exec, errors="coerce")
        df_row["anciennete_bilan_annees"] = (
            pd.Timestamp.now() - date_cloture
        ).days / 365.25
    else:
        df_row["anciennete_bilan_annees"] = np.nan

    # 2. Confidentialité comptable
    if has_bilan == 1:
        if df_row.get("chiffre_affaires", pd.Series([None])).iloc[0] == 0:
            df_row["chiffre_affaires"] = np.nan
        if df_row.get("resultat_net", pd.Series([None])).iloc[0] == 0:
            df_row["resultat_net"] = np.nan

    # 3. Exclusion des colonnes
    colonnes_a_exclure = [
        "siren",
        "nom_commercial",
        "cible_defaillance",
        "nature_jugement",
        "date_jugement",
        "date_exercice",
    ]
    features = df_row.drop(
        columns=[c for c in colonnes_a_exclure if c in df_row.columns]
    )

    if "ratio_endettement" in features.columns:
        features["ratio_endettement"] = features["ratio_endettement"].fillna(0.0)

    colonnes_num = [
        "chiffre_affaires",
        "resultat_net",
        "capitaux_propres",
        "tresorerie",
        "dettes_financieres",
        "total_actif",
        "marge_nette",
        "ratio_endettement",
        "ratio_autonomie_financiere",
        "tresorerie_jours_ca",
        "age_entreprise_ans",
        "anciennete_bilan_annees",
    ]

    for col in colonnes_num:
        if col in features.columns:
            features[col] = pd.to_numeric(
                features[col].replace(["", "NULL", "None", "nan", None], np.nan),
                errors="coerce",
            )

    masque_confidentiel = (features.get("has_bilan_depose") == 1) & (
        features.get("chiffre_affaires") <= 0
    )
    features.loc[
        masque_confidentiel,
        ["chiffre_affaires", "marge_nette", "tresorerie_jours_ca"],
    ] = np.nan

    # 4. Encodage et alignement
    features_encoded = pd.get_dummies(features, drop_first=True)
    features_encoded.columns = features_encoded.columns.str.replace(
        r"[\[\]<]", "_", regex=True
    )
    features_aligned = features_encoded.reindex(
        columns=feature_names, fill_value=0
    )

    # 5. Probabilité
    proba = float(model.predict_proba(features_aligned)[0, 1])

    # 6. Extraction des 3 raisons dominantes via SHAP
    top_raisons = []
    try:
        if explainer is not None:
            shap_vals = explainer(features_aligned).values[0]
            # Si classification binaire renvoie 2 classes, on prend la classe 1
            if len(shap_vals.shape) == 2:
                shap_vals = shap_vals[:, 1]

            # Top 3 des valeurs absolues
            top_indices = np.argsort(np.abs(shap_vals))[-3:][::-1]

            for idx in top_indices:
                nom_col = feature_names[idx]
                val_shap = float(shap_vals[idx])
                val_reelle = features_aligned.iloc[0, idx]

                nom_clair = DICTIONNAIRE_LABELS.get(nom_col, nom_col.replace("_", " ").capitalize())
                favorable = val_shap < 0  # < 0 diminue le risque de défaut (favorable)

                top_raisons.append({
                    "variable": nom_clair,
                    "valeur": None if pd.isna(val_reelle) else round(float(val_reelle), 2),
                    "impact": "favorable" if favorable else "defavorable",
                    "description": (
                        f"{nom_clair} : impact positif renforçant la solvabilité"
                        if favorable
                        else f"{nom_clair} : fragilité augmentant le risque de défaillance"
                    ),
                })
    except Exception as e:
        print(f"⚠️ Erreur lors du calcul SHAP : {e}")

    return proba, top_raisons


# -----------------------------------------------------------------------------
# MOTEUR DE RÈGLES D'OCTROI DE CRÉDIT
# -----------------------------------------------------------------------------
def calculer_arbitrage_credit(
    donnees: dict, proba: float, montant_demande: float, top_raisons: List[dict]
) -> dict:
    has_bilan = int(donnees.get("has_bilan_depose") or 0)
    ca = float(donnees.get("chiffre_affaires") or 0.0)
    fp = float(donnees.get("capitaux_propres") or 0.0)
    nature_jug = donnees.get("nature_jugement")
    #bodacc_actif = bool(
     #   donnees.get("bodacc_actif") or (nature_jug not in ["AUCUN_INCIDENT", None, ""]))
    bodacc_actif = int(donnees.get("cible_defaillance") or 0) == 1
    

    # 1. Règle NOIR
    if bodacc_actif:
        return {
            "statut": "NOIR",
            "proba_defaut": proba,
            "plafond_autorise": 0.0,
            "montant_accorde": 0.0,
            "motif": "Procédure collective active au BODACC : compte bloqué et refus formel de crédit.",
            "top_raisons": [
                {"variable": "BODACC", "impact": "defavorable", "description": f"Incident judiciaire critique : {nature_jug}"},
                {"variable": "Cadre légal", "impact": "defavorable", "description": "Interdiction d'ouverture de nouvel encours non garanti"},
                {"variable": "Solvabilité globale", "impact": "defavorable", "description": "Risque maximal d'impayé constaté"}
            ],
        }

    # 2. Règle BLOCAGE BILAN
    elif has_bilan == 1 and fp <= 0:
        return {
            "statut": "BLOCAGE_BILAN",
            "proba_defaut": proba,
            "plafond_autorise": 0.0,
            "montant_accorde": 0.0,
            "motif": f"Capitaux propres négatifs ou nuls ({fp:,.0f} €) : encours bloqué pour sous-capitalisation. Règlement comptant exigé",
            "top_raisons": [
                {"variable": "Capitaux Propres", "impact": "defavorable", "description": f"Fonds propres négatifs ou nuls ({fp:,.0f} €)"},
                {"variable": "Assise financière", "impact": "defavorable", "description": "Incapacité d'absorption des pertes"},
                {"variable": "Règle prudentielle", "impact": "defavorable", "description": "Refus d'octroi systématique sur fonds propres négatifs"}
            ],
        }

    # 3. Règle ROUGE
    elif proba > 0.30:
        return {
            "statut": "ROUGE",
            "proba_defaut": proba,
            "plafond_autorise": 0.0,
            "montant_accorde": 0.0,
            "motif": f"Risque critique de défaut ({proba * 100:.1f} % > 30 %) : aucun encours non garanti. Règlement comptant exigé.",
            "top_raisons": top_raisons,
        }

    # 4. Règle ORANGE
    elif 0.15 < proba <= 0.30:
        if has_bilan == 1 and ca >= 10000:
            capacite_nominale = min(ca * 0.07, fp * 0.25)
            plafond_ajuste = 0.75 * capacite_nominale
            motif_detail = "décote de 25 % sur capacité d'absorption"
        elif has_bilan == 1 and ca < 10000:
            capacite_nominale = min(5000.0, fp * 0.25)
            plafond_ajuste = 0.75 * capacite_nominale
            motif_detail = "décote de 25 % sur capacité d'absorption"
        else:
            capacite_nominale = 5000.0
            plafond_ajuste = 0.5 * capacite_nominale
            motif_detail = "forfait de confiance avec abattement de 50 %"

        montant_final = min(montant_demande, plafond_ajuste)
        return {
            "statut": "ORANGE",
            "proba_defaut": proba,
            "plafond_autorise": round(plafond_ajuste, 2),
            "montant_accorde": round(montant_final, 2),
            "motif": f"Risque modéré ({proba * 100:.1f} %) : {motif_detail}. Délais réduits à 30 jours ou acompte partiel.",
            "top_raisons": top_raisons,
        }

    # 5. Règle VERT
    else:
        if has_bilan == 1 and ca >= 10000:
            capacite_nominale = min(ca * 0.07, fp * 0.25)
            plafond_ajuste = capacite_nominale
            motif_detail = "100 % de la capacité validée"
        elif has_bilan == 1 and ca < 10000:
            capacite_nominale = min(5000.0, fp * 0.25)
            plafond_ajuste = capacite_nominale
            motif_detail = "forfait de confiance car chiffre d'affaires insuffisant"
        else:
            capacite_nominale = 5000.0
            plafond_ajuste = 0.8 * capacite_nominale
            motif_detail = "forfait de confiance abattement de 20 %"

        montant_final = min(montant_demande, plafond_ajuste)
        return {
            "statut": "VERT",
            "proba_defaut": proba,
            "plafond_autorise": round(plafond_ajuste, 2),
            "montant_accorde": round(montant_final, 2),
            "motif": f"Risque faible ({proba * 100:.1f} %) : {motif_detail}. Conditions commerciales standards (30-60 jours).",
            "top_raisons": top_raisons,
        }


# -----------------------------------------------------------------------------
# ROUTE PRINCIPALE
# -----------------------------------------------------------------------------
@app.get("/entreprises/{siren}", response_model=Dict[str, Any], status_code=status.HTTP_200_OK)
def get_entreprise_by_siren(
    siren: str,
    background_tasks: BackgroundTasks,
    montant_demande: float = Query(15000.0, ge=0.0),
):
    siren_propre = siren.strip()
    if len(siren_propre) != 9 or not siren_propre.isdigit():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Le numéro SIREN doit comporter exactement 9 chiffres.",
        )

    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            query = "SELECT * FROM entreprises_btp_solvabilite WHERE siren = %s;"
            cur.execute(query, (siren_propre,))
            entreprise = cur.fetchone()

            if not entreprise:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Aucune entreprise trouvée avec le SIREN {siren_propre}.",
                )

            donnees_entreprise = dict(entreprise)
            cible = int(donnees_entreprise.get("cible_defaillance") or 0)

            if cible == 1:
                proba = 1.0
                raisons = [
                    {"variable": "Cible Défaillance", "impact": "defavorable", "description": "Défaillance déjà actée au dossier"},
                ]
            else:
                proba, raisons = predire_risque_et_raisons(donnees_entreprise)

            arbitrage = calculer_arbitrage_credit(
                donnees_entreprise, proba, montant_demande, raisons
            )

            # Historisation
            background_tasks.add_task(
                enregistrer_historique_decision,
                siren=siren_propre,
                nom_commercial=donnees_entreprise.get("nom_commercial"),
                montant_demande=montant_demande,
                proba_defaut=arbitrage.get("proba_defaut"),
                statut_decision=arbitrage.get("statut"),
                plafond_accorde=arbitrage.get("montant_accorde", 0.0),
                motif_decision=arbitrage.get("motif", ""),
            )

            donnees_entreprise["decision_scoring"] = arbitrage
            return donnees_entreprise

    except psycopg2.Error as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur de connexion à la base de données : {str(e)}", 
        )
    finally:
        if conn:
            conn.close()