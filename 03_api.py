import os
from typing import Any, Dict, Optional
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor

app = FastAPI(
    title="API Solvabilité BTP",
    description="Consultation et scoring de solvabilité pour les entreprises du BTP",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# CHARGEMENT DE L'ARTEFACT XGBOOST
# -----------------------------------------------------------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "modele_1_xgboost.joblib")
model = None
feature_names = []

if os.path.exists(MODEL_PATH):
    try:
        artifact = joblib.load(MODEL_PATH)
        model = artifact["model"]
        feature_names = artifact["feature_names"]
        print(f"✅ Modèle XGBoost chargé ({len(feature_names)} features attendues).")
    except Exception as e:
        print(f"⚠️ Erreur lors du chargement de l'artefact : {e}")
else:
    print(f"⚠️ Artefact introuvable sur le chemin : {MODEL_PATH}")


def get_db_connection():
    db_url = os.getenv(
        "DATABASE_URL",
        "postgresql://scoring_b2b:INihq7lK3FeGWNEprxz8tURdmZWFl2qt@dpg-dai0psid0e5s7399fijg-a.frankfurt-postgres.render.com/scoring_b2b?sslmode=require",
    )
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)
    return psycopg2.connect(db_url)


# -----------------------------------------------------------------------------
# JOURNALISATION ASYNCHRONE DANS L'HISTORIQUE
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
    """Insère l'évaluation dans la table historique_decisions."""
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


def predire_risque_xgboost(entreprise_dict: dict) -> float:
    """Prépare les features et calcule la probabilité p de défaillance."""
    if model is None:
        return 0.0

    df_row = pd.DataFrame([entreprise_dict])

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

    features_encoded = pd.get_dummies(features, drop_first=True)
    features_encoded.columns = features_encoded.columns.str.replace(
        r"[\[\]<]", "_", regex=True
    )

    features_aligned = features_encoded.reindex(columns=feature_names, fill_value=0)
    proba = float(model.predict_proba(features_aligned)[0, 1])
    return proba


def calculer_arbitrage_credit(
    entreprise: dict, proba: float, montant_demande: float
) -> dict:
    """Applique le moteur de règles d'octroi de crédit."""
    nature_jugement = (entreprise.get("nature_jugement") or "").upper()
    has_bilan = int(entreprise.get("has_bilan_depose") or 0)
    ca = float(entreprise.get("chiffre_affaires") or 0.0)
    fp = float(entreprise.get("capitaux_propres") or 0.0)

    # 1. Statut NOIR : procédure collective active au BODACC
    if (
        "REDRESSEMENT" in nature_jugement
        or "LIQUIDATION" in nature_jugement
        or "SAUVEGARDE" in nature_jugement
    ):
        return {
            "statut": "NOIR",
            "proba_defaut": proba,
            "plafond_autorise": 0.0,
            "montant_accorde": 0.0,
            "motif": f"Procédure collective ouverte : {entreprise.get('nature_jugement')}",
        }

    # 2. Statut ROUGE : risque critique (p > 0.35)
    if proba > 0.35:
        return {
            "statut": "ROUGE",
            "proba_defaut": proba,
            "plafond_autorise": 0.0,
            "montant_accorde": 0.0,
            "motif": f"Risque critique d'impayé ({proba * 100:.1f} %). Règlement comptant exigé.",
        }

    # 3. Calcul du plafond théorique bilanciel ou forfaitaire
    if has_bilan == 1:
        plafond_base = min(ca * 0.07, max(0.0, fp * 0.25))
    else:
        plafond_base = 5000.0

    # 4. Statut ORANGE (0.15 < p <= 0.35) : application d'une décote
    if proba > 0.15:
        decote = 0.25 if has_bilan == 1 else 0.50
        plafond_ajuste = plafond_base * (1.0 - decote)
        return {
            "statut": "ORANGE",
            "proba_defaut": proba,
            "plafond_autorise": round(plafond_ajuste, 2),
            "montant_accorde": round(min(plafond_ajuste, montant_demande), 2),
            "motif": f"Risque modéré ({proba * 100:.1f} %) : décote prudentielle de {int(decote * 100)} % appliquée.",
        }

    # 5. Statut VERT (p <= 0.15) : dossier sain
    return {
        "statut": "VERT",
        "proba_defaut": proba,
        "plafond_autorise": round(plafond_base, 2),
        "montant_accorde": round(min(plafond_base, montant_demande), 2),
        "motif": f"Entreprise saine ({proba * 100:.1f} % de risque). Encours validé.",
    }


@app.get(
    "/entreprises/{siren}",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Récupère les données SQL, calcule le score et historise la décision",
)
def get_entreprise_by_siren(
    siren: str,
    background_tasks: BackgroundTasks,
    montant_demande: float = Query(
        15000.0, ge=0.0, description="Montant de l'encours demandé en euros"
    ),
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
            query = """
                SELECT *
                FROM entreprises_btp_solvabilite
                WHERE siren = %s;
            """
            cur.execute(query, (siren_propre,))
            entreprise = cur.fetchone()

            if not entreprise:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Aucune entreprise trouvée avec le SIREN {siren_propre}.",
                )

            donnees_entreprise = dict(entreprise)
            cible = int(donnees_entreprise.get("cible_defaillance") or 0)

            # Inférence ou court-circuit si incident déjà acté
            if cible == 1:
                proba = 1.0
            else:
                proba = predire_risque_xgboost(donnees_entreprise)

            arbitrage = calculer_arbitrage_credit(
                donnees_entreprise, proba, montant_demande
            )

            # Enregistrement en tâche de fond dans PostgreSQL
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