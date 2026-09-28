import os
from typing import Any, Dict, Optional
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor

app = FastAPI(
    title="API Solvabilité BTP",
    description=(
        "Consultation, scoring XGBoost temps réel et arbitrage d'octroi de"
        " crédit B2B"
    ),
    version="1.1.0",
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
print("MODEL_PATH",MODEL_PATH)
model = None
feature_names = []

if os.path.exists(MODEL_PATH):
  try:
    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    feature_names = artifact["feature_names"]
    print(
        f"✅ Modèle XGBoost chargé ({len(feature_names)} features attendues)."
    )
  except Exception as e:
    print(f"⚠️ Erreur lors du chargement de l'artefact : {e}")
else:
  print(f"⚠️ Artefact introuvable sur le chemin : {MODEL_PATH}")


def get_db_connection():
  """Établit la connexion avec la base de données PostgreSQL Render."""
  db_url = os.getenv(
      "DATABASE_URL",
      "postgresql://scoring_b2b:INihq7lK3FeGWNEprxz8tURdmZWFl2qt@dpg-dai0psid0e5s7399fijg-a.frankfurt-postgres.render.com/scoring_b2b?sslmode=require",
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
  """Insère l'évaluation dans la table historique_decisions sans bloquer l'appel HTTP."""
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
# INFÉRENCE XGBOOST AVEC FEATURE ENGINEERING DYNAMIQUE
# -----------------------------------------------------------------------------
def predire_risque_xgboost(entreprise_dict: dict) -> float:
  """Prépare les features à la volée et calcule la probabilité de défaillance p."""
  if model is None:
    return 0.0

  df_row = pd.DataFrame([entreprise_dict])

  # 1. Calcul dynamique de l'ancienneté du bilan par rapport à la date du jour
  date_exec = entreprise_dict.get("date_exercice")
  has_bilan = int(entreprise_dict.get("has_bilan_depose") or 0)

  if has_bilan == 1 and pd.notna(date_exec):
    date_cloture = pd.to_datetime(date_exec, errors="coerce")
    df_row["anciennete_bilan_annees"] = (
        pd.Timestamp.now() - date_cloture
    ).days / 365.25
  else:
    df_row["anciennete_bilan_annees"] = np.nan

  # 2. Remplacement des faux zéros liés à la confidentialité comptable
  if has_bilan == 1:
    if df_row.get("chiffre_affaires", pd.Series([None])).iloc[0] == 0:
      df_row["chiffre_affaires"] = np.nan
    if df_row.get("resultat_net", pd.Series([None])).iloc[0] == 0:
      df_row["resultat_net"] = np.nan

  # 3. Colonnes à exclure de la matrice d'apprentissage
  colonnes_a_exclure = [
      "siren",
      "nom_commercial",
      "cible_defaillance",
      "nature_jugement", 
      "date_jugement",
      "date_exercice",  # Remplacée par anciennete_bilan_annees
  ]
  features = df_row.drop(
      columns=[c for c in colonnes_a_exclure if c in df_row.columns]
  )
  print("features", features)
  # 4. Imputation du ratio d'endettement si présent
  if "ratio_endettement" in features.columns:
    features["ratio_endettement"] = features["ratio_endettement"].fillna(0.0)


# 3. Typage numérique strict et conversion des textes vides en np.nan
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
  # 4. Traitement des faux zéros liés aux bilans confidentiels
  masque_confidentiel = (features.get("has_bilan_depose") == 1) & (features.get("chiffre_affaires") <= 0)
  features.loc[
      masque_confidentiel, ["chiffre_affaires", "marge_nette", "tresorerie_jours_ca"]
  ] = np.nan

  # 5. Encodage One-Hot et nettoyage des caractères interdits par XGBoost
  features_encoded = pd.get_dummies(features, drop_first=True)
  features_encoded.columns = features_encoded.columns.str.replace(
      r"[\[\]<]", "_", regex=True
  )

  # 6. Alignement strict avec l'ordre et le nom des features de l'artefact
  features_aligned = features_encoded.reindex(
      columns=feature_names, fill_value=0
  )

  # 7. Calcul de la probabilité de la classe 1 (défaillance)
  proba = float(model.predict_proba(features_aligned)[0, 1])
  return proba


# -----------------------------------------------------------------------------
# MOTEUR DE RÈGLES D'OCTROI DE CRÉDIT (AFDCC & SEUIL BENCHMARK)
# -----------------------------------------------------------------------------
def calculer_arbitrage_credit(
    donnees: dict, proba: float, montant_demande: float
) -> dict:
  """Applique la grille d'arbitrage de crédit consolidée BTP."""
  has_bilan = int(donnees.get("has_bilan_depose") or 0)
  ca = float(donnees.get("chiffre_affaires") or 0.0)
  fp = float(donnees.get("capitaux_propres") or 0.0)
  bodacc_actif = bool(
      donnees.get("bodacc_actif")
      or (donnees.get("nature_jugement") not in ["AUCUN_INCIDENT", None])
  )

  # 1. Règle NOIR : procédure collective active
  if bodacc_actif:
        return {
            "statut": "NOIR",
            "proba_defaut": proba,
            "plafond_autorise": 0.0,
            "montant_accorde": 0.0,
            "motif": (
                "Procédure collective active au BODACC : compte bloqué et refus"
                " formel de crédit."
            ),
        }        
  # 2. Règle BLOCAGE BILAN : capitaux propres négatifs ou nuls
  elif has_bilan == 1 and fp <= 0:
        return {
            "statut": "BLOCAGE_BILAN",
            "proba_defaut": proba,
            "plafond_autorise": 0.0,
            "montant_accorde": 0.0,
            "motif": (
                f"Capitaux propres négatifs ou nuls ({fp:,.0f} €) : encours bloqué"
                " pour sous-capitalisation."
            ),
        }
        print("BLOCAGE_BILAN",fp)
  
  # 3. Règle ROUGE : risque statistique critique
  elif proba > 0.30:
            return {
            "statut": "ROUGE",
            "proba_defaut": proba,
            "plafond_autorise": 0.0,
            "montant_accorde": 0.0,
            "motif": (
                f"Risque critique de défaut ({proba * 100:.1f} % > 30 %) : aucun"
                " encours non garanti. Règlement comptant exigé."
            ),
    }
  elif 0.15< proba < 30:
        if has_bilan == 1 and ca >= 10000:
            capacite_nominale = min(ca * 0.07, fp * 0.25)
            plafond_ajuste = 0.75 * capacite_nominale
            montant_final = min(montant_demande, plafond_ajuste)
            motif_detail = "décote de 25 % sur capacité d'absorption"
  # 4. Détermination de la capacité d'absorption maximale nominale (Base 100 %)
        elif has_bilan == 1 and ca <10000:
            capacite_nominale = min(5000.0, fp * 0.25)  
            plafond_ajuste = 0.75 * capacite_nominale
            montant_final = min(montant_demande, plafond_ajuste)
            motif_detail = "décote de 25 % sur capacité d'absorption"
        elif has_bilan == 0:
            capacite_nominale = 5000
            plafond_ajuste = 0.5 *capacite_nominale
            montant_final = min(montant_demande, plafond_ajuste)
            motif_detail = "forfait de confiance avec abattement de 50 %"
        return {
            "statut": "ORANGE",
            "proba_defaut": proba,
            "plafond_autorise": round(plafond_ajuste, 2),
            "montant_accorde": round(montant_final, 2),
            "motif": (
                f"Risque modéré ({proba * 100:.1f} %) : {motif_detail}. Délais"
                " réduits à 30 jours ou acompte partiel."
            ),
            }

  elif proba <= 0.15:
        if has_bilan == 1 and ca >= 10000:
            capacite_nominale = min(ca * 0.07, fp * 0.25)
            plafond_ajuste = capacite_nominale
            montant_final = min(montant_demande, plafond_ajuste)
            motif_detail = "100 % de la capacité validée"
      # 4. Détermination de la capacité d'absorption maximale nominale (Base 100 %)
        elif has_bilan == 1 and ca <10000:
            capacite_nominale = min(5000.0, fp * 0.25)  
            plafond_ajuste = capacite_nominale
            montant_final = min(montant_demande, plafond_ajuste)
            motif_detail = "forfait de confiance car chiffre d'affaires insuffisant"

        elif has_bilan == 0:
            capacite_nominale = 5000
            plafond_ajuste = 0.8 *capacite_nominale
            montant_final = min(montant_demande, plafond_ajuste)
            motif_detail = "forfait de confiance abattement de 20% "
        montant_final = min(montant_demande, plafond_ajuste)
        return {
          "statut": "VERT",
          "proba_defaut": proba,
          "plafond_autorise": round(plafond_ajuste, 2),
          "montant_accorde": round(montant_final, 2),
          "motif": (
            f"Risque faible ({proba * 100:.1f} %) : {motif_detail}.\n"
            "Conditions commerciales standards (30-60 jours)."
            ),
          }
        
# -----------------------------------------------------------------------------
# ROUTE PRINCIPALE DE CONSULTATION & ARBITRAGE
# -----------------------------------------------------------------------------
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

      # Inférence ou court-circuit si incident déjà acté en base
      if cible == 1:
        proba = 1.0
      else:
        proba = predire_risque_xgboost(donnees_entreprise)

      arbitrage = calculer_arbitrage_credit(
          donnees_entreprise, proba, montant_demande
      )

      # Journalisation en arrière-plan dans PostgreSQL
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

      # Injection de la décision dans la réponse retournée
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