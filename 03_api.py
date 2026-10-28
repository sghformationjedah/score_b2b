import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    version="1.3.1",
)


@app.get("/health")
def health_check():
  return {"status": "ok"}


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
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "modele_1_xgboost_300_trees.joblib"

print("MODEL_PATH", MODEL_PATH)
model = None
explainer = None
feature_names = []

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
    "anciennete_bilan_ans": "Fraîcheur des données comptables",
    "tranche_effectif": "Tranche d'effectif",
    "departement": "Département",
    "code_naf": "Activité (Code NAF)",
}

COLONNES_BILAN = [
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
    "anciennete_bilan_ans",
]

COLONNES_NUM_ADMIN = [
    "age_entreprise_ans",
    "has_bilan_depose",
]

if os.path.exists(MODEL_PATH):
  try:
    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    feature_names = artifact["feature_names"]
    explainer = shap.TreeExplainer(model)
    print(
        f"✅ Modèle XGBoost et TreeSHAP initialisés ({len(feature_names)}"
        " features)."
    )
  except Exception as e:
    print(f"⚠️ Erreur lors du chargement de l'artefact : {e}")
else:
  print(f"⚠️ Artefact introuvable sur le chemin : {MODEL_PATH}")


def get_db_connection():
  """Établit la connexion avec la base de données PostgreSQL."""
  db_url = os.getenv(
      "DATABASE_URL",
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
def predire_risque_et_raisons(
    entreprise_dict: dict,
) -> Tuple[float, List[dict]]:
  if model is None:
    return 0.0, []

  df_row = pd.DataFrame([entreprise_dict])

  has_bilan = int(entreprise_dict.get("has_bilan_depose") or 0)
  df_row["tranche_effectif"] = df_row["tranche_effectif"].replace("NN", "0")

  date_exec = entreprise_dict.get("date_exercice")
  if has_bilan == 1 and pd.notna(date_exec):
    date_cloture = pd.to_datetime(date_exec, errors="coerce")
    df_row["anciennete_bilan_ans"] = (
        pd.Timestamp.now() - date_cloture
    ).days / 365.25
  else:
    df_row["anciennete_bilan_ans"] = np.nan

  colonnes_a_exclure = [
      "siren",
      "nom_commercial",
      "cible_defaillance",
      "nature_jugement",
      "date_jugement",
      "date_exercice",
      "date_creation",
  ]
  features = df_row.drop(
      columns=[c for c in colonnes_a_exclure if c in df_row.columns]
  )

  toutes_colonnes_num = COLONNES_BILAN + COLONNES_NUM_ADMIN
  for col in toutes_colonnes_num:
    if col in features.columns:
      features[col] = pd.to_numeric(
          features[col].replace(["", "NULL", "None", "nan", None], np.nan),
          errors="coerce",
      )

  if has_bilan == 0:
    for col in COLONNES_BILAN:
      if col in features.columns:
        features[col] = np.nan
  else:
    ca_val = features.get("chiffre_affaires", pd.Series([None])).iloc[0]
    if pd.notna(ca_val) and ca_val <= 0:
      for col_ca in [
          "chiffre_affaires",
          "resultat_net",
          "marge_nette",
          "tresorerie_jours_ca",
      ]:
        if col_ca in features.columns:
          features[col_ca] = np.nan

      for col in COLONNES_BILAN:
        if (
            col in features.columns
            and pd.notna(features[col].iloc[0])
            and features[col].iloc[0] == 0.0
        ):
          features[col] = np.nan

  cat = ["code_naf", "tranche_effectif", "departement"]
  cat_presentes = [c for c in cat if c in features.columns]

  features_cat = features.copy()
  for col_c in cat_presentes:
    features_cat[col_c] = features_cat[col_c].astype(str).str.strip()

  features_encoded = pd.get_dummies(
      features_cat, columns=cat_presentes, drop_first=False, dtype=int
  )
  features_encoded.columns = features_encoded.columns.str.replace(
      r"[\[\]<]", "_", regex=True
  )

  features_aligned = features_encoded.reindex(
      columns=feature_names, fill_value=0
  )

  proba = float(model.predict_proba(features_aligned)[0, 1])

  top_raisons = []
  try:
    if explainer is not None:
      shap_vals = explainer(features_aligned).values[0]
      if len(shap_vals.shape) == 2:
        shap_vals = shap_vals[:, 1]

      indices_candidats = []
      for idx, nom_col in enumerate(feature_names):
        val_reelle = features_aligned.iloc[0, idx]
        if pd.isna(val_reelle):
          continue

        est_binaire_ohe = any(
            nom_col.startswith(p)
            for p in ["tranche_effectif_", "code_naf_", "departement_"]
        )
        if est_binaire_ohe and val_reelle == 0:
          continue
        indices_candidats.append(idx)

      indices_candidats = sorted(
          indices_candidats, key=lambda i: abs(shap_vals[i]), reverse=True
      )
      top_indices = indices_candidats[:3]
      total_abs_top = sum(abs(float(shap_vals[i])) for i in top_indices)

      for idx in top_indices:
        nom_col = feature_names[idx]
        val_shap = float(shap_vals[idx])
        val_reelle = features_aligned.iloc[0, idx]

        shap_formatte = f"SHAP: {val_shap:+.4f}"
        poids_pct = (
            round((abs(val_shap) / total_abs_top) * 100, 1)
            if total_abs_top > 0
            else 0.0
        )

        nom_clair = DICTIONNAIRE_LABELS.get(
            nom_col, nom_col.replace("_", " ").capitalize()
        )
        favorable = val_shap < 0

        if any(
            nom_col.startswith(p)
            for p in ["tranche_effectif_", "code_naf_", "departement_"]
        ):
          valeur_affichee = "Oui"
          sens = (
              "profil favorable renforçant la solvabilité"
              if favorable
              else "segment ou taille à risque accru"
          )
        else:
          valeur_affichee = round(float(val_reelle), 2)
          sens = (
              "impact positif renforçant la solvabilité"
              if favorable
              else "fragilité augmentant le risque de défaut"
          )

        description = (
            f"{nom_clair} : {sens} ({shap_formatte} | {poids_pct} % du signal)"
        )

        top_raisons.append({
            "variable": nom_clair,
            "valeur": valeur_affichee,
            "impact": "favorable" if favorable else "defavorable",
            "shap_str": shap_formatte,
            "val_shap": round(val_shap, 4),
            "poids_pct": poids_pct,
            "description": description,
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
  bodacc_actif = int(donnees.get("cible_defaillance") or 0) == 1

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
        "top_raisons": [
            {
                "variable": "BODACC",
                "impact": "defavorable",
                "description": f"Incident judiciaire critique : {nature_jug}",
            },
            {
                "variable": "Cadre légal",
                "impact": "defavorable",
                "description": (
                    "Interdiction d'ouverture de nouvel encours non garanti"
                ),
            },
            {
                "variable": "Solvabilité globale",
                "impact": "defavorable",
                "description": "Risque maximal d'impayé constaté",
            },
        ],
    }
  elif has_bilan == 1 and fp <= 0:
    return {
        "statut": "BLOCAGE_BILAN",
        "proba_defaut": proba,
        "plafond_autorise": 0.0,
        "montant_accorde": 0.0,
        "motif": (
            f"Capitaux propres négatifs ou nuls ({fp:,.0f} €) : encours bloqué"
            " pour sous-capitalisation. Règlement comptant exigé."
        ),
        "top_raisons": [
            {
                "variable": "Capitaux Propres",
                "impact": "defavorable",
                "description": f"Fonds propres négatifs ou nuls ({fp:,.0f} €)",
            },
            {
                "variable": "Assise financière",
                "impact": "defavorable",
                "description": "Incapacité d'absorption des pertes",
            },
            {
                "variable": "Règle prudentielle",
                "impact": "defavorable",
                "description": (
                    "Refus d'octroi systématique sur fonds propres négatifs"
                ),
            },
        ],
    }
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
        "top_raisons": top_raisons,
    }
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
        "motif": (
            f"Risque modéré ({proba * 100:.1f} %) : {motif_detail}. Délais"
            " réduits à 30 jours ou acompte partiel."
        ),
        "top_raisons": top_raisons,
    }
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
        "motif": (
            f"Risque faible ({proba * 100:.1f} %) : {motif_detail}. Conditions"
            " commerciales standards (30-60 jours)."
        ),
        "top_raisons": top_raisons,
    }


# -----------------------------------------------------------------------------
# ROUTE AUTOCOMPLÉTION : LISTE DES SIREN
# -----------------------------------------------------------------------------
@app.get(
    "/entreprises/liste-siren",
    response_model=List[Dict[str, str]],
    status_code=status.HTTP_200_OK,
)
def get_liste_siren(limit: int = 3000):
  """Renvoie les SIREN et raisons sociales avec cast textuel strict garanti."""
  conn = None
  try:
    conn = get_db_connection()
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
      cur.execute(
          """
                SELECT 
                    siren::text AS siren, 
                    COALESCE(nom_commercial, 'Raison non renseignée') AS nom 
                FROM entreprises_btp_solvabilite 
                ORDER BY siren ASC 
                LIMIT %s;
            """,
          (limit,),
      )
      records = cur.fetchall()
      # Normalisation stricte en chaînes de caractères pour Pydantic
      return [{"siren": str(r["siren"]), "nom": str(r["nom"])} for r in records]
  except Exception as e:
    print(f"⚠️ Erreur liste-siren : {e}")
    raise HTTPException(status_code=500, detail=str(e))
  finally:
    if conn:
      conn.close()

# -----------------------------------------------------------------------------
# ROUTE PRINCIPALE : SCORING D'UNE ENTREPRISE
# -----------------------------------------------------------------------------
@app.get(
    "/entreprises/{siren}",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
)
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
      # Cast explicite pour gérer indifféremment un type VARCHAR ou BIGINT en base
      query = (
          "SELECT * FROM entreprises_btp_solvabilite WHERE siren::text = %s;"
      )
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
        raisons = [{
            "variable": "Cible Défaillance",
            "impact": "defavorable",
            "description": "Défaillance déjà actée au dossier",
        }]
      else:
        proba, raisons = predire_risque_et_raisons(donnees_entreprise)

      arbitrage = calculer_arbitrage_credit(
          donnees_entreprise, proba, montant_demande, raisons
      )

      # Historisation asynchrone
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



# -----------------------------------------------------------------------------
# ROUTE HISTORIQUE : CONSULTATION DES DÉCISIONS PASSÉES
# -----------------------------------------------------------------------------
@app.get(
    "/decisions/historique",
    response_model=List[Dict[str, Any]],
    status_code=status.HTTP_200_OK,
)
def get_historique_decisions(
    limit: int = Query(20, ge=1, le=100),
    statut: Optional[str] = Query(None),
    siren: Optional[str] = Query(None),
):
  conn = None
  try:
    conn = get_db_connection()
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
      conditions = []
      params = []

      if siren:
        conditions.append("siren::text = %s")
        params.append(siren.strip())

      if statut:
        conditions.append("statut_decision = %s")
        params.append(statut.strip())

      where_clause = ""
      if conditions:
        where_clause = "WHERE " + " AND ".join(conditions)

      query = f"""
                SELECT 
                    id,
                    siren::text AS siren,
                    nom_commercial,
                    montant_demande,
                    proba_defaut,
                    statut_decision,
                    plafond_accorde AS montant_accorde,
                    motif_decision
                FROM historique_decisions
                {where_clause}
                ORDER BY id DESC
                LIMIT %s;
            """
      params.append(limit)

      cur.execute(query, tuple(params))
      records = cur.fetchall()
      return [dict(r) for r in records]

  except psycopg2.Error as e:
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=f"Erreur lors de la lecture de l'historique : {str(e)}",
    )
  finally:
    if conn:
      conn.close()