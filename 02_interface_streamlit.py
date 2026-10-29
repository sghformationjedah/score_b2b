import os
import time
import pandas as pd
import requests
import streamlit as st
import re
import logging

logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="BTP CreditScore",
    page_icon="🏢",
    layout="wide",
)

# 1. Dictionnaires de correspondance (à placer en haut du fichier ou avant la boucle)
DEPT_LABELS = {
    # France Métropolitaine
    "01": "Ain",
    "02": "Aisne",
    "03": "Allier",
    "04": "Alpes-de-Haute-Provence",
    "05": "Hautes-Alpes",
    "06": "Alpes-Maritimes",
    "07": "Ardèche",
    "08": "Ardennes",
    "09": "Ariège",
    "10": "Aube",
    "11": "Aude",
    "12": "Aveyron",
    "13": "Bouches-du-Rhône",
    "14": "Calvados",
    "15": "Cantal",
    "16": "Charente",
    "17": "Charente-Maritime",
    "18": "Cher",
    "19": "Corrèze",
    "21": "Côte-d'Or",
    "22": "Côtes-d'Armor",
    "23": "Creuse",
    "24": "Dordogne",
    "25": "Doubs",
    "26": "Drôme",
    "27": "Eure",
    "28": "Eure-et-Loir",
    "29": "Finistère",
    "2A": "Corse-du-Sud",
    "2B": "Haute-Corse",
    "30": "Gard",
    "31": "Haute-Garonne",
    "32": "Gers",
    "33": "Gironde",
    "34": "Hérault",
    "35": "Ille-et-Vilaine",
    "36": "Indre",
    "37": "Indre-et-Loire",
    "38": "Isère",
    "39": "Jura",
    "40": "Landes",
    "41": "Loir-et-Cher",
    "42": "Loire",
    "43": "Haute-Loire",
    "44": "Loire-Atlantique",
    "45": "Loiret",
    "46": "Lot",
    "47": "Lot-et-Garonne",
    "48": "Lozère",
    "49": "Maine-et-Loire",
    "50": "Manche",
    "51": "Marne",
    "52": "Haute-Marne",
    "53": "Mayenne",
    "54": "Meurthe-et-Moselle",
    "55": "Meuse",
    "56": "Morbihan",
    "57": "Moselle",
    "58": "Nièvre",
    "59": "Nord",
    "60": "Oise",
    "61": "Orne",
    "62": "Pas-de-Calais",
    "63": "Puy-de-Dôme",
    "64": "Pyrénées-Atlantiques",
    "65": "Hautes-Pyrénées",
    "66": "Pyrénées-Orientales",
    "67": "Bas-Rhin",
    "68": "Haut-Rhin",
    "69": "Rhône",
    "70": "Haute-Saône",
    "71": "Saône-et-Loire",
    "72": "Sarthe",
    "73": "Savoie",
    "74": "Haute-Savoie",
    "75": "Paris",
    "76": "Seine-Maritime",
    "77": "Seine-et-Marne",
    "78": "Yvelines",
    "79": "Deux-Sèvres",
    "80": "Somme",
    "81": "Tarn",
    "82": "Tarn-et-Garonne",
    "83": "Var",
    "84": "Vaucluse",
    "85": "Vendée",
    "86": "Vienne",
    "87": "Haute-Vienne",
    "88": "Vosges",
    "89": "Yonne",
    "90": "Territoire de Belfort",
    "91": "Essonne",
    "92": "Hauts-de-Seine",
    "93": "Seine-Saint-Denis",
    "94": "Val-de-Marne",
    "95": "Val-d'Oise",
    # DROM (Départements et Régions d'Outre-Mer)
    "971": "Guadeloupe",
    "972": "Martinique",
    "973": "Guyane",
    "974": "La Réunion",
    "976": "Mayotte",
}

TRANCHE_LABELS = {
    # Nomenclature officielle INSEE / SIRENE
    "NN": "Non renseigné",
    "00": "0 salarié",
    "01": "1 ou 2 salariés",
    "02": "3 à 5 salariés",
    "03": "6 à 9 salariés",
    "11": "10 à 19 salariés",
    "12": "20 à 49 salariés",
    "21": "50 à 99 salariés",
    "22": "100 à 199 salariés",
    "31": "200 à 249 salariés",
    "32": "250 à 499 salariés",
    "41": "500 à 999 salariés",
    "42": "1 000 à 1 999 salariés",
    "51": "2 000 à 4 999 salariés",
    "52": "5 000 à 9 999 salariés",
    "53": "10 000 salariés et plus",
}

NAF_43 = {
    # 43.1 - Démolition et préparation des sites
    "43.11Z": "Travaux de démolition",
    "43.12A": "Travaux de terrassement courants et travaux préparatoires",
    "43.12B": "Travaux de terrassement spécialisés ou de grande masse",
    "43.13Z": "Forages et sondages",
    # 43.2 - Travaux d'installation électrique, plomberie et autres travaux d'installation
    "43.21A": "Travaux d'installation électrique dans tous locaux",
    "43.21B": "Travaux d'installation électrique sur la voie publique",
    "43.22A": "Travaux d'installation d'eau et de gaz en tous locaux",
    "43.22B": "Travaux d'installation d'équipements thermiques et de climatisation",
    "43.29A": "Travaux d'isolation",
    "43.29B": "Autres travaux d'installation n.c.a.",
    # 43.3 - Travaux de finition
    "43.31Z": "Travaux de plâtrerie",
    "43.32A": "Travaux de menuiserie bois et PVC",
    "43.32B": "Travaux de menuiserie métallique et serrurerie",
    "43.32C": "Agencement de lieux de vente",
    "43.33Z": "Travaux de revêtement des sols et des murs",
    "43.34Z": "Travaux de peinture et vitrerie",
    "43.39Z": "Autres travaux de finition",
    # 43.9 - Autres travaux de construction spécialisés
    "43.91A": "Travaux de charpente",
    "43.91B": "Travaux de couverture par éléments",
    "43.99A": "Travaux d'étanchéification",
    "43.99B": "Travaux de montage d'échafaudages et d'étaiements",
    "43.99C": "Travaux de maçonnerie générale et gros œuvre de bâtiment",
    "43.99D": "Travaux de montage d'ossatures métalliques",
    "43.99E": "Travaux de fonçage et de fonçage de puits",
    "43.99Z": "Autres travaux de construction spécialisés n.c.a.",
}

# -----------------------------------------------------------------------------
# DECLARATION VARIABLES GLOBALES
# -----------------------------------------------------------------------------


pattern_naf = r"\b(?:code\s+)?(?:naf|ape)\b"
pattern_effectif = r"\b(?:tranche\s+(?:d['’])?)?effectifs?\b|\btranche\b"

# -----------------------------------------------------------------------------
# CONFIGURATION DE L'URL API
# -----------------------------------------------------------------------------
raw_url = os.getenv("API_URL", "http://127.0.0.1:8000")

if "127.0.0.1" in raw_url or "localhost" in raw_url:
  API_URL = raw_url
elif not raw_url.startswith("http"):
  if not raw_url.endswith(".onrender.com"):
    API_URL = f"https://{raw_url}.onrender.com"
  else:
    API_URL = f"https://{raw_url}"
else:
  API_URL = raw_url

API_URL = API_URL.rstrip("/")


# -----------------------------------------------------------------------------
# EN-TÊTE CENTRÉ
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div style="text-align: center;">
        <h1 style="margin-bottom: 0px;">🏢 BTP CreditScore</h1>
        <p style="color: #6c757d; font-size: 1.1rem; margin-top: 5px;">
            Système d'évaluation du risque de défaillance & Aide à l'octroi de crédit B2B
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")

# -----------------------------------------------------------------------------
# HEALTHCHECK DE L'API
# -----------------------------------------------------------------------------
status_placeholder = st.empty()


def wake_up_backend(api_url: str, timeout_seconds: int = 70) -> bool:
  if st.session_state.get("backend_ready"):
    return True

  health_endpoint = f"{api_url.rstrip('/')}/health"
  start_time = time.time()

  with st.spinner("⏳ Vérification de l'état du serveur..."):
    while time.time() - start_time < timeout_seconds:
      try:
        response = requests.get(health_endpoint, timeout=5)
        if response.status_code in (200, 429):
          st.session_state["backend_ready"] = True
          return True
      except requests.RequestException:
        pass
      time.sleep(5)

  return False


if not wake_up_backend(API_URL):
  status_placeholder.error(
      f"L'API ({API_URL}) n'a pas répondu à temps. Vérifie l'URL et les logs du"
      " backend."
  )
  st.stop()
else:
  status_placeholder.empty()


# -----------------------------------------------------------------------------
# RÉFÉRENTIEL CLIENTS (AUTOCOMPLÉTION SANS CACHE BLOQUANT)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=1800, show_spinner=False)
def charger_liste_siren(api_url: str):
  endpoint = f"{api_url.rstrip('/')}/entreprises/liste-siren"
  try:
    # Timeout augmenté : (connexion, lecture)
    resp = requests.get(endpoint, timeout=(5, 30))

    if resp.status_code == 200:
      data = resp.json()
      if isinstance(data, list) and len(data) > 0:
        return data
      logger.warning("Réponse vide ou format inattendu reçue de l'API.")
    else:
      logger.error(
          f"Erreur API ({resp.status_code}) sur {endpoint} : {resp.text}"
      )

  except requests.exceptions.Timeout:
    logger.error(
        f"Timeout dépassé lors du chargement des SIREN depuis {endpoint}"
    )
  except requests.exceptions.RequestException as e:
    logger.error(f"Erreur réseau lors de l'appel à {endpoint} : {e}")
  except Exception as e:
    logger.error(f"Erreur inattendue : {e}")

  return []


# -----------------------------------------------------------------------------
# ONGLETS DE NAVIGATION
# -----------------------------------------------------------------------------
tab_analyse, tab_historique = st.tabs(
    ["⚡ Analyse du risque", "📋 Historique des décisions"]
)

# =============================================================================
# ONGLET 1 : ANALYSE DU RISQUE (SCORING)
# =============================================================================
with tab_analyse:
  entreprises_dispos = charger_liste_siren(API_URL)

  # Construction du dictionnaire pour le selectbox
  options_siren = {}
  for e in entreprises_dispos:
    s_val = str(e.get("siren", "")).strip()
    n_val = str(e.get("nom", "Inconnu")).strip()
    if s_val:
      options_siren[f"{s_val} — {n_val}"] = s_val

  with st.container(border=True):
    st.subheader("🔍 Évaluation d'une demande")

    col_siren, col_montant = st.columns([1.3, 1])

    with col_siren:
      if options_siren:
        liste_choix = [""] + list(options_siren.keys())
        choix = st.selectbox(
            "Rechercher un client (Tapez le SIREN ou le Nom)",
            options=liste_choix,
            format_func=lambda x: (
                "-- Sélectionner ou taper un SIREN / Nom --" if x == "" else x
            ),
            index=0,
            key="select_siren_client",
        )
        siren_input = options_siren.get(choix, "")
      else:
        # Saisie manuelle directe si la base n'a pas encore répondu
        siren_input = st.text_input(
            "Numéro SIREN (9 chiffres)",
            placeholder="Ex : 443061841",
            max_chars=9,
            key="siren_manuel_input",
        )

    with col_montant:
      montant_demande = st.number_input(
          "Montant demandé (€)",
          min_value=0.0,
          value=15000.0,
          step=1000.0,
          format="%.2f",
          help="Encours ou ligne de crédit sollicitée au comptoir.",
          key="montant_scoring_input",
      )

    st.write("")
    btn_chercher = st.button(
        "⚡ Lancer l'analyse du risque",
        type="primary",
        use_container_width=True,
        key="btn_scoring",
    )

  if btn_chercher:
    siren_nettoye = siren_input.strip() if siren_input else ""
    if (
        not siren_nettoye
        or len(siren_nettoye) != 9
        or not siren_nettoye.isdigit()
    ):
      st.warning(
          "⚠️ Veuillez sélectionner ou saisir un numéro SIREN valide composé de"
          " 9 chiffres."
      )
    else:
      with st.spinner("Interrogation de la base et inférence XGBoost..."):
        try:
          url = f"{API_URL}/entreprises/{siren_nettoye}"
          params = {"montant_demande": montant_demande}
          response = requests.get(url, params=params, timeout=10)

          if response.status_code == 200:
            data = response.json()
            decision = data.pop("decision_scoring", {})
            statut = decision.get("statut", "INCONNU")
            proba_pct = decision.get("proba_defaut", 0.0) * 100
            accord = decision.get("montant_accorde", 0.0)
            plafond = decision.get("plafond_autorise", 0.0)
            motif = decision.get("motif", "")
            top_raisons = decision.get("top_raisons", [])

            st.success("✅ Entreprise identifiée et scorée avec succès !")

            # Bandeau haut : Infos clés & Score XGBoost
            col_infos_base, col_score_droite = st.columns([2.5, 1.5])

            with col_infos_base:
              st.markdown(
                  f"### 📋 {data.get('nom_commercial') or 'Raison sociale non renseignée'}"
              )
              code_naf = data.get('code_naf')
              nom_lower = code_naf.lower()      
              match = re.search(r"\b(\d{2})\.?(\d{2})([a-z])\b", nom_lower, re.IGNORECASE)
              code_naf = f"{match.group(1)}.{match.group(2)}{match.group(3).upper()}"
              libelle = NAF_43.get(code_naf, "Code NAF inconnu")
              nom_affiche = f"NAF : {code_naf} - {libelle}"
              st.caption(
                  f"SIREN : **{data.get('siren')}** | Code NAF :"
                  f" **{nom_affiche}** | Activité : Division 43"
              )

              m1, m2, m3 = st.columns(3)
              ca_val = data.get("chiffre_affaires")
              try:
                if ca_val is not None and str(ca_val).strip() != "":
                  ca_format = f"{float(ca_val):,.0f} €".replace(",", " ")
                else:
                  ca_format = "Non déclaré"
              except (ValueError, TypeError):
                ca_format = str(ca_val)

              m1.metric("Chiffre d'Affaires", ca_format)
              m2.metric(
                  "Statut BODACC",
                  data.get("nature_jugement") or "AUCUN_INCIDENT",
              )
              m3.metric(
                  "Bilan Déposé",
                  "Oui" if data.get("has_bilan_depose") == 1 else "Non",
              )

            with col_score_droite:
              couleurs_badge = {
                  "VERT": "#10b981",
                  "ORANGE": "#f59e0b",
                  "ROUGE": "#ef4444",
                  "NOIR": "#1e293b",
                  "BLOCAGE_BILAN": "#ef4444",
              }
              couleur = couleurs_badge.get(statut, "#64748b")

              st.markdown(
                f"""
                <div style="background-color: var(--secondary-background-color); border: 2px solid {couleur}; border-radius: 10px; padding: 15px; text-align: center;">
                    <span style="background-color: {couleur}; color: white; padding: 4px 10px; border-radius: 6px; font-weight: bold; font-size: 0.85rem;">
                        DÉCISION : {statut}
                    </span>
                    <h2 style="margin: 10px 0 5px 0; color: {couleur}; font-size: 2rem;">
                        {proba_pct:.1f} %
                    </h2>
                    <p style="margin: 0; color: var(--text-color); opacity: 0.75; font-size: 0.85rem; font-weight: 600;">
                        Probabilité de défaillance (XGBoost)
                    </p>
                    <hr style="margin: 10px 0; opacity: 0.25;">
                    <p style="margin: 0; font-size: 0.95rem; color: var(--text-color);">
                        <b>Montant Demandé :</b> <span style="font-weight: bold;">{montant_demande:,.0f} €</span>
                        <br>
                        <b>Montant Accordé :</b> <span style="font-weight: bold;">{accord:,.0f} €</span>
                        <span style="opacity: 0.65;">(Plafond : {plafond:,.0f} €)</span>
                    </p>
                </div>
                <br>
                """,
                unsafe_allow_html=True, 
            )

            # Bandeau recommandation & Top raisons
            st.info(f"💡 **Recommandation du moteur d'octroi :** {motif}")

            if top_raisons:
                #st.write("Contenu brut de top_raisons :", top_raisons)
                #st.write("Type :", type(r.get("valeur")), "Valeur :", r.get("valeur"))
                st.markdown(
                    "##### 🔍 **Facteurs clés ayant motivé la décision :**"
                )
                cols_raisons = st.columns(len(top_raisons))
                for idx, r in enumerate(top_raisons):
                    with cols_raisons[idx]:
                        est_favorable = r.get("impact") == "favorable"
                        icone = "🟢" if est_favorable else "🔴"
                        couleur_cadre = "#d1fae5" if est_favorable else "#fee2e2"
                        couleur_texte = "#065f46" if est_favorable else "#991b1b"
                        val_texte = ""
                        # 1. Récupération robuste du nom de la variable
                        nom_variable = (
                            r.get("variable")
                            or r.get("feature")
                            or r.get("nom")
                            or f"Critère {idx+1}"
                        )
                        val_brute = r.get("valeur")                                                
                        # 2. Traduction des valeurs selon le type de variable                        
                        if val_brute is not None:
                            nom_lower = nom_variable.lower()               
                            pattern_dep = r"\bd[eé]p(?:artement|\.|\b|t\.?)"
                            if re.search(pattern_dep, nom_lower, re.IGNORECASE):
                                match = re.search(r"\b(2[AB]|\d{1,3})\b", nom_lower, re.IGNORECASE)                                
                                if match:
                                    code_dept = match.group(1).zfill(2).upper()
                                    nom_affiche = f"Département : {DEPT_LABELS.get(code_dept, code_dept)}"
                                else:                                    
                                    nom_affiche = nom_variable
                                # Pour une variable binaire One-Hot, on n'affiche pas la valeur " (1)"
                                val_texte = ""
                            elif re.search(pattern_naf, nom_lower, re.IGNORECASE):
                                match = re.search(r"\b(\d{2})\.?(\d{2})([a-z])\b", nom_lower, re.IGNORECASE)
                                if match:
                                  code_naf = f"{match.group(1)}.{match.group(2)}{match.group(3).upper()}"
                                  libelle = NAF_43.get(code_naf, "Code NAF inconnu")
                                  nom_affiche = f"NAF : {code_naf} - {libelle}"
                                else:
                                  nom_affiche = "Code NAF non reconnu"                
                            elif re.search(pattern_effectif, nom_lower, re.IGNORECASE):
                                # Capture 1 ou 2 chiffres isolés (ex: "1", "01", "12") ou le code "NN"
                                match = re.search(r"\b(\d{1,2}|NN)\b", nom_lower, re.IGNORECASE)
                                if match:
                                    val = match.group(1).upper()
                                    # Ajoute le zéro initial si 1 seul chiffre (ex: "1" -> "01")
                                    code_tranche = val.zfill(2) if val.isdigit() else val
                                    libelle = TRANCHE_LABELS.get(code_tranche, "Tranche inconnue")
                                    nom_affiche = f"Effectif : {code_tranche} ({libelle})"
                                else:
                                    nom_affiche = "Code tranche effectif non reconnu"
                            else:
                                nom_affiche = nom_variable
                                if isinstance(val_brute, float):
                                    valeur_affichee = f"{val_brute:,.2f}".replace(",", " ")
                                else:
                                    valeur_affichee = str(val_brute)
                                val_texte = f" ({valeur_affichee})"
                        else:
                            nom_affiche = nom_variable
                            val_texte = ""
                        # 3. Rendu HTML
                        #st.write("nom_affiche",nom_affiche)
                        st.markdown(
                            f"""
                            <div style="background-color: {couleur_cadre}; color: {couleur_texte}; padding: 10px 12px; border-radius: 8px; font-size: 0.88rem; border: 1px solid {couleur_texte}30; min-height: 75px;">
                                <b>{icone} Raison {idx+1} : {nom_affiche}</b>{val_texte}<br>
                                <span style="font-size: 0.82rem;">{r.get('description', '')}</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            st.divider()

            # Détail  complet des variables
            st.markdown("### 📑 Détail complet des variables de la base")
            with st.expander("👁️ Voir les variables de la base"):
              df_vertical = pd.DataFrame(
                  list(data.items()), columns=["Indicateur / Colonne", "Valeur"]
              )
              st.dataframe(df_vertical, use_container_width=True, height=450)

          elif response.status_code == 404:
            st.error(f"❌ SIREN {siren_nettoye} non trouvé dans la base.")
          else:
            detail = response.json().get("detail", response.text)
            st.error(f"❌ Erreur {response.status_code} : {detail}")

        except requests.exceptions.ConnectionError:
          st.error(
              f"❌ Impossible de joindre l'API sur {API_URL}. Vérifiez"
              " qu'Uvicorn est bien démarré."
          )
        except Exception as err:
          st.error(f"❌ Une erreur imprévue est survenue : {err}")

# =============================================================================
# ONGLET 2 : HISTORIQUE DES DÉCISIONS
# =============================================================================
with tab_historique:
  with st.container(border=True):
    st.subheader("📚 Consultation des arbitrages passés")

    col_siren_hist, col_statut_hist, col_limit = st.columns([2, 1.5, 1])

    with col_siren_hist:
      siren_hist = st.text_input(
          "Filtrer par SIREN (optionnel)",
          placeholder="Ex : 443061841",
          max_chars=9,
          key="siren_historique_input",
      )

    with col_statut_hist:
      statut_hist = st.selectbox(
          "Statut de la décision",
          ["TOUS", "VERT", "ORANGE", "ROUGE", "NOIR", "BLOCAGE_BILAN"],
          key="statut_historique_select",
      )

    with col_limit:
      limit_hist = st.number_input(
          "Nombre de lignes",
          min_value=5,
          max_value=100,
          value=20,
          step=5,
          key="limit_historique_input",
      )

    st.write("")
    btn_charger_hist = st.button(
        "🔍 Rechercher dans l'historique",
        type="primary",
        use_container_width=True,
        key="btn_historique",
    )

  if btn_charger_hist:
    with st.spinner("Récupération de l'historique des décisions..."):
      try:
        url_hist = f"{API_URL}/decisions/historique"
        params_hist = {
            "limit": limit_hist,
            "statut": statut_hist if statut_hist != "TOUS" else None,
            "siren": siren_hist.strip() if siren_hist.strip() else None,
        }
        params_hist = {k: v for k, v in params_hist.items() if v is not None}

        resp_hist = requests.get(url_hist, params=params_hist, timeout=10)

        if resp_hist.status_code == 200:
          data_hist = resp_hist.json()

          if not data_hist:
            st.info("Aucun historique trouvé pour ces critères de recherche.")
          else:
            df_history = pd.DataFrame(data_hist)

            if "montant_demande" in df_history.columns:
              df_history["montant_demande"] = df_history[
                  "montant_demande"
              ].apply(
                  lambda x: (
                      f"{float(x):,.0f} €".replace(",", " ")
                      if pd.notnull(x)
                      else "-"
                  )
              )
            if "montant_accorde" in df_history.columns:
              df_history["montant_accorde"] = df_history[
                  "montant_accorde"
              ].apply(
                  lambda x: (
                      f"{float(x):,.0f} €".replace(",", " ")
                      if pd.notnull(x)
                      else "-"
                  )
              )
            if "proba_defaut" in df_history.columns:
              df_history["proba_defaut"] = df_history["proba_defaut"].apply(
                  lambda x: (
                      f"{float(x)*100:.1f} %" if pd.notnull(x) else "-"
                  )
              )

            st.dataframe(df_history, use_container_width=True, hide_index=True)
        elif resp_hist.status_code == 404:
          st.info("Aucune décision enregistrée dans l'historique.")
        else:
          err_detail = resp_hist.json().get("detail", resp_hist.text)
          st.error(f"❌ Erreur {resp_hist.status_code} : {err_detail}")

      except requests.exceptions.ConnectionError:
        st.error(
            f"❌ Impossible de joindre l'API sur {API_URL}. Vérifiez"
            " qu'Uvicorn est bien démarré."
        )
      except Exception as exc:
        st.error(f"❌ Erreur lors de la récupération : {exc}")