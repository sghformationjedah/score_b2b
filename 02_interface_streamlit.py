import os
import time
import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="BTP CreditScore",
    page_icon="🏢",
    layout="wide",
)

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
  try:
    resp = requests.get(f"{api_url}/entreprises/liste-siren", timeout=5)
    if resp.status_code == 200:
      data = resp.json()
      if isinstance(data, list) and len(data) > 0:
        return data
  except Exception:
    pass
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
              st.caption(
                  f"SIREN : **{data.get('siren')}** | Code NAF :"
                  f" **{data.get('code_naf')}** | Activité : Division 43"
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
                  val_texte = (
                      f" ({r['valeur']})" if r.get("valeur") is not None else ""
                  )

                  st.markdown(
                      f"""
                      <div style="background-color: {couleur_cadre}; color: {couleur_texte}; padding: 10px 12px; border-radius: 8px; font-size: 0.88rem; border: 1px solid {couleur_texte}30; min-height: 75px;">
                          <b>{icone} Raison {idx+1} : {r.get('variable')}</b>{val_texte}<br>
                          <span style="font-size: 0.82rem;">{r.get('description')}</span>
                      </div>
                      """,
                      unsafe_allow_html=True,
                  )

            st.divider()

            # Détail complet des variables
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