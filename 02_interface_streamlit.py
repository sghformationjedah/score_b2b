import os
import pandas as pd
import requests
import streamlit as st
import time

st.set_page_config(
    page_title="BTP CreditScore B2B — Recherche Entreprise",
    page_icon="🏢",
    layout="wide",
)

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

st.markdown(
    """
    <h2 style='text-align: center; margin-bottom: 25px;'>
        🏢 BTP CreditScore — Consultation des Entreprises
    </h2>
    """,
    unsafe_allow_html=True,
)

# Vérifie si l'API est opérationnelle ou en someil

def wake_up_backend(api_url: str, timeout_seconds: int = 70) -> bool:
    if st.session_state.get("backend_ready"):
        return True

    health_endpoint = f"{api_url}/health"
    start_time = time.time()

    with st.spinner(
        "⏳ Réveil du serveur backend Render en cours (environ 30 à 60s)..."
    ):
        while time.time() - start_time < timeout_seconds:
            try:
                response = requests.get(health_endpoint, timeout=5)
                if response.status_code == 200:
                    st.session_state["backend_ready"] = True
                    return True
            except requests.RequestException:
                pass
            time.sleep(3)

    return False


# 3. Lancement du test au chargement de l'UI
if not wake_up_backend(API_URL):
    st.error(
        "Le serveur backend met trop de temps à démarrer. Rafraîchis la page dans un instant."
    )
    st.stop()


# -----------------------------------------------------------------------------
# 1. ZONE DE SAISIE
# -----------------------------------------------------------------------------
col_saisie, _ = st.columns([1, 2])

with col_saisie:
    st.subheader("🔍 Recherche dans la base")
    siren_input = st.text_input(
        "Numéro SIREN :",
        max_chars=9,
        placeholder="Ex : 443061841",
        help="Saisissez les 9 chiffres du numéro SIREN",
    )

    montant_demande = st.number_input(
        "Montant demandé (€) :",
        min_value=0.0,
        value=15000.0,
        step=1000.0,
        format="%.2f",
        help="Saisissez l'encours commercial sollicité par le client en euros",
    )

    btn_chercher = st.button(
        "Rechercher la solvabilité", type="primary", use_container_width=True
    )

# -----------------------------------------------------------------------------
# 2. APPEL API ET AFFICHAGE DES RÉSULTATS
# -----------------------------------------------------------------------------
if btn_chercher:
    if not siren_input or len(siren_input.strip()) != 9:
        st.warning("⚠️ Veuillez saisir un numéro SIREN valide à 9 chiffres.")
    else:
        with st.spinner("Interrogation de la base et inférence XGBoost..."):
            try:
                url = f"{API_URL}/entreprises/{siren_input.strip()}"
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

                    # ---------------------------------------------------------
                    # BANDEAU HAUT : INFOS CLÉS (GAUCHE) & SCORE XGBOOST (DROITE)
                    # ---------------------------------------------------------
                    col_infos_base, col_score_droite = st.columns([2.5, 1.5])

                    with col_infos_base:
                        st.markdown(
                            f"### 📋 {data.get('nom_commercial') or 'Raison sociale non renseignée'}"
                        )
                        st.caption(
                            f"SIREN : **{data.get('siren')}** | Code NAF : **{data.get('code_naf')}** | Activité : Division 43"
                        )

                        # Métriques financières
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
                            (
                                "Oui"
                                if data.get("has_bilan_depose") == 1
                                else "Non"
                            ),
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
                            <div style="background-color: #f8fafc; border: 2px solid {couleur}; border-radius: 10px; padding: 15px; text-align: center;">
                                <span style="background-color: {couleur}; color: white; padding: 4px 10px; border-radius: 6px; font-weight: bold; font-size: 0.85rem;">
                                    DÉCISION : {statut}
                                </span>
                                <h2 style="margin: 10px 0 5px 0; color: {couleur}; font-size: 2rem;">
                                    {proba_pct:.1f} %
                                </h2>
                                <p style="margin: 0; color: #475569; font-size: 0.85rem; font-weight: 600;">
                                    Probabilité de défaillance (XGBoost)
                                </p>
                                <hr style="margin: 10px 0;">
                                <p style="margin: 0; font-size: 0.95rem;">
                                    <b>Montant Demandé :</b> <span style="color: #0f172a; font-weight: bold;">{montant_demande:,.0f} €</span>
                                    <br>
                                    <b>Montant Accordé :</b> <span style="color: #0f172a; font-weight: bold;">{accord:,.0f} €</span>
                                    <span style="color: #64748b;">(Plafond : {plafond:,.0f} €)</span>
                                </p>
                            </div>
                            <br> 
                            """,
                            unsafe_allow_html=True,
                        )

                    # ---------------------------------------------------------
                    # BANDEAU RECOMMANDATION + TOP 3 DES RAISONS
                    # ---------------------------------------------------------
                    st.info(f"💡 **Recommandation du moteur d'octroi :** {motif}")

                    if top_raisons:
                        st.markdown("##### 🔍 **Facteurs clés ayant motivé la décision :**")
                        cols_raisons = st.columns(len(top_raisons))
                        for idx, r in enumerate(top_raisons):
                            with cols_raisons[idx]:
                                est_favorable = r.get("impact") == "favorable"
                                icone = "🟢" if est_favorable else "🔴"
                                couleur_cadre = "#d1fae5" if est_favorable else "#fee2e2"
                                couleur_texte = "#065f46" if est_favorable else "#991b1b"
                                
                                val_texte = f" ({r['valeur']})" if r.get("valeur") is not None else ""
                                
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

                    # ---------------------------------------------------------
                    # AFFICHAGE 1 : Tableau vertical clé / valeur
                    # ---------------------------------------------------------
                    st.markdown("### 📑 Détail complet des variables de la base")

                    df_vertical = pd.DataFrame(
                        list(data.items()),
                        columns=["Indicateur / Colonne", "Valeur"],
                    )
                    st.dataframe(df_vertical, use_container_width=True, height=450)

                    # ---------------------------------------------------------
                    # AFFICHAGE 2 : Ligne brute de la table SQL
                    # ---------------------------------------------------------
                    with st.expander("👁️ Voir la ligne brute de la table SQL"):
                        df_ligne = pd.DataFrame([data])
                        st.dataframe(df_ligne, use_container_width=True)

                elif response.status_code == 404:
                    st.error(f"❌ SIREN {siren_input} non trouvé dans la base.")
                else:
                    detail = response.json().get("detail", response.text)
                    st.error(f"❌ Erreur {response.status_code} : {detail}")

            except requests.exceptions.ConnectionError:
                st.error(
                    f"❌ Impossible de joindre l'API sur {API_URL}. Vérifiez qu'Uvicorn est bien démarré."
                )
            except Exception as err:
                st.error(f"❌ Une erreur imprévue est survenue : {err}")