import os
import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="BTP CreditScore — Recherche Entreprise",
    page_icon="🏢",
    layout="wide",
)
API_BASE_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# Ajoute https:// si l'URL est fournie sans protocole (cas de render.yaml)
if not API_BASE_URL.startswith("http://") and not API_BASE_URL.startswith("https://"):
    API_URL = f"https://{API_BASE_URL}"
else:
    API_URL = API_BASE_URL
    
# Exemple d'appel ensuite :
# response = requests.post(f"{API_URL}/predict", json=payload)


st.markdown(
    """
    <h2 style='text-align: center; margin-bottom: 25px;'>
        🏢 BTP CreditScore — Consultation des Entreprises
    </h2>
    """,
    unsafe_allow_html=True,
)

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
        "Rechercher l'entreprise", type="primary", use_container_width=True
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
                url = f"{API_BASE_URL}/entreprises/{siren_input.strip()}"
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
                                ca_format = f"{float(ca_val):,.0f} €".replace(
                                    ",", " "
                                )
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
                        # Carte visuelle du score et de la décision en haut à droite
                        couleurs_badge = {
                            "VERT": "#10b981",
                            "ORANGE": "#f59e0b",
                            "ROUGE": "#ef4444",
                            "NOIR": "#1e293b",
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
                                    <b>Montant Demandé :</b> <span style="color: #0f172a; font-weight: bold;">{montant_demande  :,.0f} €</span>
                                    <br>
                                    <b>Montant Accordé :</b> <span style="color: #0f172a; font-weight: bold;">{accord:,.0f} €</span>
                                    <span style="color: #64748b;">(Plafond : {plafond:,.0f} €)</span>
                                </p>
                            </div> 
                            """,
                            unsafe_allow_html=True,
                        )

                    st.info(f"💡 **Recommandation du moteur d'octroi :** {motif}")
                    st.divider()

                    # ---------------------------------------------------------
                    # AFFICHAGE 1 : Tableau vertical clé / valeur (Vue fiche)
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
                    f"❌ Impossible de joindre l'API sur {API_BASE_URL}. Vérifiez qu'Uvicorn est bien démarré."
                )
            except Exception as err:
                st.error(f"❌ Une erreur imprévue est survenue : {err}")