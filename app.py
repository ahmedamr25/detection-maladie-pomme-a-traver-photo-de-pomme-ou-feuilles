import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import tensorflow as tf
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
import base64

# Configuration de la page
st.set_page_config(
    page_title="🍎 Détection de Maladies des Pommes",
    page_icon="🍏",
    layout="wide",
    initial_sidebar_state="expanded"
)


# CSS personnalisé
def load_css():
    st.markdown("""
    <style>
    /* Styles généraux */
    .main {
        padding: 0rem 1rem;
    }

    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        font-size: 1.1em;
        font-weight: bold;
    }

    /* Header */
    .header-container {
        background: linear-gradient(135deg, #2E8B57 0%, #228B22 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        color: white;
        text-align: center;
    }

    .header-title {
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        color: white;
    }

    .header-subtitle {
        font-size: 1.2rem;
        opacity: 0.9;
        color: white;
    }

    /* Cards */
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.08);
        border-left: 5px solid #4CAF50;
        margin-bottom: 1.5rem;
    }

    .disease-card {
        border-left: 5px solid #f44336;
        background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
    }

    .healthy-card {
        border-left: 5px solid #4CAF50;
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
    }

    /* Metrics */
    .metric-card {
        text-align: center;
        padding: 1rem;
        border-radius: 10px;
        background: #f8f9fa;
        border: 2px solid #e9ecef;
    }

    .metric-value {
        font-size: 2.2rem;
        font-weight: bold;
        color: #2E8B57;
    }

    .metric-label {
        font-size: 1rem;
        color: #6c757d;
    }

    /* Progress bars */
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 5px 5px 0px 0px;
        padding: 10px 20px;
    }

    .stTabs [aria-selected="true"] {
        background-color: #4CAF50 !important;
        color: white !important;
    }

    /* File uploader */
    .uploadedFile {
        border: 2px dashed #4CAF50;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        background: #f8fff8;
    }

    /* Alert boxes */
    .stAlert {
        border-radius: 10px;
    }

    /* Sidebar */
    .css-1d391kg {
        padding-top: 2rem;
    }

    /* Image containers */
    .image-container {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }

    /* Badges */
    .badge {
        display: inline-block;
        padding: 0.25em 0.6em;
        font-size: 0.75em;
        font-weight: 700;
        line-height: 1;
        text-align: center;
        white-space: nowrap;
        vertical-align: baseline;
        border-radius: 0.375rem;
        color: #fff;
    }

    .badge-healthy {
        background-color: #28a745;
    }

    .badge-disease {
        background-color: #dc3545;
    }

    /* Tooltips */
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: pointer;
    }

    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #555;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }

    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }

    </style>
    """, unsafe_allow_html=True)


# Charger le CSS
load_css()


class AppleDiseaseApp:
    def __init__(self):
        # Dictionnaire des classes
        self.fruit_classes = {
            0: {"name": " Pomme Sain", "description": "Pomme en bonne santé", "color": "#4CAF50"},
            1: {"name": "️ Pomme Tavelure", "description": "Tavelure du pommier détectée", "color": "#FF9800"},
            2: {"name": " Pomme Pourrie", "description": "Pourriture détectée", "color": "#F44336"}
        }

        self.leaf_classes = {
            0: {"name": " Feuille Sain", "description": "Feuille en bonne santé", "color": "#4CAF50"},
            1: {"name": "️ Tavelure Pommier", "description": "Tavelure du pommier sur feuille", "color": "#FF9800"},
            2: {"name": " Pourriture Noire", "description": "Pourriture noire détectée", "color": "#9C27B0"},
            3: {"name": " Rouille Cèdre", "description": "Rouille du cèdre détectée", "color": "#795548"}
        }

        # Informations sur les maladies
        self.disease_info = {
            "Tavelure Pommier": {
                "symptoms": "Taches olive-brun sur feuilles et fruits, déformations",
                "treatment": "Fongicides (soufre, cuivre), élimination des feuilles infectées",
                "prevention": "Bon drainage, taille d'aération, variétés résistantes"
            },
            "Pourriture Noire": {
                "symptoms": "Taches violettes sur fruits, pourriture marron",
                "treatment": "Retrait des fruits infectés, fongicides (captane)",
                "prevention": "Élagage, récolte au bon moment"
            },
            "Rouille Cèdre": {
                "symptoms": "Taches orange sur feuilles, défoliation",
                "treatment": "Fongicides (myclobutanil), élimination des genévriers",
                "prevention": "Distance avec genévriers, variétés résistantes"
            }
        }

        # Chemins des modèles
        self.model_dir = "models"
        self.loaded_models = {}

        # Initialiser l'historique
        if 'history' not in st.session_state:
            st.session_state.history = []

        # Charger les modèles
        self.load_models()

    def load_models(self):
        """Charger tous les modèles disponibles"""
        try:
            model_files = {
                'fruit': {
                    'resnet50': 'fruit_resnet50.h5',
                    'efficientnet': 'fruit_efficientnet.h5',
                    'cnn_custom': 'fruit_cnn_custom.h5'
                },
                'leaf': {
                    'resnet50': 'leaf_resnet50.h5',
                    'efficientnet': 'leaf_efficientnet.h5',
                    'cnn_custom': 'leaf_cnn_custom.h5'
                }
            }

            for data_type, models in model_files.items():
                for model_name, filename in models.items():
                    model_path = os.path.join(self.model_dir, filename)
                    if os.path.exists(model_path):
                        self.loaded_models[f"{data_type}_{model_name}"] = tf.keras.models.load_model(model_path)

            st.sidebar.success(f" {len(self.loaded_models)} modèles chargés")

        except Exception as e:
            st.sidebar.error(f"Erreur chargement modèles: {str(e)}")

    def preprocess_image(self, image, img_size=224):
        """Prétraiter l'image pour la prédiction"""
        # Convertir en array numpy
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image

        # Assurer 3 canaux
        if len(img_array.shape) == 2:  # Niveaux de gris
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif img_array.shape[2] == 4:  # RGBA
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)

        # Redimensionner
        img_array = cv2.resize(img_array, (img_size, img_size))

        # Normaliser
        img_array = img_array.astype(np.float32) / 255.0

        # Ajouter dimension batch
        img_array = np.expand_dims(img_array, axis=0)

        return img_array

    def predict(self, image, data_type='fruit', model_type='resnet50'):
        """Faire une prédiction avec un modèle spécifique"""
        model_key = f"{data_type}_{model_type}"

        if model_key not in self.loaded_models:
            return None, None, None

        # Prétraiter l'image
        processed_img = self.preprocess_image(image)

        # Faire la prédiction
        model = self.loaded_models[model_key]
        predictions = model.predict(processed_img, verbose=0)

        # Obtenir la classe prédite
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0]) * 100

        # Obtenir les informations de la classe
        if data_type == 'fruit':
            class_info = self.fruit_classes.get(predicted_class)
        else:
            class_info = self.leaf_classes.get(predicted_class)

        return class_info, confidence, predictions[0]

    def ensemble_predict(self, image, data_type='fruit'):
        """Prédiction par ensemble de modèles"""
        predictions_list = []
        confidences = []
        model_names = []

        # Déterminer quels modèles utiliser
        if data_type == 'fruit':
            available_models = ['resnet50', 'efficientnet', 'cnn_custom']
        else:
            available_models = ['resnet50', 'efficientnet', 'cnn_custom']

        for model_name in available_models:
            model_key = f"{data_type}_{model_name}"
            if model_key in self.loaded_models:
                class_info, confidence, pred_probs = self.predict(image, data_type, model_name)
                if class_info:
                    predictions_list.append(pred_probs)
                    confidences.append(confidence)
                    model_names.append(model_name)

        if predictions_list:
            # Moyenne des prédictions
            avg_predictions = np.mean(predictions_list, axis=0)
            predicted_class = np.argmax(avg_predictions)
            avg_confidence = np.mean(confidences)

            # Obtenir les informations de la classe
            if data_type == 'fruit':
                class_info = self.fruit_classes.get(predicted_class)
            else:
                class_info = self.leaf_classes.get(predicted_class)

            return class_info, avg_confidence, avg_predictions, model_names

        return None, None, None, []

    def display_header(self):
        """Afficher l'en-tête de l'application"""
        st.markdown("""
        <div class="header-container">
            <h1 class="header-title"> Détection Intelligente de Maladies des Pommes</h1>
            <p class="header-subtitle">
                Utilisez l'IA pour détecter les maladies des pommes à partir de photos de fruits ou de feuilles
            </p>
        </div>
        """, unsafe_allow_html=True)

    def display_sidebar(self):
        """Afficher la barre latérale"""
        with st.sidebar:
            st.markdown("## ⚙ Configuration")

            # Type d'analyse
            st.markdown("###  Type d'analyse")
            analysis_type = st.radio(
                "Que souhaitez-vous analyser ?",
                ["Pomme (Fruit)", " Feuille"],
                key="analysis_type"
            )

            # Modèle à utiliser
            st.markdown("###  Modèle IA")
            model_options = {
                " Pomme (Fruit)": ["ResNet50", "EfficientNet-B0", "CNN Personnalisé", "Ensemble (3 modèles)"],
                " Feuille": ["ResNet50", "EfficientNet-B0", "CNN Personnalisé", "Ensemble (3 modèles)"]
            }

            selected_model = st.selectbox(
                "Choisissez le modèle:",
                model_options[analysis_type],
                key="model_selection"
            )

            # Seuil de confiance
            st.markdown("###  Seuil de confiance")
            confidence_threshold = st.slider(
                "Seuil minimum de confiance (%)",
                50, 100, 85,
                key="confidence_threshold"
            )

            # Informations
            st.markdown("---")
            st.markdown("##  Statistiques")
            st.metric("Modèles chargés", len(self.loaded_models))
            st.metric("Analyses effectuées", len(st.session_state.history))

            # Reset button
            if st.button(" Réinitialiser", type="secondary"):
                st.session_state.history = []
                st.rerun()

            st.markdown("---")
            st.markdown("##  À propos")
            st.info("""
            Cette application utilise le deep learning pour détecter:
            - **Tavelure du pommier**
            - **Pourriture noire**
            - **Rouille du cèdre**

            Basé sur les datasets Fruits-360 et PlantVillage.
            """)

    def display_upload_section(self):
        """Afficher la section de téléchargement"""
        st.markdown("##  Télécharger une image")

        col1, col2 = st.columns([2, 1])

        with col1:
            uploaded_file = st.file_uploader(
                "Glissez-déposez votre image ici",
                type=['jpg', 'jpeg', 'png', 'bmp'],
                help="Format recommandé: JPG, PNG. Taille max: 10MB"
            )

        with col2:
            st.markdown("###  Ou utilisez un exemple:")
            example_option = st.selectbox(
                "Choisir un exemple:",
                ["-- Sélectionner --", "Pomme saine", "Pomme malade", "Feuille saine", "Feuille malade"]
            )

        # Afficher l'image sélectionnée
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            self.display_image_preview(image, "Image téléchargée")
            return image, "uploaded"
        elif example_option != "-- Sélectionner --":
            # Charger une image exemple (dans un cas réel, vous auriez des images prédéfinies)
            image = self.load_example_image(example_option)
            if image:
                self.display_image_preview(image, f"Exemple: {example_option}")
                return image, "example"

        return None, None

    def load_example_image(self, example_type):
        """Charger une image exemple"""
        # Créer une image synthétique pour la démo
        img_size = 400
        img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 200

        if "Pomme saine" in example_type:
            # Créer une pomme verte
            center = (img_size // 2, img_size // 2)
            radius = 150
            cv2.circle(img, center, radius, (100, 200, 100), -1)
        elif "Pomme malade" in example_type:
            # Créer une pomme avec taches
            center = (img_size // 2, img_size // 2)
            radius = 150
            cv2.circle(img, center, radius, (150, 100, 100), -1)
            # Ajouter des taches
            for _ in range(10):
                x = np.random.randint(100, 300)
                y = np.random.randint(100, 300)
                cv2.circle(img, (x, y), 15, (50, 50, 50), -1)

        return Image.fromarray(img)

    def display_image_preview(self, image, title):
        """Afficher un aperçu de l'image"""
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(f"### {title}")
            st.image(image, use_column_width=True)

            # Informations sur l'image
            img_array = np.array(image)
            st.caption(f"Dimensions: {img_array.shape[1]}x{img_array.shape[0]} pixels")

    def display_results(self, class_info, confidence, predictions, data_type, model_names):
        """Afficher les résultats de la prédiction"""
        st.markdown("---")
        st.markdown("##  Résultats de l'analyse")

        # Déterminer si c'est une maladie
        is_disease = "Sain" not in class_info["name"] if class_info else False

        # Afficher la carte de résultat
        if is_disease:
            st.markdown(f'<div class="card disease-card">', unsafe_allow_html=True)
            st.error(f"##  MALADIE DÉTECTÉE")
        else:
            st.markdown(f'<div class="card healthy-card">', unsafe_allow_html=True)
            st.success(f"##  SAIN")

        # Informations principales
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Diagnostic", class_info["name"])

        with col2:
            st.metric("Confiance", f"{confidence:.1f}%")

        with col3:
            if len(model_names) > 1:
                st.metric("Modèles utilisés", f"{len(model_names)} modèles")
            else:
                st.metric("Modèle utilisé", model_names[0] if model_names else "N/A")

        st.markdown('</div>', unsafe_allow_html=True)

        # Graphique des probabilités
        self.display_probability_chart(predictions, data_type, class_info["name"])

        # Informations sur la maladie si détectée
        if is_disease:
            self.display_disease_info(class_info["name"])

        # Recommandations
        self.display_recommendations(is_disease, class_info["name"])

        # Ajouter à l'historique
        self.add_to_history(class_info["name"], confidence, is_disease)

    def display_probability_chart(self, predictions, data_type, predicted_class):
        """Afficher le graphique des probabilités"""
        st.markdown("###  Probabilités par classe")

        if data_type == 'fruit':
            classes_dict = self.fruit_classes
        else:
            classes_dict = self.leaf_classes

        # Créer un DataFrame pour les probabilités
        prob_data = []
        for class_idx, class_info in classes_dict.items():
            prob_data.append({
                "Classe": class_info["name"],
                "Probabilité": predictions[class_idx] * 100,
                "Couleur": class_info["color"]
            })

        df_probs = pd.DataFrame(prob_data)

        # Trier par probabilité
        df_probs = df_probs.sort_values("Probabilité", ascending=False)

        # Créer le graphique avec Plotly
        fig = go.Figure(data=[
            go.Bar(
                x=df_probs["Probabilité"],
                y=df_probs["Classe"],
                orientation='h',
                marker_color=df_probs["Couleur"],
                text=df_probs["Probabilité"].round(1).astype(str) + '%',
                textposition='outside',
                hovertemplate='<b>%{y}</b><br>Probabilité: %{x:.1f}%<extra></extra>'
            )
        ])

        fig.update_layout(
            title="Distribution des probabilités",
            xaxis_title="Probabilité (%)",
            yaxis_title="Classe",
            height=400,
            showlegend=False,
            xaxis=dict(range=[0, 100]),
            plot_bgcolor='white'
        )

        st.plotly_chart(fig, use_container_width=True)

        # Afficher aussi sous forme de tableau
        with st.expander(" Voir les détails numériques"):
            df_display = df_probs.copy()
            df_display["Probabilité"] = df_display["Probabilité"].round(2)
            st.dataframe(df_display, use_container_width=True)

    def display_disease_info(self, disease_name):
        """Afficher les informations sur la maladie"""
        st.markdown("### 🩺 Informations sur la maladie")

        # Extraire le nom de la maladie
        disease_key = None
        for key in self.disease_info.keys():
            if key in disease_name:
                disease_key = key
                break

        if disease_key:
            info = self.disease_info[disease_key]

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("####  Symptômes")
                st.info(info["symptoms"])

            with col2:
                st.markdown("####  Traitement")
                st.warning(info["treatment"])

            with col3:
                st.markdown("####  Prévention")
                st.success(info["prevention"])
        else:
            st.warning("Informations détaillées non disponibles pour cette maladie.")

    def display_recommendations(self, is_disease, disease_name):
        """Afficher les recommandations"""
        st.markdown("###  Recommandations")

        if is_disease:
            if "Tavelure" in disease_name:
                recommendations = [
                    "🔸 Appliquer un fongicide à base de soufre ou de cuivre",
                    "🔸 Éliminer et détruire les feuilles et fruits infectés",
                    "🔸 Tailler pour améliorer la circulation d'air",
                    "🔸 Éviter l'irrigation par aspersion",
                    "🔸 Choisir des variétés résistantes pour les nouvelles plantations"
                ]
            elif "Pourriture" in disease_name:
                recommendations = [
                    "🔸 Retirer immédiatement les fruits infectés",
                    "🔸 Appliquer un fongicide captane avant la récolte",
                    "🔸 Assurer une bonne circulation d'air dans le verger",
                    "🔸 Récolter au bon moment pour éviter les blessures",
                    "🔸 Stocker les fruits dans des conditions optimales"
                ]
            elif "Rouille" in disease_name:
                recommendations = [
                    "🔸 Éliminer les genévriers à proximité (hôte alternatif)",
                    "🔸 Appliquer un fongicide myclobutanil au printemps",
                    "🔸 Planter des variétés de pommiers résistants",
                    "🔸 Surveiller régulièrement l'apparition des symptômes",
                    "🔸 Tailler les branches infectées"
                ]
            else:
                recommendations = [
                    "🔸 Consulter un agronome pour un diagnostic précis",
                    "🔸 Isoler la plante infectée si possible",
                    "🔸 Éviter la propagation à d'autres plantes",
                    "🔸 Maintenir de bonnes pratiques culturales"
                ]
        else:
            recommendations = [
                " Votre plante semble en bonne santé !",
                "🔸 Continuer les bonnes pratiques culturales",
                "🔸 Surveiller régulièrement l'apparition de symptômes",
                "🔸 Maintenir une fertilisation équilibrée",
                "🔸 Assurer une irrigation adaptée"
            ]

        for rec in recommendations:
            st.markdown(f"- {rec}")

    def add_to_history(self, diagnosis, confidence, is_disease):
        """Ajouter une analyse à l'historique"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.history.append({
            "timestamp": timestamp,
            "diagnosis": diagnosis,
            "confidence": confidence,
            "is_disease": is_disease
        })

    def display_history(self):
        """Afficher l'historique des analyses"""
        if st.session_state.history:
            st.markdown("---")
            st.markdown("##  Historique des analyses")

            # Convertir en DataFrame
            df_history = pd.DataFrame(st.session_state.history)

            # Formater les colonnes
            df_history["Confiance"] = df_history["confidence"].apply(lambda x: f"{x:.1f}%")
            df_history["Statut"] = df_history["is_disease"].apply(lambda x: " Maladie" if x else " Sain")

            # Afficher le tableau
            st.dataframe(
                df_history[["timestamp", "diagnosis", "Confiance", "Statut"]],
                column_config={
                    "timestamp": "Date/Heure",
                    "diagnosis": "Diagnostic",
                    "Confiance": "Confiance",
                    "Statut": "Statut"
                },
                use_container_width=True,
                hide_index=True
            )

            # Télécharger l'historique
            csv = df_history.to_csv(index=False)
            st.download_button(
                label=" Télécharger l'historique (CSV)",
                data=csv,
                file_name="historique_analyses.csv",
                mime="text/csv"
            )

    def display_footer(self):
        """Afficher le pied de page"""
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #666; padding: 2rem;'>
            <p> <strong>Système de Détection de Maladies des Pommes</strong> </p>
            <p>Utilise les datasets Fruits-360 et PlantVillage | Développé avec TensorFlow & Streamlit</p>
            <p style='font-size: 0.9rem; margin-top: 1rem;'>
                 <em>Cet outil fournit des recommandations préliminaires. 
                Consultez toujours un professionnel pour un diagnostic définitif.</em>
            </p>
        </div>
        """, unsafe_allow_html=True)

    def run(self):
        """Exécuter l'application principale"""
        # Afficher l'en-tête
        self.display_header()

        # Barre latérale
        self.display_sidebar()

        # Section principale
        tab1, tab2, tab3 = st.tabs([" Analyse", " Dashboard", " Aide"])

        with tab1:
            # Section de téléchargement
            image, source = self.display_upload_section()

            # Bouton d'analyse
            if image is not None:
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button(" Analyser l'image", type="primary", use_container_width=True):
                        with st.spinner(" Analyse en cours..."):
                            # Déterminer le type d'analyse
                            analysis_type = st.session_state.analysis_type
                            data_type = "fruit" if "Pomme" in analysis_type else "leaf"

                            # Déterminer le modèle
                            model_selection = st.session_state.model_selection
                            if "Ensemble" in model_selection:
                                # Prédiction par ensemble
                                class_info, confidence, predictions, model_names = self.ensemble_predict(
                                    image, data_type
                                )
                                model_used = "ensemble"
                            else:
                                # Prédiction avec un modèle spécifique
                                model_mapping = {
                                    "ResNet50": "resnet50",
                                    "EfficientNet-B0": "efficientnet",
                                    "CNN Personnalisé": "cnn_custom"
                                }
                                model_type = model_mapping[model_selection]
                                class_info, confidence, predictions = self.predict(
                                    image, data_type, model_type
                                )
                                model_names = [model_type]

                            # Vérifier le seuil de confiance
                            threshold = st.session_state.confidence_threshold

                            if class_info and confidence >= threshold:
                                # Afficher les résultats
                                self.display_results(
                                    class_info, confidence, predictions,
                                    data_type, model_names
                                )
                            elif class_info:
                                st.warning(f"""
                                 **Confiance trop faible ({confidence:.1f}%)**

                                Le modèle n'est pas suffisamment confiant dans sa prédiction 
                                (seuil: {threshold}%).

                                **Conseils:**
                                - Essayez avec une image plus claire
                                - Vérifiez que l'image montre bien une pomme/feuille
                                - Essayez un autre modèle
                                """)
                            else:
                                st.error(" Erreur lors de l'analyse. Veuillez réessayer.")

            # Historique
            self.display_history()

        with tab2:
            self.display_dashboard()

        with tab3:
            self.display_help()

        # Pied de page
        self.display_footer()

    def display_dashboard(self):
        """Afficher le tableau de bord"""
        st.markdown("##  Tableau de bord")

        # Statistiques globales
        if st.session_state.history:
            df_history = pd.DataFrame(st.session_state.history)

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                total = len(df_history)
                st.metric("Total analyses", total)

            with col2:
                diseases = df_history["is_disease"].sum()
                st.metric("Maladies détectées", diseases)

            with col3:
                healthy = total - diseases
                st.metric("Plantes saines", healthy)

            with col4:
                avg_confidence = df_history["confidence"].mean()
                st.metric("Confiance moyenne", f"{avg_confidence:.1f}%")

            # Graphiques
            col1, col2 = st.columns(2)

            with col1:
                # Répartition santé/maladie
                fig1 = go.Figure(data=[go.Pie(
                    labels=["Saines", "Malades"],
                    values=[healthy, diseases],
                    hole=.3,
                    marker_colors=['#4CAF50', '#F44336']
                )])
                fig1.update_layout(title="Répartition santé/maladie")
                st.plotly_chart(fig1, use_container_width=True)

            with col2:
                # Évolution des analyses
                if len(df_history) > 1:
                    df_history["timestamp_dt"] = pd.to_datetime(df_history["timestamp"])
                    df_history = df_history.sort_values("timestamp_dt")

                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(
                        x=df_history["timestamp_dt"],
                        y=df_history["confidence"],
                        mode='lines+markers',
                        name='Confiance',
                        line=dict(color='#4CAF50')
                    ))
                    fig2.update_layout(
                        title="Évolution de la confiance",
                        xaxis_title="Date",
                        yaxis_title="Confiance (%)"
                    )
                    st.plotly_chart(fig2, use_container_width=True)

            # Top diagnostics
            st.markdown("###  Top diagnostics")
            diagnosis_counts = df_history["diagnosis"].value_counts().reset_index()
            diagnosis_counts.columns = ["Diagnostic", "Nombre"]

            fig3 = px.bar(
                diagnosis_counts,
                x="Nombre",
                y="Diagnostic",
                orientation='h',
                color="Nombre",
                color_continuous_scale='Greens'
            )
            fig3.update_layout(title="Fréquence des diagnostics")
            st.plotly_chart(fig3, use_container_width=True)

        else:
            st.info(" Aucune analyse effectuée. Commencez par analyser une image !")

    def display_help(self):
        """Afficher la section d'aide"""
        st.markdown("## ℹ Guide d'utilisation")

        with st.expander(" Comment prendre une bonne photo ?", expanded=True):
            st.markdown("""
            ### Pour des résultats optimaux :

            ** Pour les pommes :**
            - Prenez la photo en lumière naturelle
            - Placez la pomme sur un fond neutre
            - Capturez les détails de la surface
            - Évitez les reflets et ombres fortes

            ** Pour les feuilles :**
            - Photographiez la feuille à plat
            - Incluez toute la feuille dans le cadre
            - Montrez les deux côtés si possible
            - Évitez les feuilles mouillées

            ** À éviter :**
            - Photos floues ou sombres
            - Mains ou objets dans le cadre
            - Reflets de flash
            - Feuilles/fruits partiellement cachés
            """)

        with st.expander(" À propos des modèles IA"):
            st.markdown("""
            ### Modèles disponibles :

            ** ResNet50**
            - Architecture profonde (50 couches)
            - Très précise pour la classification
            - Utilise le transfer learning

            **⚡ EfficientNet-B0**
            - Optimisé pour l'efficacité
            - Bon équilibre précision/performance
            - Nécessite moins de ressources

            **️ CNN Personnalisé**
            - Architecture conçue spécifiquement
            - Plus léger que les autres
            - Bon pour l'apprentissage

            ** Ensemble**
            - Combine les 3 modèles
            - Meilleure précision globale
            - Plus robuste aux variations
            """)

        with st.expander(" Maladies détectées"):
            st.markdown("""
            ### Maladies prises en charge :

            ** Tavelure du pommier**
            - Symptômes : Taches olive-brun sur feuilles et fruits
            - Traitement : Fongicides, élimination des parties infectées
            - Prévention : Variétés résistantes, bonne circulation d'air

            ** Pourriture noire**
            - Symptômes : Taches violettes, pourriture marron
            - Traitement : Retrait des fruits, fongicides
            - Prévention : Récolte au bon moment, stockage adapté

            ** Rouille du cèdre**
            - Symptômes : Taches orange sur feuilles
            - Traitement : Élimination des genévriers, fongicides
            - Prévention : Distance avec hôtes alternatifs
            """)

        with st.expander(" Questions fréquentes"):
            st.markdown("""
            **Q: Quelle est la précision du système ?**
            R: La précision varie selon le modèle et l'image. En moyenne : 85-95%.

            **Q: Puis-je utiliser l'application sur mobile ?**
            R: Oui ! L'application est responsive et fonctionne sur tous les appareils.

            **Q: Les données sont-elles sauvegardées ?**
            R: L'historique est stocké localement dans votre session. Aucune donnée n'est envoyée à nos serveurs.

            **Q: Que faire si le diagnostic est incertain ?**
            R: Consultez toujours un agronome ou un professionnel pour confirmation.

            **Q: Comment améliorer les résultats ?**
            R: Utilisez des images de bonne qualité, essayez différents modèles, utilisez le mode "Ensemble".
            """)


def main():
    """Fonction principale"""
    # Initialiser l'application
    app = AppleDiseaseApp()

    # Exécuter l'application
    app.run()


if __name__ == "__main__":
    main()