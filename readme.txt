1. Installation des dépendances :
bash
# Créer un environnement virtuel
python -m venv venv

# Activer (Windows)
venv\Scripts\activate

# Activer (Mac/Linux)
source venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt
2. Préparer les données :
bash
# Créer la structure de dossiers
mkdir -p models data results logs static/images static/css
3. Entraîner les modèles :
bash
# Pour un entraînement complet
python train.py

# Pour un entraînement rapide (test)
python train.py --quick
4. Lancer l'application :
bash
# Mode développement
streamlit run app.py

# Avec options spécifiques
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
5. Tester l'application :
bash
# Exécuter les tests
python test_app.py

# Lancer avec des données réelles
# 1. Téléchargez Fruits-360: https://www.kaggle.com/datasets/moltean/fruits
# 2. Téléchargez PlantVillage: https://www.kaggle.com/datasets/mohitsingh1804/plantvillage?resource=download
# 3. Placez-les dans le dossier data/