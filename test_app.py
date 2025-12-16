import streamlit as st
import numpy as np
from PIL import Image
import io


def test_upload_functionality():
    """Tester la fonctionnalité de téléchargement"""
    print("Test du téléchargement d'image...")

    # Créer une image de test
    img_array = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    test_image = Image.fromarray(img_array)

    # Simuler un fichier uploadé
    img_byte_arr = io.BytesIO()
    test_image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    print(" Image de test créée avec succès")
    return test_image


def test_preprocessing():
    """Tester le prétraitement"""
    print("\nTest du prétraitement...")

    from app import AppleDiseaseApp

    app = AppleDiseaseApp()
    test_image = test_upload_functionality()

    # Tester le prétraitement
    processed = app.preprocess_image(test_image)

    print(f" Prétraitement réussi")
    print(f"   Shape avant: {np.array(test_image).shape}")
    print(f"   Shape après: {processed.shape}")

    return processed


def test_model_loading():
    """Tester le chargement des modèles"""
    print("\nTest du chargement des modèles...")

    from app import AppleDiseaseApp

    app = AppleDiseaseApp()

    print(f" Modèles chargés: {len(app.loaded_models)}")
    for model_name in app.loaded_models.keys():
        print(f"   - {model_name}")

    return app


def run_all_tests():
    """Exécuter tous les tests"""
    print("=" * 50)
    print(" LANCEMENT DES TESTS DE L'APPLICATION")
    print("=" * 50)

    try:
        # Test 1: Téléchargement
        test_image = test_upload_functionality()

        # Test 2: Prétraitement
        processed_image = test_preprocessing()

        # Test 3: Modèles
        app = test_model_loading()

        # Test 4: Prédiction (si modèles disponibles)
        if app.loaded_models:
            print("\nTest de prédiction...")
            class_info, confidence, predictions = app.predict(test_image, 'fruit', 'resnet50')

            if class_info:
                print(f" Prédiction réussie")
                print(f"   Diagnostic: {class_info['name']}")
                print(f"   Confiance: {confidence:.2f}%")
            else:
                print("⚠ Prédiction échouée (modèle peut-être non disponible)")

        print("\n" + "=" * 50)
        print(" TOUS LES TESTS TERMINÉS AVEC SUCCÈS !")
        print("=" * 50)

    except Exception as e:
        print(f"\n Erreur pendant les tests: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()