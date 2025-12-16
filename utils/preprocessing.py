import cv2
import numpy as np
import os
from tqdm import tqdm
from sklearn.utils import class_weight


class ImageProcessor:
    """Classe utilitaire pour le prétraitement des images"""

    @staticmethod
    def load_and_preprocess_images(folder_path, img_size=(224, 224), max_images_per_class=None):
        """
        Charger et prétraiter les images d'un dossier

        Args:
            folder_path: Chemin du dossier contenant les sous-dossiers de classes
            img_size: Taille de redimensionnement
            max_images_per_class: Nombre maximum d'images par classe

        Returns:
            images: Liste des images prétraitées
            labels: Liste des labels
            class_names: Noms des classes
        """
        images = []
        labels = []
        class_names = []

        if not os.path.exists(folder_path):
            print(f" Dossier non trouvé: {folder_path}")
            return None, None, None

        # Lister les classes (sous-dossiers)
        class_dirs = [d for d in os.listdir(folder_path)
                      if os.path.isdir(os.path.join(folder_path, d))]
        class_dirs.sort()

        print(f" Détection de {len(class_dirs)} classes dans {folder_path}")

        for class_idx, class_name in enumerate(class_dirs):
            class_path = os.path.join(folder_path, class_name)
            class_names.append(class_name)

            # Lister les images de la classe
            image_files = [f for f in os.listdir(class_path)
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.jfif'))]

            # Limiter si nécessaire
            if max_images_per_class:
                image_files = image_files[:max_images_per_class]

            print(f"   Classe '{class_name}': {len(image_files)} images")

            # Charger les images
            for img_file in tqdm(image_files, desc=f"Chargement {class_name}", leave=False):
                img_path = os.path.join(class_path, img_file)

                try:
                    # Lire l'image
                    img = cv2.imread(img_path)
                    if img is None:
                        continue

                    # Convertir BGR en RGB
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    # Redimensionner
                    img = cv2.resize(img, img_size)

                    # Normaliser [0, 1]
                    img = img.astype(np.float32) / 255.0

                    images.append(img)
                    labels.append(class_idx)

                except Exception as e:
                    print(f" Erreur avec {img_path}: {e}")
                    continue

        return np.array(images), np.array(labels), class_names

    @staticmethod
    def augment_image(image):
        """
        Appliquer des augmentations à une image

        Args:
            image: Image d'entrée

        Returns:
            augmented: Liste d'images augmentées
        """
        augmented = []

        # Image originale
        augmented.append(image)

        # Rotation à différents angles
        for angle in [90, 180, 270]:
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h))
            augmented.append(rotated)

        # Flip horizontal
        flipped_h = cv2.flip(image, 1)
        augmented.append(flipped_h)

        # Flip vertical
        flipped_v = cv2.flip(image, 0)
        augmented.append(flipped_v)

        # Ajustement de luminosité
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hsv[:, :, 2] = hsv[:, :, 2] * 0.7
        darker = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        augmented.append(darker)

        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hsv[:, :, 2] = hsv[:, :, 2] * 1.3
        brighter = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        augmented.append(brighter)

        return augmented

    @staticmethod
    def calculate_class_weights(y_train):
        """
        Calculer les poids des classes pour l'entraînement

        Args:
            y_train: Labels d'entraînement

        Returns:
            class_weights: Dictionnaire des poids
        """
        classes = np.unique(y_train)
        weights = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=y_train
        )

        class_weights = {cls: weight for cls, weight in zip(classes, weights)}

        print(" Poids des classes:")
        for cls, weight in class_weights.items():
            print(f"  Classe {cls}: {weight:.2f}")

        return class_weights

    @staticmethod
    def create_heatmap(image, model, last_conv_layer_name, pred_index=None):
        """
        Créer une heatmap Grad-CAM pour visualiser les zones importantes

        Args:
            image: Image d'entrée
            model: Modèle Keras
            last_conv_layer_name: Nom de la dernière couche convolutionnelle
            pred_index: Index de la classe prédite

        Returns:
            heatmap: Heatmap Grad-CAM
        """
        import tensorflow as tf
        from tensorflow import keras

        # Créer un modèle qui retourne les sorties de la dernière couche conv + prédictions
        grad_model = keras.models.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(last_conv_layer_name).output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(np.expand_dims(image, axis=0))
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]

        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # Normaliser la heatmap
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

        return heatmap.numpy()

    @staticmethod
    def overlay_heatmap(image, heatmap, alpha=0.4):
        """
        Superposer une heatmap sur l'image originale

        Args:
            image: Image originale
            heatmap: Heatmap à superposer
            alpha: Transparence de la heatmap

        Returns:
            superimposed: Image avec heatmap superposée
        """
        # Redimensionner la heatmap à la taille de l'image
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

        # Convertir en couleur
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # Superposer
        superimposed = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)

        return superimposed