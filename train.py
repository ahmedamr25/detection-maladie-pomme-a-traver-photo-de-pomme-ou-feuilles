import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

# TensorFlow imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
    TensorBoard
)
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.applications import ResNet50, EfficientNetB0, MobileNetV2
from tensorflow.keras.utils import to_categorical

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-4
NUM_CLASSES_FRUIT = 3  # Sain, Tavelure, Pourriture
NUM_CLASSES_LEAF = 4  # Sain, Tavelure, Pourriture noire, Rouille


class AppleDiseaseModelTrainer:
    def __init__(self):
        self.img_size = IMG_SIZE
        self.models_dir = 'models'
        self.results_dir = 'results'

        # Créer les répertoires nécessaires
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs('logs', exist_ok=True)

        # Labels pour fruits et feuilles
        self.fruit_labels = {
            0: 'Pomme Sain',
            1: 'Pomme Tavelure',
            2: 'Pomme Pourriture'
        }

        self.leaf_labels = {
            0: 'Feuille Sain',
            1: 'Tavelure Pommier',
            2: 'Pourriture Noire',
            3: 'Rouille Cèdre'
        }

    def create_synthetic_data(self, num_samples=300, data_type='fruit'):
        """
        Créer des données synthétiques pour le test
        Dans un cas réel, vous chargeriez les vrais datasets
        """
        print(f"Création de données synthétiques pour {data_type}...")

        if data_type == 'fruit':
            num_classes = NUM_CLASSES_FRUIT
        else:
            num_classes = NUM_CLASSES_LEAF

        images = []
        labels = []

        for i in range(num_samples):
            # Créer une image synthétique avec des patterns
            img = np.random.rand(self.img_size, self.img_size, 3) * 0.3

            # Ajouter des patterns selon la classe
            label = i % num_classes

            if label == 1:  # Maladie: ajouter des taches
                num_spots = np.random.randint(5, 15)
                for _ in range(num_spots):
                    x = np.random.randint(0, self.img_size)
                    y = np.random.randint(0, self.img_size)
                    radius = np.random.randint(5, 15)
                    cv2.circle(img, (x, y), radius, (0.8, 0.2, 0.2), -1)
            elif label == 2:  # Pourriture: zones sombres
                num_dark = np.random.randint(3, 8)
                for _ in range(num_dark):
                    x = np.random.randint(0, self.img_size)
                    y = np.random.randint(0, self.img_size)
                    w = np.random.randint(20, 40)
                    h = np.random.randint(20, 40)
                    img[y:y + h, x:x + w] *= 0.3

            images.append(img)
            labels.append(label)

        return np.array(images), np.array(labels)

    def create_data_generators(self, X_train, y_train, X_val, y_val, augmentation=True):
        """
        Créer des générateurs de données avec augmentation
        """
        if augmentation:
            train_datagen = ImageDataGenerator(
                rotation_range=40,
                width_shift_range=0.3,
                height_shift_range=0.3,
                shear_range=0.3,
                zoom_range=0.3,
                horizontal_flip=True,
                vertical_flip=True,
                brightness_range=[0.7, 1.3],
                fill_mode='nearest'
            )
        else:
            train_datagen = ImageDataGenerator()

        val_datagen = ImageDataGenerator()

        train_generator = train_datagen.flow(
            X_train, y_train,
            batch_size=BATCH_SIZE,
            shuffle=True
        )

        val_generator = val_datagen.flow(
            X_val, y_val,
            batch_size=BATCH_SIZE,
            shuffle=False
        )

        return train_generator, val_generator

    def build_resnet50(self, num_classes, fine_tune=False):
        """
        Construire le modèle ResNet50
        """
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(self.img_size, self.img_size, 3),
            pooling='avg'
        )

        if not fine_tune:
            # Geler les couches de base
            for layer in base_model.layers:
                layer.trainable = False

        model = models.Sequential([
            base_model,
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation='softmax')
        ])

        return model

    def build_efficientnet(self, num_classes):
        """
        Construire le modèle EfficientNet-B0
        """
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(self.img_size, self.img_size, 3),
            pooling='avg'
        )

        # Geler les premières couches
        for layer in base_model.layers[:-30]:
            layer.trainable = False

        model = models.Sequential([
            base_model,
            layers.Dropout(0.4),
            layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation='softmax')
        ])

        return model

    def build_custom_cnn(self, num_classes):
        """
        Construire un CNN personnalisé
        """
        model = models.Sequential([
            # Bloc 1
            layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                          input_shape=(self.img_size, self.img_size, 3)),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Bloc 2
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Bloc 3
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Bloc 4
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),

            # Couches denses
            layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])

        return model

    def compile_model(self, model, learning_rate=LEARNING_RATE):
        """
        Compiler le modèle avec les optimiseurs et métriques
        """
        optimizer = Adam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        )

        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc')
            ]
        )

        return model

    def train_model(self, model, train_gen, val_gen, model_name, data_type):
        """
        Entraîner le modèle avec callbacks
        """
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=os.path.join(self.models_dir, f'{model_name}_{data_type}.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            TensorBoard(
                log_dir=os.path.join('logs', f'{model_name}_{data_type}'),
                histogram_freq=1
            )
        ]

        print(f"\n{'=' * 60}")
        print(f"Entraînement du modèle {model_name} pour {data_type}")
        print(f"{'=' * 60}")

        history = model.fit(
            train_gen,
            epochs=EPOCHS,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )

        return history

    def evaluate_model(self, model, X_test, y_test, model_name, data_type):
        """
        Évaluer le modèle sur les données de test
        """
        print(f"\nÉvaluation du modèle {model_name}...")

        # Prédictions
        y_pred_proba = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)

        # Métriques
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")

        # Rapport de classification
        if data_type == 'fruit':
            target_names = list(self.fruit_labels.values())
        else:
            target_names = list(self.leaf_labels.values())

        report = classification_report(y_test, y_pred, target_names=target_names)
        print("\nRapport de classification:")
        print(report)

        # Matrice de confusion
        self.plot_confusion_matrix(y_test, y_pred, model_name, data_type, target_names)

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }

    def plot_confusion_matrix(self, y_true, y_pred, model_name, data_type, class_names):
        """
        Tracer et sauvegarder la matrice de confusion
        """
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )

        plt.title(f'Matrice de Confusion - {model_name} ({data_type})', fontsize=16)
        plt.ylabel('Vraies Étiquettes', fontsize=12)
        plt.xlabel('Étiquettes Prédites', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        # Sauvegarder
        save_path = os.path.join(self.results_dir, f'confusion_matrix_{model_name}_{data_type}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Matrice de confusion sauvegardée: {save_path}")

    def plot_training_history(self, history, model_name, data_type):
        """
        Tracer les courbes d'apprentissage
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        metrics = [
            ('accuracy', 'Accuracy'),
            ('loss', 'Loss'),
            ('precision', 'Precision'),
            ('recall', 'Recall')
        ]

        for idx, (metric, title) in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]

            if metric in history.history:
                ax.plot(history.history[metric], label=f'Train {title}')
                ax.plot(history.history[f'val_{metric}'], label=f'Val {title}')
                ax.set_title(f'{title} - {model_name}', fontsize=14)
                ax.set_xlabel('Epochs', fontsize=12)
                ax.set_ylabel(title, fontsize=12)
                ax.legend()
                ax.grid(True, alpha=0.3)

        plt.suptitle(f'Courbes d\'Apprentissage - {model_name} ({data_type})', fontsize=16)
        plt.tight_layout()

        # Sauvegarder
        save_path = os.path.join(self.results_dir, f'training_history_{model_name}_{data_type}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Courbes d'apprentissage sauvegardées: {save_path}")

    def compare_models(self, results_dict):
        """
        Comparer les performances des différents modèles
        """
        comparison_data = []

        for (model_name, data_type), metrics in results_dict.items():
            comparison_data.append({
                'Modèle': model_name,
                'Type': data_type,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1']
            })

        df_comparison = pd.DataFrame(comparison_data)

        # Sauvegarder la comparaison
        df_comparison.to_csv(os.path.join(self.results_dir, 'model_comparison.csv'), index=False)

        # Tracer la comparaison
        plt.figure(figsize=(12, 6))

        df_plot = df_comparison.melt(id_vars=['Modèle', 'Type'],
                                     value_vars=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                                     var_name='Metric', value_name='Score')

        sns.barplot(data=df_plot, x='Modèle', y='Score', hue='Metric')
        plt.title('Comparaison des Performances des Modèles', fontsize=16)
        plt.ylim(0, 1)
        plt.legend(title='Métrique', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        plt.tight_layout()

        save_path = os.path.join(self.results_dir, 'model_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Comparaison des modèles sauvegardée: {save_path}")

        return df_comparison

    def train_all_models(self):
        """
        Entraîner tous les modèles pour fruits et feuilles
        """
        all_results = {}

        # Pour chaque type de données (fruits et feuilles)
        for data_type in ['fruit', 'leaf']:
            print(f"\n{'#' * 60}")
            print(f"TRAITEMENT DES DONNÉES: {data_type.upper()}")
            print(f"{'#' * 60}")

            # Créer des données synthétiques (remplacer par vrai dataset)
            if data_type == 'fruit':
                num_classes = NUM_CLASSES_FRUIT
                num_samples = 500
            else:
                num_classes = NUM_CLASSES_LEAF
                num_samples = 600

            X, y = self.create_synthetic_data(num_samples, data_type)

            # Diviser les données
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
            )

            print(f"Données {data_type}:")
            print(f"  Train: {X_train.shape[0]} échantillons")
            print(f"  Validation: {X_val.shape[0]} échantillons")
            print(f"  Test: {X_test.shape[0]} échantillons")

            # Créer les générateurs
            train_gen, val_gen = self.create_data_generators(X_train, y_train, X_val, y_val)

            # Liste des modèles à entraîner
            models_to_train = [
                ('resnet50', self.build_resnet50),
                ('efficientnet', self.build_efficientnet),
                ('cnn_custom', self.build_custom_cnn)
            ]

            # Entraîner chaque modèle
            for model_name, model_builder in models_to_train:
                print(f"\n{'=' * 50}")
                print(f"Modèle: {model_name.upper()} - {data_type.upper()}")
                print(f"{'=' * 50}")

                # Construire le modèle
                model = model_builder(num_classes)

                # Compiler
                model = self.compile_model(model)

                # Afficher le résumé
                model.summary()

                # Entraîner
                history = self.train_model(model, train_gen, val_gen, model_name, data_type)

                # Évaluer
                results = self.evaluate_model(model, X_test, y_test, model_name, data_type)

                # Tracer l'historique
                self.plot_training_history(history, model_name, data_type)

                # Sauvegarder les résultats
                all_results[(model_name, data_type)] = results

        # Comparer tous les modèles
        print("\n" + "=" * 60)
        print("COMPARAISON DE TOUS LES MODÈLES")
        print("=" * 60)

        comparison_df = self.compare_models(all_results)
        print("\nComparaison des modèles:")
        print(comparison_df.to_string())

        # Déterminer le meilleur modèle pour chaque type
        best_fruit = comparison_df[comparison_df['Type'] == 'fruit'].sort_values('F1-Score', ascending=False).iloc[0]
        best_leaf = comparison_df[comparison_df['Type'] == 'leaf'].sort_values('F1-Score', ascending=False).iloc[0]

        print(f"\n MEILLEUR MODÈLE FRUITS: {best_fruit['Modèle']} (F1-Score: {best_fruit['F1-Score']:.4f})")
        print(f" MEILLEUR MODÈLE FEUILLES: {best_leaf['Modèle']} (F1-Score: {best_leaf['F1-Score']:.4f})")

        return all_results, comparison_df


def main():
    """
    Fonction principale pour l'entraînement
    """
    print(" Initialisation de l'entraînement des modèles...")
    print(f"TensorFlow version: {tf.__version__}")

    # Vérifier GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f" GPU détecté: {gpus}")
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    else:
        print(" Pas de GPU détecté, utilisation du CPU")

    # Initialiser le trainer
    trainer = AppleDiseaseModelTrainer()

    # Entraîner tous les modèles
    results, comparison_df = trainer.train_all_models()

    print("\n" + "=" * 60)
    print(" ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!")
    print("=" * 60)

    # Sauvegarder un rapport final
    report = f"""
     RAPPORT FINAL D'ENTRAÎNEMENT
    {'=' * 40}

     Date: {pd.Timestamp.now()}

    ️ Configuration:
    - Taille image: {IMG_SIZE}x{IMG_SIZE}
    - Batch size: {BATCH_SIZE}
    - Epochs max: {EPOCHS}
    - Learning rate: {LEARNING_RATE}

     Meilleurs modèles:
    - Fruits: {comparison_df[comparison_df['Type'] == 'fruit'].sort_values('F1-Score', ascending=False).iloc[0]['Modèle']}
    - Feuilles: {comparison_df[comparison_df['Type'] == 'leaf'].sort_values('F1-Score', ascending=False).iloc[0]['Modèle']}

     Modèles sauvegardés dans: {trainer.models_dir}
     Résultats sauvegardés dans: {trainer.results_dir}

     Pour utiliser l'application:
    streamlit run app.py
    """

    with open(os.path.join(trainer.results_dir, 'training_report.txt'), 'w') as f:
        f.write(report)

    print(report)


if __name__ == "__main__":
    main()