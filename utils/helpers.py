import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import plotly.graph_objects as go
import plotly.express as px


class ModelHelper:
    """Classe helper pour les opérations sur les modèles"""

    @staticmethod
    def save_model_info(model, model_name, data_type, history=None, metrics=None):
        """
        Sauvegarder les informations du modèle

        Args:
            model: Modèle Keras
            model_name: Nom du modèle
            data_type: Type de données (fruit/leaf)
            history: Historique d'entraînement
            metrics: Métriques d'évaluation
        """
        info = {
            'model_name': model_name,
            'data_type': data_type,
            'date_saved': datetime.now().isoformat(),
            'model_summary': [],
            'input_shape': model.input_shape,
            'output_shape': model.output_shape
        }

        # Sauvegarder le résumé du modèle
        stringlist = []
        model.summary(print_fn=lambda x: stringlist.append(x))
        info['model_summary'] = stringlist

        # Sauvegarder l'historique si disponible
        if history:
            info['training_history'] = {
                'epochs': len(history.history.get('loss', [])),
                'final_train_loss': history.history.get('loss', [])[-1] if history.history.get('loss') else None,
                'final_val_loss': history.history.get('val_loss', [])[-1] if history.history.get('val_loss') else None,
                'final_train_accuracy': history.history.get('accuracy', [])[-1] if history.history.get(
                    'accuracy') else None,
                'final_val_accuracy': history.history.get('val_accuracy', [])[-1] if history.history.get(
                    'val_accuracy') else None
            }

        # Sauvegarder les métriques
        if metrics:
            info['evaluation_metrics'] = metrics

        # Sauvegarder dans un fichier JSON
        filename = f'models/{model_name}_{data_type}_info.json'
        with open(filename, 'w') as f:
            json.dump(info, f, indent=4)

        print(f" Informations du modèle sauvegardées: {filename}")

    @staticmethod
    def load_model_info(model_name, data_type):
        """
        Charger les informations du modèle

        Args:
            model_name: Nom du modèle
            data_type: Type de données

        Returns:
            info: Informations du modèle
        """
        filename = f'models/{model_name}_{data_type}_info.json'
        try:
            with open(filename, 'r') as f:
                info = json.load(f)
            return info
        except FileNotFoundError:
            print(f" Fichier d'information non trouvé: {filename}")
            return None

    @staticmethod
    def create_performance_report(results_dict, save_path='results/performance_report.md'):
        """
        Créer un rapport de performance au format Markdown

        Args:
            results_dict: Dictionnaire des résultats
            save_path: Chemin de sauvegarde
        """
        report = "#  Rapport de Performance des Modèles\n\n"
        report += f"Date de génération: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        # Tableau des performances
        report += "## Comparaison des Modèles\n\n"
        report += "| Modèle | Type | Accuracy | Precision | Recall | F1-Score |\n"
        report += "|--------|------|----------|-----------|--------|----------|\n"

        for (model_name, data_type), metrics in results_dict.items():
            report += f"| {model_name} | {data_type} | "
            report += f"{metrics.get('accuracy', 0):.4f} | "
            report += f"{metrics.get('precision', 0):.4f} | "
            report += f"{metrics.get('recall', 0):.4f} | "
            report += f"{metrics.get('f1', 0):.4f} |\n"

        report += "\n## Recommandations\n\n"

        # Trouver les meilleurs modèles
        fruit_results = {k: v for k, v in results_dict.items() if k[1] == 'fruit'}
        leaf_results = {k: v for k, v in results_dict.items() if k[1] == 'leaf'}

        if fruit_results:
            best_fruit = max(fruit_results.items(), key=lambda x: x[1].get('f1', 0))
            report += f"###  Meilleur modèle pour les fruits:\n"
            report += f"- **{best_fruit[0][0]}** avec F1-Score: {best_fruit[1].get('f1', 0):.4f}\n\n"

        if leaf_results:
            best_leaf = max(leaf_results.items(), key=lambda x: x[1].get('f1', 0))
            report += f"###  Meilleur modèle pour les feuilles:\n"
            report += f"- **{best_leaf[0][0]}** avec F1-Score: {best_leaf[1].get('f1', 0):.4f}\n\n"

        report += "## Notes d'utilisation\n\n"
        report += "1. Utiliser le modèle avec le meilleur F1-Score pour chaque type\n"
        report += "2. L'ensemble (voting) peut améliorer la robustesse\n"
        report += "3. Considérer le temps d'inférence pour les applications temps réel\n"

        # Sauvegarder le rapport
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f" Rapport de performance sauvegardé: {save_path}")

        return report

    @staticmethod
    def visualize_predictions(images, true_labels, pred_labels, class_names, n_samples=10):
        """
        Visualiser les prédictions

        Args:
            images: Images
            true_labels: Vrais labels
            pred_labels: Labels prédits
            class_names: Noms des classes
            n_samples: Nombre d'échantillons à visualiser
        """
        n_samples = min(n_samples, len(images))

        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()

        for i in range(n_samples):
            ax = axes[i]

            # Afficher l'image
            ax.imshow(images[i])

            # Couleur selon la prédiction (vert=correct, rouge=incorrect)
            color = 'green' if true_labels[i] == pred_labels[i] else 'red'

            # Titre
            true_name = class_names[true_labels[i]]
            pred_name = class_names[pred_labels[i]]
            ax.set_title(f"Vrai: {true_name}\nPrédit: {pred_name}",
                         color=color, fontsize=10)

            ax.axis('off')

        # Cacher les axes vides
        for i in range(n_samples, len(axes)):
            axes[i].axis('off')

        plt.suptitle("Visualisation des prédictions", fontsize=16)
        plt.tight_layout()
        plt.savefig('results/predictions_visualization.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(" Visualisation des prédictions sauvegardée")

    @staticmethod
    def create_interactive_confusion_matrix(y_true, y_pred, class_names, title="Matrice de Confusion"):
        """
        Créer une matrice de confusion interactive

        Args:
            y_true: Vrais labels
            y_pred: Labels prédits
            class_names: Noms des classes
            title: Titre du graphique
        """
        cm = confusion_matrix(y_true, y_pred)

        # Créer un heatmap interactif
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=class_names,
            y=class_names,
            colorscale='Blues',
            text=cm,
            texttemplate='%{text}',
            textfont={"size": 12},
            hoverongaps=False,
            hovertemplate='Vrai: %{y}<br>Prédit: %{x}<br>Nombre: %{z}<extra></extra>'
        ))

        fig.update_layout(
            title=title,
            xaxis_title='Prédit',
            yaxis_title='Vrai',
            width=700,
            height=600,
            yaxis=dict(autorange='reversed'),
            margin=dict(l=50, r=50, t=80, b=50)
        )

        # Sauvegarder
        fig.write_html('results/interactive_confusion_matrix.html')

        print(" Matrice de confusion interactive sauvegardée")

        return fig


class DataAnalyzer:
    """Classe pour analyser les données"""

    @staticmethod
    def analyze_dataset(images, labels, class_names):
        """
        Analyser les caractéristiques du dataset

        Args:
            images: Images
            labels: Labels
            class_names: Noms des classes
        """
        print("\n Analyse du Dataset:")
        print(f"   Nombre total d'images: {len(images)}")
        print(f"   Taille des images: {images[0].shape}")
        print(f"   Nombre de classes: {len(class_names)}")

        # Distribution des classes
        unique, counts = np.unique(labels, return_counts=True)
        distribution = dict(zip(unique, counts))

        print("\n   Distribution des classes:")
        for class_idx, count in distribution.items():
            class_name = class_names[class_idx] if class_idx < len(class_names) else f"Classe {class_idx}"
            percentage = (count / len(labels)) * 100
            print(f"     {class_name}: {count} images ({percentage:.1f}%)")

        # Statistiques des pixels
        pixel_stats = {
            'mean': np.mean(images),
            'std': np.std(images),
            'min': np.min(images),
            'max': np.max(images)
        }

        print(f"\n   Statistiques des pixels:")
        print(f"     Moyenne: {pixel_stats['mean']:.3f}")
        print(f"     Écart-type: {pixel_stats['std']:.3f}")
        print(f"     Min: {pixel_stats['min']:.3f}")
        print(f"     Max: {pixel_stats['max']:.3f}")

        return {
            'total_images': len(images),
            'class_distribution': distribution,
            'pixel_stats': pixel_stats
        }