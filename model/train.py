"""
Training script for the medical image classifier.
Handles data preparation, model training, evaluation, and export.
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import json
import logging
from typing import Dict, Tuple

from model import MedicalImageClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation and metrics calculation."""

    def __init__(self, classifier: MedicalImageClassifier):
        self.classifier = classifier
        self.class_names = classifier.class_names

    def evaluate_model(
        self,
        test_generator: tf.keras.utils.Sequence,
        save_path: Path
    ) -> Dict:
        """Comprehensive model evaluation with multiple metrics."""
        logger.info("Evaluating model performance...")

        # Get predictions
        test_generator.reset()
        predictions = self.classifier.model.predict(test_generator, verbose=1)
        predicted_classes = np.argmax(predictions, axis=1)

        # Get true labels
        true_classes = test_generator.classes

        # Calculate metrics
        metrics = {}

        # Basic metrics
        test_loss, test_accuracy, test_precision, test_recall = \
            self.classifier.model.evaluate(test_generator, verbose=0)

        metrics.update({
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'f1_score': 2 * (test_precision * test_recall) / (test_precision + test_recall)
        })

        # Classification report
        class_report = classification_report(
            true_classes,
            predicted_classes,
            target_names=self.class_names,
            output_dict=True
        )
        metrics['classification_report'] = class_report

        # Confusion matrix
        cm = confusion_matrix(true_classes, predicted_classes)
        metrics['confusion_matrix'] = cm.tolist()

        # ROC AUC (multi-class)
        try:
            # Binarize labels for multi-class ROC AUC
            y_true_binary = label_binarize(true_classes, classes=range(len(self.class_names)))
            if len(self.class_names) == 2:
                y_true_binary = np.hstack([1 - y_true_binary, y_true_binary])

            auc_scores = []
            for i in range(len(self.class_names)):
                auc = roc_auc_score(y_true_binary[:, i], predictions[:, i])
                auc_scores.append(auc)
                metrics[f'auc_{self.class_names[i]}'] = auc

            metrics['mean_auc'] = np.mean(auc_scores)

        except Exception as e:
            logger.warning(f"Could not calculate AUC scores: {e}")
            metrics['mean_auc'] = None

        # Save detailed metrics
        self._save_evaluation_plots(
            true_classes, predicted_classes, predictions, cm, save_path
        )

        # Save metrics to JSON
        metrics_path = save_path / "evaluation_metrics.json"
        with open(metrics_path, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj

            serializable_metrics = {k: convert_numpy(v) for k, v in metrics.items()}
            json.dump(serializable_metrics, f, indent=2)

        logger.info(f"Evaluation completed. Metrics saved to {metrics_path}")

        # Print summary
        print(f"\n{'='*50}")
        print("MODEL EVALUATION SUMMARY")
        print(f"{'='*50}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Precision: {test_precision:.4f}")
        print(f"Test Recall: {test_recall:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        if metrics['mean_auc']:
            print(f"Mean AUC: {metrics['mean_auc']:.4f}")
        print(f"{'='*50}\n")

        return metrics

    def _save_evaluation_plots(
        self,
        true_classes: np.ndarray,
        predicted_classes: np.ndarray,
        predictions: np.ndarray,
        confusion_matrix: np.ndarray,
        save_path: Path
    ) -> None:
        """Save evaluation plots."""
        plt.style.use('default')

        # Confusion Matrix Plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            confusion_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path / "confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Class-wise accuracy
        class_accuracy = []
        for i, class_name in enumerate(self.class_names):
            class_mask = true_classes == i
            if np.sum(class_mask) > 0:
                accuracy = np.mean(predicted_classes[class_mask] == i)
                class_accuracy.append(accuracy)
            else:
                class_accuracy.append(0)

        plt.figure(figsize=(12, 6))
        bars = plt.bar(self.class_names, class_accuracy)
        plt.title('Class-wise Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Class')
        plt.xticks(rotation=45)

        # Add value labels on bars
        for bar, acc in zip(bars, class_accuracy):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(save_path / "class_accuracy.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Confidence distribution
        confidence_scores = np.max(predictions, axis=1)
        correct_predictions = predicted_classes == true_classes

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.hist(confidence_scores[correct_predictions], bins=20, alpha=0.7,
                label='Correct', color='green')
        plt.hist(confidence_scores[~correct_predictions], bins=20, alpha=0.7,
                label='Incorrect', color='red')
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.title('Confidence Distribution')
        plt.legend()

        plt.subplot(1, 2, 2)
        for i, class_name in enumerate(self.class_names):
            class_mask = true_classes == i
            if np.sum(class_mask) > 0:
                plt.hist(confidence_scores[class_mask], bins=15, alpha=0.6,
                        label=class_name)
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.title('Confidence by True Class')
        plt.legend()

        plt.tight_layout()
        plt.savefig(save_path / "confidence_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()


def train_and_evaluate_model(data_dir: Path, model_dir: Path) -> None:
    """Complete training and evaluation pipeline."""
    logger.info("Starting medical image classifier training pipeline...")

    # Create output directories
    model_dir.mkdir(parents=True, exist_ok=True)
    evaluation_dir = model_dir / "evaluation"
    evaluation_dir.mkdir(exist_ok=True)

    # Initialize classifier
    classifier = MedicalImageClassifier(
        num_classes=5,
        learning_rate=1e-4,
        dropout_rate=0.3
    )

    # Build model
    model = classifier.build_model()
    logger.info(f"Model architecture:\n{model.summary()}")

    # Prepare data
    train_gen, val_gen, test_gen = classifier.prepare_data(
        data_dir, batch_size=32
    )

    # Training phase 1: Transfer learning
    logger.info("Phase 1: Transfer Learning")
    history = classifier.train(
        train_gen, val_gen,
        epochs=30,
        patience=8,
        save_path=model_dir
    )

    # Training phase 2: Fine-tuning (optional)
    logger.info("Phase 2: Fine-tuning")
    fine_tune_history = classifier.fine_tune(
        train_gen, val_gen,
        epochs=15,
        fine_tune_at=100
    )

    # Save training history
    training_history = {
        'transfer_learning': history,
        'fine_tuning': fine_tune_history
    }

    with open(model_dir / "training_history.json", 'w') as f:
        json.dump(training_history, f, indent=2)

    # Plot training history
    plot_training_history(training_history, evaluation_dir)

    # Evaluate model
    evaluator = ModelEvaluator(classifier)
    metrics = evaluator.evaluate_model(test_gen, evaluation_dir)

    # Export model for production
    export_path = model_dir / "saved_model"
    classifier.save_model(export_path)

    logger.info("Training and evaluation completed successfully!")
    logger.info(f"Model saved to: {export_path}")
    logger.info(f"Evaluation results: {evaluation_dir}")

    return classifier, metrics


def plot_training_history(history: Dict, save_path: Path) -> None:
    """Plot training history."""
    plt.figure(figsize=(15, 5))

    # Extract all history data
    all_history = {}
    for phase, phase_history in history.items():
        for metric, values in phase_history.items():
            if metric not in all_history:
                all_history[metric] = []
            all_history[metric].extend(values)

    # Plot accuracy
    plt.subplot(1, 3, 1)
    if 'accuracy' in all_history:
        plt.plot(all_history['accuracy'], label='Training Accuracy')
    if 'val_accuracy' in all_history:
        plt.plot(all_history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 3, 2)
    if 'loss' in all_history:
        plt.plot(all_history['loss'], label='Training Loss')
    if 'val_loss' in all_history:
        plt.plot(all_history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot learning rate
    plt.subplot(1, 3, 3)
    if 'lr' in all_history:
        plt.plot(all_history['lr'], label='Learning Rate')
        plt.title('Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.yscale('log')
        plt.legend()

    plt.tight_layout()
    plt.savefig(save_path / "training_history.png", dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main training script."""
    # Setup paths
    data_dir = Path("data/raw")
    model_dir = Path("models/medical_model")

    # Check if synthetic data exists, create if needed
    if not (data_dir / "train").exists():
        logger.info("Synthetic dataset not found. Creating...")
        from data.download_dataset import main as create_dataset
        from data.data_generator import main as generate_images

        create_dataset()
        generate_images()

    # Train and evaluate model
    classifier, metrics = train_and_evaluate_model(data_dir, model_dir)

    print("\nTraining pipeline completed successfully!")
    print(f"Final test accuracy: {metrics['test_accuracy']:.4f}")


if __name__ == "__main__":
    main()