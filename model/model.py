"""
Medical image classification model using EfficientNetB0 transfer learning.
Production-grade CNN classifier for chest X-ray triage.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers, callbacks
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MedicalImageClassifier:
    """Production-grade medical image classifier using EfficientNetB0."""

    def __init__(
        self,
        num_classes: int = 5,
        input_shape: Tuple[int, int, int] = (224, 224, 3),
        learning_rate: float = 1e-4,
        dropout_rate: float = 0.3
    ):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.model = None
        self.history = None
        self.class_names = ["Normal", "Pneumonia", "Pneumothorax", "Infiltration", "Mass"]

    def build_model(self) -> Model:
        """Build the transfer learning model with EfficientNetB0."""
        logger.info("Building EfficientNetB0 transfer learning model...")

        # Load pre-trained EfficientNetB0
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )

        # Freeze base model for transfer learning
        base_model.trainable = False

        # Build custom head
        inputs = tf.keras.Input(shape=self.input_shape)

        # Preprocessing for EfficientNet
        x = tf.keras.applications.efficientnet.preprocess_input(inputs)

        # Base model
        x = base_model(x, training=False)

        # Custom classification head
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.dropout_rate)(x)

        # Dense layers
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.dropout_rate)(x)

        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.dropout_rate / 2)(x)

        # Output layer
        outputs = layers.Dense(self.num_classes, activation='softmax', name='predictions')(x)

        self.model = Model(inputs, outputs)

        # Compile model
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )

        logger.info(f"Model built successfully with {self.model.count_params():,} parameters")
        return self.model

    def create_data_generators(
        self,
        data_dir: Path,
        batch_size: int = 32,
        validation_split: float = 0.2
    ) -> Tuple[ImageDataGenerator, ImageDataGenerator]:
        """Create data generators with augmentation."""
        logger.info("Creating data generators...")

        # Training data generator with augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=validation_split
        )

        # Validation data generator (no augmentation)
        val_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=validation_split
        )

        return train_datagen, val_datagen

    def prepare_data(
        self,
        data_dir: Path,
        batch_size: int = 32
    ) -> Tuple[tf.keras.utils.Sequence, tf.keras.utils.Sequence, tf.keras.utils.Sequence]:
        """Prepare training, validation, and test data."""
        train_datagen, val_datagen = self.create_data_generators(data_dir, batch_size)

        # Training generator
        train_generator = train_datagen.flow_from_directory(
            data_dir / "train",
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True,
            seed=42
        )

        # Validation generator
        val_generator = val_datagen.flow_from_directory(
            data_dir / "train",  # Use same directory with validation_split
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False,
            seed=42
        )

        # Test generator
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_directory(
            data_dir / "test",
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )

        # Store class indices for later use
        self.class_indices = train_generator.class_indices
        self.class_names = list(self.class_indices.keys())

        logger.info(f"Data prepared - Train: {train_generator.samples}, "
                   f"Val: {val_generator.samples}, Test: {test_generator.samples}")

        return train_generator, val_generator, test_generator

    def train(
        self,
        train_generator: tf.keras.utils.Sequence,
        val_generator: tf.keras.utils.Sequence,
        epochs: int = 50,
        patience: int = 10,
        save_path: Optional[Path] = None
    ) -> Dict:
        """Train the model with early stopping and learning rate reduction."""
        logger.info(f"Starting training for {epochs} epochs...")

        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        # Callbacks
        callback_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=patience//2,
                min_lr=1e-7,
                verbose=1
            )
        ]

        if save_path:
            callback_list.append(
                callbacks.ModelCheckpoint(
                    filepath=save_path / "best_model.h5",
                    monitor='val_accuracy',
                    save_best_only=True,
                    verbose=1
                )
            )

        # Train model
        self.history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=callback_list,
            verbose=1
        )

        logger.info("Training completed!")
        return self.history.history

    def fine_tune(
        self,
        train_generator: tf.keras.utils.Sequence,
        val_generator: tf.keras.utils.Sequence,
        epochs: int = 20,
        fine_tune_at: int = 100
    ) -> Dict:
        """Fine-tune the model by unfreezing top layers."""
        logger.info(f"Fine-tuning model from layer {fine_tune_at}...")

        if self.model is None:
            raise ValueError("Model not trained. Train the model first.")

        # Unfreeze the base model
        base_model = self.model.layers[2]  # EfficientNet base model
        base_model.trainable = True

        # Fine-tune from this layer onwards
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

        # Use a lower learning rate for fine-tuning
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate/10),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )

        # Fine-tune with callbacks
        fine_tune_history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=[
                callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                ),
                callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.2,
                    patience=3,
                    min_lr=1e-8
                )
            ],
            verbose=1
        )

        logger.info("Fine-tuning completed!")
        return fine_tune_history.history

    def predict_with_confidence(self, image: np.ndarray) -> Tuple[str, float, np.ndarray]:
        """Make prediction with confidence score."""
        if self.model is None:
            raise ValueError("Model not loaded. Load or train model first.")

        # Ensure image is in correct format
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)

        # Make prediction
        predictions = self.model.predict(image, verbose=0)
        confidence_scores = predictions[0]

        # Get predicted class
        predicted_class_idx = np.argmax(confidence_scores)
        predicted_class = self.class_names[predicted_class_idx]
        confidence = float(confidence_scores[predicted_class_idx])

        return predicted_class, confidence, confidence_scores

    def save_model(self, save_path: Path) -> None:
        """Save model in SavedModel format for TensorFlow Serving."""
        if self.model is None:
            raise ValueError("Model not built. Build and train model first.")

        save_path.mkdir(parents=True, exist_ok=True)

        # Save as SavedModel
        tf.saved_model.save(self.model, str(save_path))

        # Save class names and metadata
        metadata = {
            "class_names": self.class_names,
            "class_indices": self.class_indices,
            "input_shape": list(self.input_shape),
            "num_classes": self.num_classes
        }

        with open(save_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Model saved to {save_path}")

    def load_model(self, model_path: Path) -> None:
        """Load saved model."""
        self.model = tf.keras.models.load_model(str(model_path))

        # Load metadata if available
        metadata_path = model_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                self.class_names = metadata.get("class_names", self.class_names)
                self.class_indices = metadata.get("class_indices", {})

        logger.info(f"Model loaded from {model_path}")


def main():
    """Example usage of the medical image classifier."""
    # Initialize classifier
    classifier = MedicalImageClassifier()

    # Build model
    model = classifier.build_model()
    model.summary()


if __name__ == "__main__":
    main()