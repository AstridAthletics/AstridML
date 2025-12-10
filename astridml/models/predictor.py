"""Machine learning model for symptom prediction and performance forecasting."""

import numpy as np
from tensorflow import keras
from typing import Optional, Dict, List
import json


class SymptomPredictor:
    """
    Neural network model for predicting menstrual symptoms and athletic performance.

    Uses a feedforward neural network to predict future symptoms based on
    historical wearable and cycle data.
    """

    def __init__(
        self,
        input_dim: Optional[int] = None,
        hidden_layers: List[int] = [128, 64, 32],
        dropout_rate: float = 0.3,
    ):
        """
        Initialize the symptom predictor.

        Args:
            input_dim: Dimension of input features
            hidden_layers: List of hidden layer sizes
            dropout_rate: Dropout rate for regularization
        """
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.model: Optional[keras.Model] = None
        self.history = None
        self.is_fitted = False

    def build_model(self, output_dim: int = 3) -> keras.Model:
        """
        Build the neural network architecture.

        Args:
            output_dim: Number of output predictions (e.g., energy, mood, pain)

        Returns:
            Compiled Keras model
        """
        if self.input_dim is None:
            raise ValueError("input_dim must be specified before building model")

        model = keras.Sequential(
            [
                keras.layers.Input(shape=(self.input_dim,)),
                keras.layers.BatchNormalization(),
            ]
        )

        # Add hidden layers with dropout
        for units in self.hidden_layers:
            model.add(
                keras.layers.Dense(
                    units,
                    activation="relu",
                    kernel_regularizer=keras.regularizers.l2(0.01),
                )
            )
            model.add(keras.layers.Dropout(self.dropout_rate))

        # Output layer
        model.add(keras.layers.Dense(output_dim, activation="linear"))

        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="mse",
            metrics=["mae", "mse"],
        )

        self.model = model
        return model

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
        verbose: int = 1,
    ) -> Dict:
        """
        Train the model on data.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            epochs: Number of training epochs
            batch_size: Batch size for training
            verbose: Verbosity level

        Returns:
            Training history dictionary
        """
        if self.model is None:
            if self.input_dim is None:
                self.input_dim = X_train.shape[1]

            output_dim = y_train.shape[1] if len(y_train.shape) > 1 else 1
            self.build_model(output_dim)

        # Early stopping and learning rate reduction
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss" if X_val is not None else "loss",
                patience=15,
                restore_best_weights=True,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss" if X_val is not None else "loss",
                factor=0.5,
                patience=5,
                min_lr=1e-6,
            ),
        ]

        validation_data = (
            (X_val, y_val) if X_val is not None and y_val is not None else None
        )

        history = self.model.fit(
            X_train,
            y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose,
        )

        self.history = history.history
        self.is_fitted = True
        return self.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.

        Args:
            X: Input features

        Returns:
            Predicted values
        """
        if self.model is None:
            raise ValueError("Model must be trained before prediction")

        return self.model.predict(X, verbose=0)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.

        Args:
            X_test: Test features
            y_test: Test targets

        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")

        results = self.model.evaluate(X_test, y_test, verbose=0, return_dict=True)

        # Convert to float and ensure proper metric names
        metrics = {}
        for name, value in results.items():
            metrics[name] = float(value)

        return metrics

    def save(self, filepath: str):
        """
        Save the model to disk.

        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save")

        self.model.save(filepath)

        # Save configuration
        config = {
            "input_dim": self.input_dim,
            "hidden_layers": self.hidden_layers,
            "dropout_rate": self.dropout_rate,
        }
        with open(f"{filepath}_config.json", "w") as f:
            json.dump(config, f)

    def load(self, filepath: str):
        """
        Load a saved model from disk.

        Args:
            filepath: Path to the saved model
        """
        self.model = keras.models.load_model(filepath)
        self.is_fitted = True

        # Load configuration
        try:
            with open(f"{filepath}_config.json", "r") as f:
                config = json.load(f)
                self.input_dim = config["input_dim"]
                self.hidden_layers = config["hidden_layers"]
                self.dropout_rate = config["dropout_rate"]
        except FileNotFoundError:
            pass
