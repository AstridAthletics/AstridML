"""Unit tests for ML models."""

import pytest
import numpy as np
import tempfile
import os

from astridml.models import SymptomPredictor, RecommendationEngine
from astridml.sdg import SyntheticDataGenerator
from astridml.dpm import DataPreprocessor


class TestSymptomPredictor:
    """Test suite for SymptomPredictor."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample training data."""
        sdg = SyntheticDataGenerator(seed=42)
        df = sdg.generate_combined_data(n_days=90)

        preprocessor = DataPreprocessor()
        target_cols = ["energy_level", "mood_score", "pain_level"]
        X, y, _ = preprocessor.fit_transform(df, target_cols)

        return X, y

    def test_initialization(self):
        """Test predictor initialization."""
        predictor = SymptomPredictor(input_dim=50)
        assert predictor.input_dim == 50
        assert predictor.model is None
        assert not predictor.is_fitted

    def test_initialization_with_custom_params(self):
        """Test predictor initialization with custom parameters."""
        predictor = SymptomPredictor(input_dim=50, hidden_layers=[64, 32], dropout_rate=0.2)
        assert predictor.hidden_layers == [64, 32]
        assert predictor.dropout_rate == 0.2

    def test_build_model(self):
        """Test model building."""
        predictor = SymptomPredictor(input_dim=50)
        model = predictor.build_model(output_dim=3)

        assert model is not None
        assert predictor.model is not None
        assert len(model.layers) > 0

    def test_build_model_without_input_dim_raises(self):
        """Test that building model without input_dim raises error."""
        predictor = SymptomPredictor()

        with pytest.raises(ValueError, match="input_dim must be specified"):
            predictor.build_model()

    def test_build_model_output_shape(self):
        """Test model has correct output shape."""
        predictor = SymptomPredictor(input_dim=50)
        model = predictor.build_model(output_dim=3)

        # Test with dummy input
        dummy_input = np.random.randn(1, 50)
        output = model.predict(dummy_input, verbose=0)

        assert output.shape == (1, 3)

    def test_train(self, sample_data):
        """Test model training."""
        X, y = sample_data

        predictor = SymptomPredictor(input_dim=X.shape[1])
        history = predictor.train(X, y, epochs=5, verbose=0)

        assert "loss" in history
        assert len(history["loss"]) > 0
        assert predictor.model is not None

    def test_train_with_validation(self, sample_data):
        """Test training with validation data."""
        X, y = sample_data

        # Split data
        split = int(0.8 * len(X))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        predictor = SymptomPredictor(input_dim=X.shape[1])
        history = predictor.train(X_train, y_train, X_val, y_val, epochs=5, verbose=0)

        assert "val_loss" in history
        assert len(history["val_loss"]) > 0

    def test_train_reduces_loss(self, sample_data):
        """Test that training reduces loss."""
        X, y = sample_data

        predictor = SymptomPredictor(input_dim=X.shape[1])
        history = predictor.train(X, y, epochs=20, verbose=0)

        # Loss should generally decrease (allowing some fluctuation)
        initial_loss = history["loss"][0]
        final_loss = history["loss"][-1]

        assert final_loss < initial_loss

    def test_predict(self, sample_data):
        """Test making predictions."""
        X, y = sample_data

        predictor = SymptomPredictor(input_dim=X.shape[1])
        predictor.train(X, y, epochs=5, verbose=0)

        predictions = predictor.predict(X[:10])

        assert predictions.shape == (10, 3)
        assert not np.isnan(predictions).any()

    def test_predict_without_training_raises(self):
        """Test that predicting without training raises error."""
        predictor = SymptomPredictor(input_dim=50)

        with pytest.raises(ValueError, match="must be trained"):
            predictor.predict(np.random.randn(5, 50))

    def test_evaluate(self, sample_data):
        """Test model evaluation."""
        X, y = sample_data

        predictor = SymptomPredictor(input_dim=X.shape[1])
        predictor.train(X, y, epochs=10, verbose=0)

        metrics = predictor.evaluate(X, y)

        assert "loss" in metrics
        assert "mae" in metrics
        assert "mse" in metrics
        assert all(isinstance(v, float) for v in metrics.values())

    def test_evaluate_without_training_raises(self):
        """Test that evaluating without training raises error."""
        predictor = SymptomPredictor(input_dim=50)

        with pytest.raises(ValueError, match="must be trained"):
            predictor.evaluate(np.random.randn(5, 50), np.random.randn(5, 3))

    def test_save_and_load(self, sample_data):
        """Test saving and loading model."""
        X, y = sample_data

        predictor = SymptomPredictor(input_dim=X.shape[1])
        predictor.train(X, y, epochs=5, verbose=0)

        # Save model
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "model.keras")
            predictor.save(filepath)

            # Load model
            new_predictor = SymptomPredictor()
            new_predictor.load(filepath)

            # Test predictions match
            pred1 = predictor.predict(X[:5])
            pred2 = new_predictor.predict(X[:5])

            np.testing.assert_allclose(pred1, pred2, rtol=1e-5)

    def test_save_without_model_raises(self):
        """Test that saving without model raises error."""
        predictor = SymptomPredictor(input_dim=50)

        with pytest.raises(ValueError, match="No model to save"):
            predictor.save("/tmp/model.keras")

    def test_early_stopping(self, sample_data):
        """Test that early stopping works."""
        X, y = sample_data

        # Add noise to make training harder
        y_noisy = y + np.random.randn(*y.shape) * 2

        predictor = SymptomPredictor(input_dim=X.shape[1])
        history = predictor.train(
            X, y_noisy, epochs=100, verbose=0  # Set high, but should stop early
        )

        # Should stop before 100 epochs
        assert len(history["loss"]) < 100


class TestRecommendationEngine:
    """Test suite for RecommendationEngine."""

    @pytest.fixture
    def recommender(self):
        """Create a recommendation engine instance."""
        return RecommendationEngine()

    def test_initialization(self, recommender):
        """Test recommendation engine initialization."""
        assert recommender is not None

    def test_assess_energy_level(self, recommender):
        """Test energy level assessment."""
        assert recommender._assess_energy_level(2) == "low_energy"
        assert recommender._assess_energy_level(5) == "normal"
        assert recommender._assess_energy_level(8) == "high_energy"

    def test_assess_pain_level(self, recommender):
        """Test pain level assessment."""
        assert recommender._assess_pain_level(2) == "normal"
        assert recommender._assess_pain_level(7) == "high_pain"

    def test_assess_sleep_quality(self, recommender):
        """Test sleep quality assessment."""
        assert recommender._assess_sleep_quality(50) == "poor_sleep"
        assert recommender._assess_sleep_quality(80) == "normal"

    def test_assess_hrv(self, recommender):
        """Test HRV assessment."""
        avg_hrv = 60

        assert recommender._assess_hrv(70, avg_hrv) == "high_hrv"
        assert recommender._assess_hrv(50, avg_hrv) == "low_hrv"
        assert recommender._assess_hrv(60, avg_hrv) == "normal"

    def test_generate_recommendations_menstrual_phase(self, recommender):
        """Test recommendations for menstrual phase."""
        current_data = {
            "cycle_phase": "menstrual",
            "energy_level": 3,
            "pain_level": 7,
            "mood_score": 5,
            "sleep_quality_score": 70,
            "heart_rate_variability": 60,
            "heart_rate_variability_rolling_7d": 65,
        }

        recommendations = recommender.generate_recommendations(current_data)

        assert "nutrition" in recommendations
        assert "recovery" in recommendations
        assert "performance" in recommendations
        assert len(recommendations["nutrition"]) > 0
        assert len(recommendations["recovery"]) > 0
        assert len(recommendations["performance"]) > 0

    def test_generate_recommendations_follicular_phase(self, recommender):
        """Test recommendations for follicular phase."""
        current_data = {
            "cycle_phase": "follicular",
            "energy_level": 8,
            "pain_level": 1,
            "mood_score": 8,
            "sleep_quality_score": 85,
            "heart_rate_variability": 75,
            "heart_rate_variability_rolling_7d": 65,
        }

        recommendations = recommender.generate_recommendations(current_data)

        assert len(recommendations["nutrition"]) > 0
        assert len(recommendations["performance"]) > 0

        # Should recommend high-intensity training
        performance_text = " ".join(recommendations["performance"])
        assert "high-intensity" in performance_text.lower() or "intense" in performance_text.lower()

    def test_generate_recommendations_with_predictions(self, recommender):
        """Test recommendations with prediction data."""
        current_data = {
            "cycle_phase": "luteal",
            "energy_level": 6,
            "pain_level": 2,
            "mood_score": 6,
            "sleep_quality_score": 75,
            "heart_rate_variability": 65,
            "heart_rate_variability_rolling_7d": 65,
        }

        predictions = {"energy_level": 3, "mood_score": 5, "pain_level": 4}  # Predicted low energy

        recommendations = recommender.generate_recommendations(current_data, predictions)

        # Should mention predicted low energy
        recovery_text = " ".join(recommendations["recovery"])
        assert "predicted" in recovery_text.lower() or "tomorrow" in recovery_text.lower()

    def test_generate_recommendations_all_phases(self, recommender):
        """Test that recommendations work for all cycle phases."""
        phases = ["menstrual", "follicular", "ovulatory", "luteal"]

        for phase in phases:
            current_data = {
                "cycle_phase": phase,
                "energy_level": 5,
                "pain_level": 2,
                "mood_score": 6,
                "sleep_quality_score": 75,
                "heart_rate_variability": 65,
                "heart_rate_variability_rolling_7d": 65,
            }

            recommendations = recommender.generate_recommendations(current_data)

            assert "nutrition" in recommendations
            assert "recovery" in recommendations
            assert "performance" in recommendations

    def test_generate_recommendations_high_pain(self, recommender):
        """Test recommendations with high pain levels."""
        current_data = {
            "cycle_phase": "menstrual",
            "energy_level": 4,
            "pain_level": 8,
            "mood_score": 5,
            "sleep_quality_score": 70,
            "heart_rate_variability": 60,
            "heart_rate_variability_rolling_7d": 65,
        }

        recommendations = recommender.generate_recommendations(current_data)

        # Should recommend pain management strategies
        recovery_text = " ".join(recommendations["recovery"])
        assert "rest" in recovery_text.lower() or "gentle" in recovery_text.lower()

    def test_generate_recommendations_poor_sleep(self, recommender):
        """Test recommendations with poor sleep quality."""
        current_data = {
            "cycle_phase": "luteal",
            "energy_level": 4,
            "pain_level": 2,
            "mood_score": 5,
            "sleep_quality_score": 45,
            "heart_rate_variability": 65,
            "heart_rate_variability_rolling_7d": 65,
        }

        recommendations = recommender.generate_recommendations(current_data)

        # Should recommend sleep improvements
        recovery_text = " ".join(recommendations["recovery"])
        assert "sleep" in recovery_text.lower()

    def test_format_recommendations(self, recommender):
        """Test recommendation formatting."""
        recommendations = {
            "nutrition": ["Eat more protein", "Stay hydrated"],
            "recovery": ["Get 8 hours of sleep"],
            "performance": ["Focus on technique"],
        }

        formatted = recommender.format_recommendations(recommendations)

        assert "NUTRITION" in formatted
        assert "RECOVERY" in formatted
        assert "PERFORMANCE" in formatted
        assert "Eat more protein" in formatted
        assert "Stay hydrated" in formatted

    def test_format_recommendations_empty_categories(self, recommender):
        """Test formatting with empty categories."""
        recommendations = {"nutrition": ["Eat well"], "recovery": [], "performance": []}

        formatted = recommender.format_recommendations(recommendations)

        assert "NUTRITION" in formatted
        assert "Eat well" in formatted
        # Empty categories should not appear
        assert formatted.count("RECOVERY") == 0

    def test_recommendations_contain_actionable_advice(self, recommender):
        """Test that recommendations contain actionable advice."""
        current_data = {
            "cycle_phase": "ovulatory",
            "energy_level": 8,
            "pain_level": 1,
            "mood_score": 8,
            "sleep_quality_score": 85,
            "heart_rate_variability": 75,
            "heart_rate_variability_rolling_7d": 65,
        }

        recommendations = recommender.generate_recommendations(current_data)

        # Check that recommendations are non-empty strings
        for category, items in recommendations.items():
            for item in items:
                assert isinstance(item, str)
                assert len(item) > 10  # Should be meaningful advice
