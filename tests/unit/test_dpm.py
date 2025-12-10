"""Unit tests for Data Preprocessing Module."""

import pytest
import pandas as pd
import numpy as np

from astridml.dpm import DataPreprocessor
from astridml.sdg import SyntheticDataGenerator


class TestDataPreprocessor:
    """Test suite for DataPreprocessor."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        sdg = SyntheticDataGenerator(seed=42)
        return sdg.generate_combined_data(n_days=60)

    @pytest.fixture
    def preprocessor(self):
        """Create a preprocessor instance."""
        return DataPreprocessor()

    def test_initialization(self, preprocessor):
        """Test preprocessor initialization."""
        assert preprocessor is not None
        assert preprocessor.scaler is not None
        assert not preprocessor.is_fitted

    def test_validate_data_valid(self, preprocessor, sample_data):
        """Test data validation with valid data."""
        is_valid, errors = preprocessor.validate_data(sample_data)
        assert is_valid
        assert len(errors) == 0

    def test_validate_data_missing_columns(self, preprocessor):
        """Test data validation with missing required columns."""
        df = pd.DataFrame({"foo": [1, 2, 3]})
        is_valid, errors = preprocessor.validate_data(df)

        assert not is_valid
        assert len(errors) > 0
        assert any("Missing required columns" in err for err in errors)

    def test_validate_data_invalid_date(self, preprocessor, sample_data):
        """Test data validation with invalid date format."""
        df = sample_data.copy()
        df["date"] = "invalid_date"

        is_valid, errors = preprocessor.validate_data(df)
        assert not is_valid

    def test_handle_missing_values(self, preprocessor, sample_data):
        """Test missing value handling."""
        df = sample_data.copy()

        # Introduce some missing values
        df.loc[5, "resting_heart_rate"] = np.nan
        df.loc[10, "sleep_hours"] = np.nan
        df.loc[15, "steps"] = np.nan

        df_cleaned = preprocessor.handle_missing_values(df)

        # Check no missing values remain
        assert not df_cleaned.isnull().any().any()

    def test_handle_missing_values_forward_fill(self, preprocessor, sample_data):
        """Test that time series columns use forward fill."""
        df = sample_data.copy()

        # Set a missing value in time series column
        original_value = df.loc[4, "resting_heart_rate"]
        df.loc[5, "resting_heart_rate"] = np.nan

        df_cleaned = preprocessor.handle_missing_values(df)

        # Should be filled with previous value
        assert df_cleaned.loc[5, "resting_heart_rate"] == original_value

    def test_engineer_features_temporal(self, preprocessor, sample_data):
        """Test temporal feature engineering."""
        df = preprocessor.engineer_features(sample_data)

        assert "day_of_week" in df.columns
        assert "is_weekend" in df.columns
        assert df["day_of_week"].between(0, 6).all()
        assert df["is_weekend"].isin([0, 1]).all()

    def test_engineer_features_rolling(self, preprocessor, sample_data):
        """Test rolling average features."""
        df = preprocessor.engineer_features(sample_data)

        # Check rolling features exist
        assert "resting_heart_rate_rolling_7d" in df.columns
        assert "energy_level_rolling_7d" in df.columns
        assert "mood_score_rolling_7d_std" in df.columns

        # Check no NaNs in rolling features
        assert not df["resting_heart_rate_rolling_7d"].isnull().any()

    def test_engineer_features_trends(self, preprocessor, sample_data):
        """Test trend feature engineering."""
        df = preprocessor.engineer_features(sample_data)

        # Check trend features exist
        assert "resting_heart_rate_trend" in df.columns
        assert "energy_level_trend" in df.columns

        # First value should be 0 (no previous data)
        assert df["resting_heart_rate_trend"].iloc[0] == 0

    def test_engineer_features_cycle_phase_encoding(self, preprocessor, sample_data):
        """Test cycle phase one-hot encoding."""
        df = preprocessor.engineer_features(sample_data)

        # Check one-hot encoded columns exist
        phase_cols = [col for col in df.columns if col.startswith("phase_")]
        assert len(phase_cols) > 0

        # Check that exactly one phase is active per row
        phase_sum = df[phase_cols].sum(axis=1)
        assert (phase_sum == 1).all()

    def test_engineer_features_derived_metrics(self, preprocessor, sample_data):
        """Test derived metric features."""
        df = preprocessor.engineer_features(sample_data)

        assert "recovery_ratio" in df.columns
        assert "hrv_rhr_ratio" in df.columns
        assert "energy_training_interaction" in df.columns

        # Check no NaNs
        assert not df["recovery_ratio"].isnull().any()
        assert not df["hrv_rhr_ratio"].isnull().any()

    def test_prepare_features_shape(self, preprocessor, sample_data):
        """Test feature preparation produces correct shape."""
        df = preprocessor.engineer_features(sample_data)
        X, y, feature_cols = preprocessor.prepare_features(df)

        assert X.shape[0] == len(df)
        assert X.shape[1] == len(feature_cols)
        assert y is None

    def test_prepare_features_with_targets(self, preprocessor, sample_data):
        """Test feature preparation with target columns."""
        df = preprocessor.engineer_features(sample_data)
        target_cols = ["energy_level", "mood_score", "pain_level"]

        X, y, feature_cols = preprocessor.prepare_features(df, target_cols)

        assert X.shape[0] == len(df)
        assert y.shape[0] == len(df)
        assert y.shape[1] == len(target_cols)

        # Target columns should not be in features
        for col in target_cols:
            assert col not in feature_cols

    def test_fit_transform(self, preprocessor, sample_data):
        """Test fit_transform pipeline."""
        target_cols = ["energy_level", "mood_score", "pain_level"]
        X, y, feature_cols = preprocessor.fit_transform(sample_data, target_cols)

        assert preprocessor.is_fitted
        assert X.shape[0] == len(sample_data)
        assert y.shape[0] == len(sample_data)
        assert len(feature_cols) > 0
        assert preprocessor.feature_columns == feature_cols

    def test_fit_transform_scaling(self, preprocessor, sample_data):
        """Test that features are properly scaled."""
        target_cols = ["energy_level"]
        X, _, _ = preprocessor.fit_transform(sample_data, target_cols)

        # After standardization, mean should be close to 0, std close to 1
        mean = np.abs(X.mean(axis=0))
        std = X.std(axis=0)

        assert np.allclose(mean, 0, atol=0.1)
        assert np.allclose(std, 1, atol=0.1)

    def test_transform_without_fit_raises(self, preprocessor, sample_data):
        """Test that transform without fit raises error."""
        with pytest.raises(ValueError, match="must be fitted"):
            preprocessor.transform(sample_data)

    def test_transform_after_fit(self, preprocessor, sample_data):
        """Test transform after fitting."""
        # Fit on data
        target_cols = ["energy_level"]
        preprocessor.fit_transform(sample_data, target_cols)

        # Generate new data
        sdg = SyntheticDataGenerator(seed=99)
        new_data = sdg.generate_combined_data(n_days=30)

        # Transform new data
        X_new, y_new = preprocessor.transform(new_data, target_cols)

        assert X_new.shape[0] == len(new_data)
        assert X_new.shape[1] == len(preprocessor.feature_columns)

    def test_get_feature_names(self, preprocessor, sample_data):
        """Test getting feature names."""
        preprocessor.fit_transform(sample_data)
        feature_names = preprocessor.get_feature_names()

        assert len(feature_names) > 0
        assert isinstance(feature_names, list)

    def test_get_feature_names_before_fit_raises(self, preprocessor):
        """Test that getting feature names before fit raises error."""
        with pytest.raises(ValueError, match="must be fitted"):
            preprocessor.get_feature_names()

    def test_fit_transform_invalid_data_raises(self, preprocessor):
        """Test that invalid data raises error."""
        df = pd.DataFrame({"foo": [1, 2, 3]})

        with pytest.raises(ValueError, match="validation failed"):
            preprocessor.fit_transform(df)

    def test_feature_count_consistency(self, preprocessor, sample_data):
        """Test that feature count is consistent across multiple transforms."""
        preprocessor.fit_transform(sample_data)
        n_features_1 = len(preprocessor.feature_columns)

        sdg = SyntheticDataGenerator(seed=99)
        new_data = sdg.generate_combined_data(n_days=30)

        X_new, _ = preprocessor.transform(new_data)
        n_features_2 = X_new.shape[1]

        assert n_features_1 == n_features_2

    def test_no_data_leakage(self, preprocessor):
        """Test that preprocessor doesn't leak information between datasets."""
        sdg = SyntheticDataGenerator(seed=42)
        data1 = sdg.generate_combined_data(n_days=60)

        preprocessor.fit_transform(data1, ["energy_level"])
        mean_1 = preprocessor.scaler.mean_

        # Fit on different data
        sdg2 = SyntheticDataGenerator(seed=99)
        data2 = sdg2.generate_combined_data(n_days=60)

        preprocessor2 = DataPreprocessor()
        preprocessor2.fit_transform(data2, ["energy_level"])
        mean_2 = preprocessor2.scaler.mean_

        # Means should be different (different data distributions)
        assert not np.allclose(mean_1, mean_2)

    def test_handles_small_dataset(self, preprocessor):
        """Test preprocessing works with small datasets."""
        sdg = SyntheticDataGenerator(seed=42)
        small_data = sdg.generate_combined_data(n_days=10)

        X, y, _ = preprocessor.fit_transform(small_data, ["energy_level"])

        assert X.shape[0] == 10
        assert y.shape[0] == 10

    def test_date_sorting(self, preprocessor, sample_data):
        """Test that data is sorted by date after engineering."""
        # Shuffle data
        df = sample_data.sample(frac=1, random_state=42)

        # Engineer features (should sort by date)
        df_processed = preprocessor.engineer_features(df)

        # Check dates are sorted
        dates = pd.to_datetime(df_processed["date"])
        assert dates.is_monotonic_increasing

    def test_rolling_features_respect_time_order(self, preprocessor, sample_data):
        """Test that rolling features respect temporal order."""
        df = preprocessor.engineer_features(sample_data)

        # Rolling average should be close to actual values (within reason)
        for i in range(7, len(df)):
            actual_mean = df.loc[i - 7 : i, "energy_level"].mean()
            rolling_mean = df.loc[i, "energy_level_rolling_7d"]
            assert abs(actual_mean - rolling_mean) < 0.1

    def test_multiple_target_columns(self, preprocessor, sample_data):
        """Test handling multiple target columns."""
        target_cols = ["energy_level", "mood_score", "pain_level", "training_load"]
        X, y, feature_cols = preprocessor.fit_transform(sample_data, target_cols)

        assert y.shape[1] == len(target_cols)

        # None of the target columns should be in features
        for col in target_cols:
            assert col not in feature_cols