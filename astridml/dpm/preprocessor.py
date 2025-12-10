"""Data Preprocessing Module for cleaning and feature engineering."""

import numpy as np
import pandas as pd
from typing import Optional, List, Tuple
from sklearn.preprocessing import StandardScaler


class DataPreprocessor:
    """
    Preprocesses wearable and symptom data for machine learning models.

    Handles data cleaning, missing value imputation, feature engineering,
    and normalization.
    """

    def __init__(self):
        """Initialize the data preprocessor."""
        self.scaler = StandardScaler()
        self.feature_columns: Optional[List[str]] = None
        self.is_fitted = False

    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate input data format and required columns.

        Args:
            df: Input DataFrame

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []

        required_cols = ["date", "cycle_day", "cycle_phase"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")

        if "date" in df.columns:
            try:
                pd.to_datetime(df["date"])
            except Exception as e:
                errors.append(f"Invalid date format: {e}")

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            errors.append("No numeric columns found")

        return len(errors) == 0, errors

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values using appropriate strategies.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with missing values handled
        """
        df = df.copy()

        # Forward fill for time series data (suitable for vital signs)
        time_series_cols = [
            "resting_heart_rate",
            "heart_rate_variability",
            "sleep_hours",
            "sleep_quality_score",
        ]
        for col in time_series_cols:
            if col in df.columns:
                df[col] = df[col].ffill().bfill()

        # Fill zero for count-based metrics
        count_cols = ["steps", "active_minutes", "calories_burned"]
        for col in count_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        # Fill with mean for other numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].mean())

        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived features for improved model performance.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with engineered features
        """
        df = df.copy()

        # Ensure date is datetime
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        # Temporal features
        df["day_of_week"] = df["date"].dt.dayofweek
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

        # Rolling averages (7-day windows)
        rolling_cols = [
            "resting_heart_rate",
            "heart_rate_variability",
            "sleep_hours",
            "sleep_quality_score",
            "training_load",
            "energy_level",
            "mood_score",
            "pain_level",
        ]

        for col in rolling_cols:
            if col in df.columns:
                df[f"{col}_rolling_7d"] = (
                    df[col].rolling(window=8, min_periods=1).mean()
                )
                df[f"{col}_rolling_7d_std"] = (
                    df[col].rolling(window=8, min_periods=1).std().fillna(0)
                )

        # Trend features (difference from previous day)
        trend_cols = [
            "resting_heart_rate",
            "heart_rate_variability",
            "sleep_quality_score",
            "energy_level",
            "mood_score",
        ]

        for col in trend_cols:
            if col in df.columns:
                df[f"{col}_trend"] = df[col].diff().fillna(0)

        # Recovery metrics
        if "sleep_hours" in df.columns and "training_load" in df.columns:
            df["recovery_ratio"] = df["sleep_hours"] / (df["training_load"] + 1)

        if (
            "heart_rate_variability" in df.columns
            and "resting_heart_rate" in df.columns
        ):
            df["hrv_rhr_ratio"] = (
                df["heart_rate_variability"] / df["resting_heart_rate"]
            )

        # Cycle phase encoding (one-hot)
        # Always create all 4 phase columns for consistency
        if "cycle_phase" in df.columns:
            phase_dummies = pd.get_dummies(df["cycle_phase"], prefix="phase")
            # Ensure all phase columns exist, even if not in data
            for phase in ["menstrual", "follicular", "ovulatory", "luteal"]:
                col_name = f"phase_{phase}"
                if col_name not in phase_dummies.columns:
                    phase_dummies[col_name] = 0
            # Sort columns to ensure consistent order
            phase_cols = sorted(
                [col for col in phase_dummies.columns if col.startswith("phase_")]
            )
            phase_dummies = phase_dummies[phase_cols]
            df = pd.concat([df, phase_dummies], axis=1)

        # Interaction features
        if "energy_level" in df.columns and "training_load" in df.columns:
            df["energy_training_interaction"] = df["energy_level"] * df["training_load"]

        return df

    def prepare_features(
        self, df: pd.DataFrame, target_cols: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray], List[str]]:
        """
        Prepare features for machine learning, including scaling.

        Args:
            df: Input DataFrame with engineered features
            target_cols: Optional list of target column names

        Returns:
            Tuple of (feature array, target array, feature names)
        """
        df = df.copy()

        # Columns to exclude from features
        exclude_cols = ["date", "cycle_phase"]
        if target_cols:
            exclude_cols.extend(target_cols)

        # Select numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]

        X = df[feature_cols].values
        y = None
        if target_cols:
            y = (
                df[target_cols].values
                if len(target_cols) > 1
                else df[target_cols[0]].values
            )

        return X, y, feature_cols

    def fit_transform(
        self, df: pd.DataFrame, target_cols: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray], List[str]]:
        """
        Fit the preprocessor and transform the data.

        Args:
            df: Input DataFrame
            target_cols: Optional list of target column names

        Returns:
            Tuple of (scaled feature array, target array, feature names)
        """
        # Validate
        is_valid, errors = self.validate_data(df)
        if not is_valid:
            raise ValueError(f"Data validation failed: {errors}")

        # Clean and engineer
        df = self.handle_missing_values(df)
        df = self.engineer_features(df)

        # Prepare features
        X, y, feature_cols = self.prepare_features(df, target_cols)

        # Fit and transform
        X_scaled = self.scaler.fit_transform(X)

        self.feature_columns = feature_cols
        self.is_fitted = True

        return X_scaled, y, feature_cols

    def transform(
        self, df: pd.DataFrame, target_cols: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Transform new data using fitted preprocessor.

        Args:
            df: Input DataFrame
            target_cols: Optional list of target column names

        Returns:
            Tuple of (scaled feature array, target array)
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")

        # Validate
        is_valid, errors = self.validate_data(df)
        if not is_valid:
            raise ValueError(f"Data validation failed: {errors}")

        # Clean and engineer
        df = self.handle_missing_values(df)
        df = self.engineer_features(df)

        # Select exact same features as training in the same order
        missing_cols = [col for col in self.feature_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing feature columns: {missing_cols}")

        X = df[self.feature_columns].values

        # Extract targets if specified
        y = None
        if target_cols:
            y = (
                df[target_cols].values
                if len(target_cols) > 1
                else df[target_cols[0]].values
            )

        # Transform using fitted scaler
        X_scaled = self.scaler.transform(X)

        return X_scaled, y

    def get_feature_names(self) -> List[str]:
        """
        Get the list of feature names after preprocessing.

        Returns:
            List of feature names
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before getting feature names")
        return self.feature_columns
