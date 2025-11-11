"""Data Preprocessing Module for cleaning and feature engineering."""

import numpy as np
import pandas as pd
from typing import Optional, List, Tuple
from sklearn.preprocessing import StandardScaler


class DataPreprocessor:
    """
    Preprocesses wearable and symptom data for machine learning models.

    This class handles the complete preprocessing pipeline including data validation,
    missing value imputation, feature engineering, and standardization. It follows
    a fit-transform pattern similar to scikit-learn transformers.

    Attributes
    ----------
    scaler : StandardScaler
        StandardScaler instance for normalizing features.
    feature_columns : Optional[List[str]]
        List of feature column names after preprocessing. None until fitted.
    is_fitted : bool
        Flag indicating whether the preprocessor has been fitted.

    Examples
    --------
    >>> import pandas as pd
    >>> from astridml.dpm import DataPreprocessor
    >>> 
    >>> # Create sample data
    >>> data = pd.DataFrame({
    ...     'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
    ...     'cycle_day': [1, 2, 3],
    ...     'cycle_phase': ['menstrual', 'menstrual', 'follicular'],
    ...     'resting_heart_rate': [65.0, 66.0, 64.0],
    ...     'energy_level': [6.0, 7.0, 8.0],
    ...     'mood_score': [5.0, 6.0, 7.0]
    ... })
    >>> 
    >>> # Initialize and fit-transform
    >>> preprocessor = DataPreprocessor()
    >>> X, y, feature_names = preprocessor.fit_transform(
    ...     data, 
    ...     target_cols=['energy_level', 'mood_score']
    ... )
    >>> print(f"Features shape: {X.shape}, Feature names: {len(feature_names)}")
    Features shape: (3, N), Feature names: N
    >>> 
    >>> # Transform new data
    >>> new_data = pd.DataFrame({...})  # Similar structure
    >>> X_new, y_new = preprocessor.transform(new_data)
    """

    def __init__(self):
        """
        Initialize the data preprocessor.

        Creates a new DataPreprocessor instance with an unfitted state.
        The scaler and feature columns will be initialized when fit_transform
        is called for the first time.

        Returns
        -------
        DataPreprocessor
            A new DataPreprocessor instance.

        Examples
        --------
        >>> preprocessor = DataPreprocessor()
        >>> print(preprocessor.is_fitted)
        False
        """
        self.scaler = StandardScaler()
        self.feature_columns: Optional[List[str]] = None
        self.is_fitted = False

    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate input data format and required columns.

        Checks that the input DataFrame contains required columns ('date', 'cycle_day',
        'cycle_phase'), that dates are in a valid format, and that at least one
        numeric column exists.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame to validate. Must contain 'date', 'cycle_day', and
            'cycle_phase' columns.

        Returns
        -------
        Tuple[bool, List[str]]
            A tuple containing:
            - is_valid : bool
                True if data passes all validation checks, False otherwise.
            - errors : List[str]
                List of error messages describing validation failures. Empty list
                if validation passes.

        Raises
        ------
        None
            This method does not raise exceptions; errors are returned in the
            error list.

        Examples
        --------
        >>> import pandas as pd
        >>> preprocessor = DataPreprocessor()
        >>> 
        >>> # Valid data
        >>> valid_df = pd.DataFrame({
        ...     'date': ['2024-01-01'],
        ...     'cycle_day': [1],
        ...     'cycle_phase': ['menstrual'],
        ...     'resting_heart_rate': [65.0]
        ... })
        >>> is_valid, errors = preprocessor.validate_data(valid_df)
        >>> print(is_valid, errors)
        True []
        >>> 
        >>> # Invalid data (missing required column)
        >>> invalid_df = pd.DataFrame({'date': ['2024-01-01']})
        >>> is_valid, errors = preprocessor.validate_data(invalid_df)
        >>> print(is_valid, errors)
        False ['Missing required columns: [\'cycle_day\', \'cycle_phase\']']
        """
        errors = []

        required_cols = ['date', 'cycle_day', 'cycle_phase']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")

        if 'date' in df.columns:
            try:
                pd.to_datetime(df['date'])
            except Exception as e:
                errors.append(f"Invalid date format: {e}")

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            errors.append("No numeric columns found")

        return len(errors) == 0, errors

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values using appropriate imputation strategies.

        Applies different imputation strategies based on column type:
        - Time series columns (heart rate, sleep metrics): forward fill then
          backward fill
        - Count-based metrics (steps, active minutes): fill with 0
        - Other numeric columns: fill with column mean

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame that may contain missing values. Will not be modified
            in place; a copy is returned.

        Returns
        -------
        pd.DataFrame
            DataFrame with all missing values imputed. Original DataFrame is not
            modified.

        Raises
        ------
        None
            This method does not raise exceptions. If all values in a column are
            missing, mean imputation will result in NaN, which may cause issues
            downstream.

        Notes
        -----
        The following columns use forward/backward fill:
        - resting_heart_rate
        - heart_rate_variability
        - sleep_hours
        - sleep_quality_score

        The following columns are filled with 0:
        - steps
        - active_minutes
        - calories_burned

        All other numeric columns use mean imputation.
        """
        df = df.copy()

        # Forward fill for time series data (suitable for vital signs)
        time_series_cols = [
            'resting_heart_rate', 'heart_rate_variability',
            'sleep_hours', 'sleep_quality_score'
        ]
        for col in time_series_cols:
            if col in df.columns:
                df[col] = df[col].ffill().bfill()

        # Fill zero for count-based metrics
        count_cols = ['steps', 'active_minutes', 'calories_burned']
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

        Generates temporal features, rolling statistics, trend indicators,
        recovery metrics, cycle phase encodings, and interaction features.
        The input DataFrame is sorted by date before feature engineering.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with raw features. Must contain a 'date' column
            that can be converted to datetime. Will not be modified in place.

        Returns
        -------
        pd.DataFrame
            DataFrame with engineered features added. Original columns are
            preserved. New features include:
            - Temporal: day_of_week, is_weekend
            - Rolling statistics: *_rolling_7d, *_rolling_7d_std
            - Trends: *_trend (difference from previous day)
            - Recovery: recovery_ratio, hrv_rhr_ratio
            - Cycle phase: phase_* (one-hot encoded)
            - Interactions: energy_training_interaction

        Raises
        ------
        ValueError
            If 'date' column cannot be converted to datetime format.

        Notes
        -----
        The DataFrame is sorted by date and reset_index is called, so the
        original index is lost. Rolling windows use min_periods=1 to handle
        short time series.
        """
        df = df.copy()

        # Ensure date is datetime
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)

        # Temporal features
        df['day_of_week'] = df['date'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

        # Rolling averages (7-day windows)
        rolling_cols = [
            'resting_heart_rate', 'heart_rate_variability',
            'sleep_hours', 'sleep_quality_score', 'training_load',
            'energy_level', 'mood_score', 'pain_level'
        ]

        for col in rolling_cols:
            if col in df.columns:
                df[f'{col}_rolling_7d'] = df[col].rolling(window=7, min_periods=1).mean()
                df[f'{col}_rolling_7d_std'] = df[col].rolling(window=7, min_periods=1).std().fillna(0)

        # Trend features (difference from previous day)
        trend_cols = [
            'resting_heart_rate', 'heart_rate_variability',
            'sleep_quality_score', 'energy_level', 'mood_score'
        ]

        for col in trend_cols:
            if col in df.columns:
                df[f'{col}_trend'] = df[col].diff().fillna(0)

        # Recovery metrics
        if 'sleep_hours' in df.columns and 'training_load' in df.columns:
            df['recovery_ratio'] = df['sleep_hours'] / (df['training_load'] + 1)

        if 'heart_rate_variability' in df.columns and 'resting_heart_rate' in df.columns:
            df['hrv_rhr_ratio'] = df['heart_rate_variability'] / df['resting_heart_rate']

        # Cycle phase encoding (one-hot)
        if 'cycle_phase' in df.columns:
            phase_dummies = pd.get_dummies(df['cycle_phase'], prefix='phase')
            df = pd.concat([df, phase_dummies], axis=1)

        # Interaction features
        if 'energy_level' in df.columns and 'training_load' in df.columns:
            df['energy_training_interaction'] = df['energy_level'] * df['training_load']

        return df

    def prepare_features(
        self,
        df: pd.DataFrame,
        target_cols: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray], List[str]]:
        """
        Prepare features for machine learning by selecting and extracting numeric columns.

        Separates features from targets, excluding date, cycle_phase, and target
        columns from the feature set. Returns arrays suitable for scikit-learn
        or TensorFlow models.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with engineered features. Should already have been
            processed through engineer_features().
        target_cols : Optional[List[str]], default=None
            List of column names to use as target variables. These columns will
            be excluded from features and returned separately. If None, no target
            array is returned.

        Returns
        -------
        Tuple[np.ndarray, Optional[np.ndarray], List[str]]
            A tuple containing:
            - X : np.ndarray
                Feature array of shape (n_samples, n_features) with all numeric
                columns except excluded ones.
            - y : Optional[np.ndarray]
                Target array of shape (n_samples,) or (n_samples, n_targets).
                None if target_cols is None.
            - feature_cols : List[str]
                List of feature column names in the same order as X columns.

        Raises
        ------
        KeyError
            If any column in target_cols is not present in df.
        ValueError
            If no numeric feature columns remain after exclusions.

        Notes
        -----
        The 'date' and 'cycle_phase' columns are always excluded from features.
        Only numeric columns are included in the feature array.
        """
        df = df.copy()

        # Columns to exclude from features
        exclude_cols = ['date', 'cycle_phase']
        if target_cols:
            exclude_cols.extend(target_cols)

        # Select numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]

        X = df[feature_cols].values
        y = None
        if target_cols:
            y = df[target_cols].values if len(target_cols) > 1 else df[target_cols[0]].values

        return X, y, feature_cols

    def fit_transform(
        self,
        df: pd.DataFrame,
        target_cols: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray], List[str]]:
        """
        Fit the preprocessor and transform the data in one step.

        This is the main method for preprocessing training data. It validates the
        input, handles missing values, engineers features, prepares feature arrays,
        and fits the standard scaler. The preprocessor state is saved for use
        with transform() on new data.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with raw wearable and symptom data. Must contain
            'date', 'cycle_day', and 'cycle_phase' columns, plus numeric features.
        target_cols : Optional[List[str]], default=None
            List of column names to use as target variables for supervised
            learning. If None, only features are returned.

        Returns
        -------
        Tuple[np.ndarray, Optional[np.ndarray], List[str]]
            A tuple containing:
            - X_scaled : np.ndarray
                Scaled feature array of shape (n_samples, n_features) with
                zero mean and unit variance.
            - y : Optional[np.ndarray]
                Target array of shape (n_samples,) or (n_samples, n_targets).
                None if target_cols is None.
            - feature_cols : List[str]
                List of feature column names in the same order as X_scaled columns.

        Raises
        ------
        ValueError
            If data validation fails (missing required columns, invalid date format,
            or no numeric columns). The error message includes details about
            validation failures.

        Examples
        --------
        >>> import pandas as pd
        >>> from astridml.dpm import DataPreprocessor
        >>> 
        >>> # Prepare training data
        >>> train_data = pd.DataFrame({
        ...     'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
        ...     'cycle_day': [1, 2, 3],
        ...     'cycle_phase': ['menstrual', 'menstrual', 'follicular'],
        ...     'resting_heart_rate': [65.0, 66.0, 64.0],
        ...     'energy_level': [6.0, 7.0, 8.0],
        ...     'mood_score': [5.0, 6.0, 7.0]
        ... })
        >>> 
        >>> preprocessor = DataPreprocessor()
        >>> X, y, feature_names = preprocessor.fit_transform(
        ...     train_data,
        ...     target_cols=['energy_level', 'mood_score']
        ... )
        >>> print(f"X shape: {X.shape}, y shape: {y.shape}")
        X shape: (3, N), y shape: (3, 2)
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
        self,
        df: pd.DataFrame,
        target_cols: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Transform new data using a fitted preprocessor.

        Applies the same preprocessing pipeline (validation, missing value handling,
        feature engineering, scaling) to new data using the scaler and feature
        columns learned during fit_transform(). The preprocessor must be fitted
        before calling this method.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with raw wearable and symptom data. Must have the
            same structure as training data and contain the same feature columns
            (after engineering) that were present during fitting.
        target_cols : Optional[List[str]], default=None
            List of column names to extract as target variables. Should match
            the target columns used during training if applicable.

        Returns
        -------
        Tuple[np.ndarray, Optional[np.ndarray]]
            A tuple containing:
            - X_scaled : np.ndarray
                Scaled feature array of shape (n_samples, n_features) using
                the scaler fitted on training data.
            - y : Optional[np.ndarray]
                Target array of shape (n_samples,) or (n_samples, n_targets).
                None if target_cols is None.

        Raises
        ------
        ValueError
            If preprocessor has not been fitted (is_fitted is False).
            If data validation fails (missing required columns, invalid date format,
            or no numeric columns).
            If the number of features in the input data does not match the number
            of features used during fitting (feature mismatch).

        Notes
        -----
        The feature engineering step may create different features if optional
        columns are missing, which can cause feature mismatch errors. Ensure
        input data has the same structure as training data.
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

        # Prepare features with same columns as training
        X, y, _ = self.prepare_features(df, target_cols)

        # Ensure same features as training
        if X.shape[1] != len(self.feature_columns):
            raise ValueError(
                f"Feature mismatch: expected {len(self.feature_columns)}, got {X.shape[1]}"
            )

        # Transform
        X_scaled = self.scaler.transform(X)

        return X_scaled, y

    def get_feature_names(self) -> List[str]:
        """
        Get the list of feature names after preprocessing.

        Returns the feature column names that were learned during fit_transform().
        These names correspond to the columns in the feature arrays returned by
        fit_transform() and transform().

        Parameters
        ----------
        None

        Returns
        -------
        List[str]
            List of feature column names in the order they appear in the feature
            arrays. The list corresponds to the columns of X_scaled returned by
            fit_transform() or transform().

        Raises
        ------
        ValueError
            If preprocessor has not been fitted (is_fitted is False). Call
            fit_transform() first.

        Examples
        --------
        >>> import pandas as pd
        >>> from astridml.dpm import DataPreprocessor
        >>> 
        >>> preprocessor = DataPreprocessor()
        >>> data = pd.DataFrame({...})  # Training data
        >>> X, y, _ = preprocessor.fit_transform(data)
        >>> 
        >>> # Get feature names
        >>> feature_names = preprocessor.get_feature_names()
        >>> print(f"Number of features: {len(feature_names)}")
        >>> print(f"First 5 features: {feature_names[:5]}")
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before getting feature names")
        return self.feature_columns