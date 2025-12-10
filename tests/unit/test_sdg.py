"""Unit tests for Synthetic Data Generator."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from hypothesis import given, strategies as st

from astridml.sdg import SyntheticDataGenerator


class TestSyntheticDataGenerator:
    """Test suite for SyntheticDataGenerator."""

    def test_initialization(self):
        """Test SDG initialization."""
        sdg = SyntheticDataGenerator(seed=42)
        assert sdg is not None
        assert sdg.rng is not None

    def test_initialization_reproducibility(self):
        """Test that same seed produces same results."""
        sdg1 = SyntheticDataGenerator(seed=42)
        sdg2 = SyntheticDataGenerator(seed=42)

        data1 = sdg1.generate_wearable_data(n_days=10)
        data2 = sdg2.generate_wearable_data(n_days=10)

        pd.testing.assert_frame_equal(data1, data2)

    def test_get_cycle_phase(self):
        """Test cycle phase determination."""
        sdg = SyntheticDataGenerator()

        assert sdg._get_cycle_phase(1) == "menstrual"
        assert sdg._get_cycle_phase(3) == "menstrual"
        assert sdg._get_cycle_phase(10) == "follicular"
        assert sdg._get_cycle_phase(14) == "ovulatory"
        assert sdg._get_cycle_phase(20) == "luteal"
        assert sdg._get_cycle_phase(28) == "luteal"

        # Test cycle wrapping
        assert sdg._get_cycle_phase(29) == "menstrual"
        assert sdg._get_cycle_phase(35) == "follicular"

    def test_generate_wearable_data_shape(self):
        """Test wearable data generation produces correct shape."""
        sdg = SyntheticDataGenerator(seed=42)
        n_days = 30

        df = sdg.generate_wearable_data(n_days=n_days)

        assert len(df) == n_days
        assert "date" in df.columns
        assert "resting_heart_rate" in df.columns
        assert "heart_rate_variability" in df.columns
        assert "sleep_hours" in df.columns
        assert "cycle_phase" in df.columns

    def test_generate_wearable_data_types(self):
        """Test wearable data has correct data types."""
        sdg = SyntheticDataGenerator(seed=42)
        df = sdg.generate_wearable_data(n_days=10)

        assert df["resting_heart_rate"].dtype == np.float64
        assert df["steps"].dtype in [np.int64, np.int32]
        assert df["sleep_hours"].dtype == np.float64
        assert df["cycle_phase"].dtype == object

    def test_generate_wearable_data_ranges(self):
        """Test wearable data values are in realistic ranges."""
        sdg = SyntheticDataGenerator(seed=42)
        df = sdg.generate_wearable_data(n_days=90)

        # Check realistic ranges
        assert df["resting_heart_rate"].between(40, 100).all()
        assert df["heart_rate_variability"].between(20, 150).all()
        assert df["sleep_hours"].between(3, 12).all()
        assert df["sleep_quality_score"].between(0, 100).all()
        assert df["steps"].ge(0).all()
        assert df["active_minutes"].ge(0).all()
        assert df["calories_burned"].gt(0).all()
        assert df["training_load"].ge(0).all()
        assert df["cycle_day"].between(1, 28).all()

    def test_generate_symptom_data_shape(self):
        """Test symptom data generation produces correct shape."""
        sdg = SyntheticDataGenerator(seed=42)
        n_days = 30

        df = sdg.generate_symptom_data(n_days=n_days)

        assert len(df) == n_days
        assert "date" in df.columns
        assert "energy_level" in df.columns
        assert "mood_score" in df.columns
        assert "pain_level" in df.columns
        assert "cycle_phase" in df.columns

    def test_generate_symptom_data_ranges(self):
        """Test symptom data values are in valid ranges."""
        sdg = SyntheticDataGenerator(seed=42)
        df = sdg.generate_symptom_data(n_days=90)

        assert df["energy_level"].between(1, 10).all()
        assert df["mood_score"].between(1, 10).all()
        assert df["pain_level"].between(0, 10).all()
        assert df["bloating"].between(0, 10).all()
        assert df["breast_tenderness"].between(0, 10).all()
        assert df["cycle_day"].between(1, 28).all()
        assert df["flow_level"].between(0, 5).all()

    def test_menstruation_correlation(self):
        """Test that flow_level correlates with menstrual phase."""
        sdg = SyntheticDataGenerator(seed=42)
        df = sdg.generate_symptom_data(n_days=90)

        # Flow should only occur during menstrual phase
        menstrual_rows = df[df["cycle_phase"] == "menstrual"]
        non_menstrual_rows = df[df["cycle_phase"] != "menstrual"]

        assert menstrual_rows["flow_level"].sum() > 0
        assert non_menstrual_rows["flow_level"].sum() == 0

    def test_is_menstruating_flag(self):
        """Test that is_menstruating flag matches cycle day."""
        sdg = SyntheticDataGenerator(seed=42)
        df = sdg.generate_symptom_data(n_days=90)

        # is_menstruating should be True for cycle days 1-5
        for _, row in df.iterrows():
            expected = 1 <= row["cycle_day"] <= 5
            assert row["is_menstruating"] == expected

    def test_generate_combined_data(self):
        """Test combined data generation."""
        sdg = SyntheticDataGenerator(seed=42)
        n_days = 30

        df = sdg.generate_combined_data(n_days=n_days)

        assert len(df) == n_days

        # Check both wearable and symptom columns present
        assert "resting_heart_rate" in df.columns
        assert "energy_level" in df.columns
        assert "cycle_phase" in df.columns

        # Check no duplicate columns
        assert len(df.columns) == len(set(df.columns))

    def test_combined_data_consistency(self):
        """Test that combined data has consistent cycle information."""
        sdg = SyntheticDataGenerator(seed=42)
        df = sdg.generate_combined_data(n_days=60)

        # Each date should have matching cycle_day and cycle_phase
        for _, row in df.iterrows():
            phase = row["cycle_phase"]
            day = row["cycle_day"]

            if phase == "menstrual":
                assert 1 <= day <= 5
            elif phase == "follicular":
                assert 6 <= day <= 13
            elif phase == "ovulatory":
                assert 14 <= day <= 16
            elif phase == "luteal":
                assert 17 <= day <= 28

    def test_custom_start_date(self):
        """Test generation with custom start date."""
        sdg = SyntheticDataGenerator(seed=42)
        start_date = datetime(2024, 1, 1)

        df = sdg.generate_wearable_data(n_days=10, start_date=start_date)

        first_date = pd.to_datetime(df["date"].iloc[0])
        assert first_date.year == 2024
        assert first_date.month == 1
        assert first_date.day == 1

    def test_date_sequence(self):
        """Test that dates are sequential."""
        sdg = SyntheticDataGenerator(seed=42)
        df = sdg.generate_wearable_data(n_days=30)

        dates = pd.to_datetime(df["date"])
        date_diffs = dates.diff().dropna()

        # All differences should be 1 day
        assert (date_diffs == timedelta(days=1)).all()

    @given(
        n_days=st.integers(min_value=1, max_value=365),
        seed=st.integers(min_value=0, max_value=10000),
    )
    def test_property_valid_generation(self, n_days, seed):
        """Property-based test: generation should always produce valid data."""
        sdg = SyntheticDataGenerator(seed=seed)

        df_wearable = sdg.generate_wearable_data(n_days=n_days)
        df_symptom = sdg.generate_symptom_data(n_days=n_days)

        # Check shape
        assert len(df_wearable) == n_days
        assert len(df_symptom) == n_days

        # Check no NaNs
        assert not df_wearable.isnull().any().any()
        assert not df_symptom.isnull().any().any()

        # Check cycle consistency
        assert df_wearable["cycle_day"].between(1, 28).all()
        assert df_symptom["cycle_day"].between(1, 28).all()

    def test_cycle_phase_modifiers(self):
        """Test that cycle phase modifiers are applied correctly."""
        sdg = SyntheticDataGenerator(seed=42)

        # Get modifiers for each phase
        menstrual_mods = sdg._generate_cycle_modifiers("menstrual")
        follicular_mods = sdg._generate_cycle_modifiers("follicular")
        ovulatory_mods = sdg._generate_cycle_modifiers("ovulatory")
        luteal_mods = sdg._generate_cycle_modifiers("luteal")

        # Check that modifiers exist
        assert "energy" in menstrual_mods
        assert "performance" in follicular_mods
        assert "mood" in ovulatory_mods
        assert "pain" in luteal_mods

        # Follicular/ovulatory should have positive energy modifiers
        assert follicular_mods["energy"] > 0
        assert ovulatory_mods["energy"] > 0

        # Menstrual should have negative energy modifier
        assert menstrual_mods["energy"] < 0

    def test_no_negative_values(self):
        """Test that no metric has negative values where inappropriate."""
        sdg = SyntheticDataGenerator(seed=42)
        df = sdg.generate_combined_data(n_days=100)

        # These should never be negative
        assert (df["steps"] >= 0).all()
        assert (df["active_minutes"] >= 0).all()
        assert (df["calories_burned"] >= 0).all()
        assert (df["training_load"] >= 0).all()
        assert (df["sleep_hours"] >= 0).all()
        assert (df["pain_level"] >= 0).all()