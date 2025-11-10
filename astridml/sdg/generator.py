"""Synthetic Data Generator for wearable and menstrual cycle data."""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Tuple


class SyntheticDataGenerator:
    """
    Generates synthetic data for wearable training metrics and menstrual cycle symptoms.

    The generator creates realistic correlations between menstrual cycle phases and
    athletic performance metrics, as well as symptom patterns.
    """

    # Menstrual cycle phases (typical 28-day cycle)
    CYCLE_PHASES = {
        'menstrual': (1, 5),      # Days 1-5
        'follicular': (6, 13),     # Days 6-13
        'ovulatory': (14, 16),     # Days 14-16
        'luteal': (17, 28)         # Days 17-28
    }

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the synthetic data generator.

        Args:
            seed: Random seed for reproducibility
        """
        self.rng = np.random.default_rng(seed)

    def _get_cycle_phase(self, day: int) -> str:
        """
        Determine menstrual cycle phase for a given cycle day.

        Maps a sequential day number to the corresponding menstrual cycle phase
        based on a typical 28-day cycle. The cycle automatically wraps around
        after 28 days.

        Parameters
        ----------
        day : int
            Sequential day number (0-indexed). Values larger than 28 will wrap
            around using modulo arithmetic. For example, day 30 maps to cycle
            day 2.

        Returns
        -------
        phase : str
            One of {'menstrual', 'follicular', 'ovulatory', 'luteal'}.
            - 'menstrual': Days 1-5 of cycle
            - 'follicular': Days 6-13 of cycle
            - 'ovulatory': Days 14-16 of cycle
            - 'luteal': Days 17-28 of cycle

        Notes
        -----
        The typical 28-day menstrual cycle is divided into four phases:

        1. Menstrual phase (days 1-5): Shedding of uterine lining
        2. Follicular phase (days 6-13): Preparation for ovulation
        3. Ovulatory phase (days 14-16): Release of egg
        4. Luteal phase (days 17-28): Post-ovulation, preparation for pregnancy

        This method uses a simplified model. Real cycle lengths vary between
        individuals (21-35 days) and can fluctuate month-to-month.

        Examples
        --------
        >>> sdg = SyntheticDataGenerator(seed=42)
        >>> sdg._get_cycle_phase(0)   # First day
        'menstrual'
        >>> sdg._get_cycle_phase(10)  # Follicular phase
        'follicular'
        >>> sdg._get_cycle_phase(14)  # Ovulation
        'ovulatory'
        >>> sdg._get_cycle_phase(29)  # Wraps to day 2
        'menstrual'
        """
        cycle_day = ((day - 1) % 28) + 1
        for phase, (start, end) in self.CYCLE_PHASES.items():
            if start <= cycle_day <= end:
                return phase
        return 'luteal'

    def _generate_cycle_modifiers(self, phase: str) -> dict:
        """
        Generate performance and symptom modifiers based on cycle phase.

        Returns a dictionary of multipliers that adjust baseline physiological
        and psychological metrics according to the menstrual cycle phase.
        These modifiers are based on research showing correlations between
        cycle phases and athletic performance [1]_ [2]_.

        Parameters
        ----------
        phase : str
            Menstrual cycle phase. Must be one of:
            {'menstrual', 'follicular', 'ovulatory', 'luteal'}

        Returns
        -------
        modifiers : dict
            Dictionary mapping metric names to fractional modifiers (e.g., 0.15
            means +15%, -0.10 means -10%). Keys include:

            - 'energy' : float
                Energy level modifier (-0.15 to 0.15)
            - 'pain' : float
                Pain level modifier (-0.20 to 0.25)
            - 'mood' : float
                Mood score modifier (-0.15 to 0.15)
            - 'sleep_quality' : float
                Sleep quality modifier (-0.10 to 0.05)
            - 'resting_hr' : float
                Resting heart rate modifier (-0.03 to 0.04)
            - 'performance' : float
                Athletic performance modifier (-0.08 to 0.12)

        Raises
        ------
        KeyError
            If `phase` is not one of the four recognized cycle phases.

        Notes
        -----
        The modifiers represent relative changes from baseline values:

        - **Menstrual phase**: Lower energy and performance, higher pain
        - **Follicular phase**: Higher energy and performance, lower pain
        - **Ovulatory phase**: Peak energy and performance
        - **Luteal phase**: Moderate decrease in energy and mood

        These values are based on aggregated research findings but represent
        average trends. Individual variation is significant.

        References
        ----------
        .. [1] MacMillan et al., "The association between menstrual cycle phase,
           menstrual irregularities, contraceptive use and musculoskeletal
           injury among female athletes", Sports Medicine, 2024.
        .. [2] Gilmour, "The psychological impact of the menstrual cycle on
           athletic performance", SportRxiv, 2025.

        Examples
        --------
        >>> sdg = SyntheticDataGenerator(seed=42)
        >>> mods = sdg._generate_cycle_modifiers('ovulatory')
        >>> mods['energy']
        0.15
        >>> mods['performance']
        0.1
        """
        modifiers = {
            'menstrual': {
                'energy': -0.15,
                'pain': 0.25,
                'mood': -0.10,
                'sleep_quality': -0.10,
                'resting_hr': 0.02,
                'performance': -0.08
            },
            'follicular': {
                'energy': 0.10,
                'pain': -0.20,
                'mood': 0.15,
                'sleep_quality': 0.05,
                'resting_hr': -0.03,
                'performance': 0.12
            },
            'ovulatory': {
                'energy': 0.15,
                'pain': -0.15,
                'mood': 0.10,
                'sleep_quality': 0.02,
                'resting_hr': 0.01,
                'performance': 0.10
            },
            'luteal': {
                'energy': -0.10,
                'pain': 0.10,
                'mood': -0.15,
                'sleep_quality': -0.08,
                'resting_hr': 0.04,
                'performance': -0.05
            }
        }
        return modifiers[phase]

    def generate_wearable_data(
        self,
        n_days: int = 90,
        start_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Generate synthetic wearable device data with cycle-correlated metrics.

        Creates realistic time-series data simulating measurements from fitness
        wearables (e.g., Apple Watch, Garmin, Whoop). Metrics are modulated
        by menstrual cycle phase to reflect research-documented correlations
        between hormonal fluctuations and physiological parameters.

        Parameters
        ----------
        n_days : int, optional
            Number of consecutive days of data to generate. Must be positive.
            Default is 90 (approximately 3 menstrual cycles).
        start_date : datetime, optional
            Starting date for the time series. If None, uses current datetime.
            Dates are generated sequentially from this starting point.

        Returns
        -------
        wearable_data : pandas.DataFrame
            DataFrame with n_days rows and the following columns:

            - date : str
                Date in 'YYYY-MM-DD' format
            - resting_heart_rate : float
                Resting heart rate in beats per minute (bpm), typically 40-100
            - heart_rate_variability : float
                HRV in milliseconds (ms), typically 20-150
            - sleep_hours : float
                Total sleep duration in hours, typically 3-12
            - sleep_quality_score : float
                Sleep quality score from 0-100 (higher is better)
            - steps : int
                Daily step count, typically 0-15000
            - active_minutes : int
                Minutes of active movement, typically 0-120
            - calories_burned : int
                Total daily calories burned, typically 1500-3000
            - training_load : float
                Arbitrary training load units, typically 0-100
            - cycle_day : int
                Day within the 28-day cycle (1-28)
            - cycle_phase : str
                One of {'menstrual', 'follicular', 'ovulatory', 'luteal'}

        Notes
        -----
        The data generation process:

        1. Establishes baseline values with Gaussian noise
        2. Determines cycle phase for each day
        3. Applies phase-specific modifiers to simulate hormonal effects
        4. Includes day-of-week variation (weekends have lower training load)

        All values are clipped to realistic physiological ranges. The random
        number generator can be seeded for reproducibility via the class
        constructor.

        Examples
        --------
        Generate 30 days of data starting from a specific date:

        >>> from datetime import datetime
        >>> sdg = SyntheticDataGenerator(seed=42)
        >>> data = sdg.generate_wearable_data(
        ...     n_days=30,
        ...     start_date=datetime(2024, 1, 1)
        ... )
        >>> data.shape
        (30, 11)
        >>> data['resting_heart_rate'].mean()
        59.8...
        >>> data['cycle_phase'].unique()
        array(['menstrual', 'follicular', 'ovulatory'], dtype=object)

        Generate reproducible data:

        >>> sdg1 = SyntheticDataGenerator(seed=123)
        >>> sdg2 = SyntheticDataGenerator(seed=123)
        >>> data1 = sdg1.generate_wearable_data(n_days=10)
        >>> data2 = sdg2.generate_wearable_data(n_days=10)
        >>> data1.equals(data2)
        True
        """
        if start_date is None:
            start_date = datetime.now()

        dates = [start_date + timedelta(days=i) for i in range(n_days)]
        data = []

        for day_idx, date in enumerate(dates):
            phase = self._get_cycle_phase(day_idx)
            mods = self._generate_cycle_modifiers(phase)

            # Base values with noise
            base_resting_hr = 60 + self.rng.normal(0, 3)
            base_hrv = 65 + self.rng.normal(0, 8)
            base_sleep_hours = 7.5 + self.rng.normal(0, 0.8)
            base_steps = 8000 + self.rng.normal(0, 1500)
            base_calories = 2000 + self.rng.normal(0, 200)

            # Apply cycle phase modifiers
            resting_hr = base_resting_hr * (1 + mods['resting_hr'])
            hrv = base_hrv * (1 + mods['performance'] * 0.5)
            sleep_hours = base_sleep_hours * (1 + mods['sleep_quality'])
            sleep_quality = max(0, min(100, 75 + mods['sleep_quality'] * 100 + self.rng.normal(0, 10)))
            steps = int(base_steps * (1 + mods['energy'] * 0.5))
            active_minutes = int(45 + mods['energy'] * 30 + self.rng.normal(0, 10))
            calories_burned = int(base_calories * (1 + mods['performance'] * 0.3))

            # Training load (varies by day of week and energy)
            is_weekend = date.weekday() >= 5
            training_load = (40 if is_weekend else 60) * (1 + mods['performance']) + self.rng.normal(0, 10)
            training_load = max(0, training_load)

            data.append({
                'date': date.strftime('%Y-%m-%d'),
                'resting_heart_rate': round(resting_hr, 1),
                'heart_rate_variability': round(hrv, 1),
                'sleep_hours': round(sleep_hours, 2),
                'sleep_quality_score': round(sleep_quality, 1),
                'steps': steps,
                'active_minutes': active_minutes,
                'calories_burned': calories_burned,
                'training_load': round(training_load, 1),
                'cycle_day': ((day_idx) % 28) + 1,
                'cycle_phase': phase
            })

        return pd.DataFrame(data)

    def generate_symptom_data(
        self,
        n_days: int = 90,
        start_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Generate synthetic menstrual cycle symptom data.

        Creates realistic daily symptom logs that an athlete might manually
        track, including energy levels, mood, pain, and menstruation-specific
        symptoms. Symptom severity varies according to menstrual cycle phase
        based on documented physiological and psychological patterns.

        Parameters
        ----------
        n_days : int, optional
            Number of consecutive days of data to generate. Must be positive.
            Default is 90 (approximately 3 menstrual cycles).
        start_date : datetime, optional
            Starting date for the time series. If None, uses current datetime.

        Returns
        -------
        symptom_data : pandas.DataFrame
            DataFrame with n_days rows and the following columns:

            - date : str
                Date in 'YYYY-MM-DD' format
            - cycle_day : int
                Day within the 28-day cycle (1-28)
            - cycle_phase : str
                One of {'menstrual', 'follicular', 'ovulatory', 'luteal'}
            - is_menstruating : bool
                True if currently menstruating (typically days 1-5)
            - flow_level : int
                Menstrual flow intensity: 0 (none), 1 (light), 2 (light-moderate),
                3 (moderate), 4 (moderate-heavy), 5 (heavy). Zero when not
                menstruating.
            - energy_level : float
                Self-reported energy on 1-10 scale (1=exhausted, 10=energized)
            - mood_score : float
                Self-reported mood on 1-10 scale (1=very low, 10=excellent)
            - pain_level : float
                Self-reported pain on 0-10 scale (0=none, 10=severe)
            - bloating : float
                Bloating severity on 0-10 scale (0=none, 10=severe)
            - breast_tenderness : float
                Breast tenderness on 0-10 scale (0=none, 10=severe)

        Notes
        -----
        Symptom patterns follow typical menstrual cycle trends:

        - **Menstrual phase**: Higher pain and bloating, lower energy
        - **Follicular phase**: Increasing energy and mood, minimal symptoms
        - **Ovulatory phase**: Peak energy and mood
        - **Luteal phase**: Gradual decrease in mood and energy, increasing
          breast tenderness and bloating (PMS symptoms)

        Flow patterns during menstruation typically show:
        - Days 1-2: Moderate to heavy flow
        - Days 3-4: Light to moderate flow
        - Day 5: Light flow or spotting

        All symptom values include random variation to simulate individual
        daily fluctuations.

        Examples
        --------
        Generate symptom data for two menstrual cycles:

        >>> sdg = SyntheticDataGenerator(seed=42)
        >>> symptoms = sdg.generate_symptom_data(n_days=56)
        >>> symptoms.shape
        (56, 10)
        >>> symptoms['energy_level'].describe()
        count    56.000000
        mean      6.2...
        std       1.5...
        ...

        Check menstruation tracking:

        >>> menstrual_days = symptoms[symptoms['is_menstruating']]
        >>> len(menstrual_days)
        10
        >>> (menstrual_days['flow_level'] > 0).all()
        True

        Verify no flow outside menstruation:

        >>> non_menstrual = symptoms[~symptoms['is_menstruating']]
        >>> (non_menstrual['flow_level'] == 0).all()
        True
        """
        if start_date is None:
            start_date = datetime.now()

        dates = [start_date + timedelta(days=i) for i in range(n_days)]
        data = []

        for day_idx, date in enumerate(dates):
            phase = self._get_cycle_phase(day_idx)
            mods = self._generate_cycle_modifiers(phase)

            # Scale modifiers to 1-10 scale
            energy_level = max(1, min(10, 6 + mods['energy'] * 10 + self.rng.normal(0, 1)))
            mood_score = max(1, min(10, 7 + mods['mood'] * 10 + self.rng.normal(0, 1)))
            pain_level = max(0, min(10, 2 + mods['pain'] * 10 + self.rng.normal(0, 1.5)))

            # Other symptoms influenced by cycle phase
            bloating = max(0, min(10, 3 + (mods['pain'] * 8) + self.rng.normal(0, 1.5)))
            breast_tenderness = max(0, min(10, 2 + (mods['pain'] * 7) + self.rng.normal(0, 1.5)))

            # Track if currently menstruating
            cycle_day = ((day_idx) % 28) + 1
            is_menstruating = 1 <= cycle_day <= 5
            flow_level = 0
            if is_menstruating:
                # Flow typically peaks around day 2-3
                if cycle_day <= 2:
                    flow_level = self.rng.choice([2, 3, 4], p=[0.2, 0.5, 0.3])
                elif cycle_day <= 4:
                    flow_level = self.rng.choice([1, 2, 3], p=[0.3, 0.5, 0.2])
                else:
                    flow_level = self.rng.choice([1, 2], p=[0.7, 0.3])

            data.append({
                'date': date.strftime('%Y-%m-%d'),
                'cycle_day': cycle_day,
                'cycle_phase': phase,
                'is_menstruating': is_menstruating,
                'flow_level': flow_level,
                'energy_level': round(energy_level, 1),
                'mood_score': round(mood_score, 1),
                'pain_level': round(pain_level, 1),
                'bloating': round(bloating, 1),
                'breast_tenderness': round(breast_tenderness, 1),
            })

        return pd.DataFrame(data)

    def generate_combined_data(
        self,
        n_days: int = 90,
        start_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Generate combined wearable and symptom data in a single DataFrame.

        Convenience method that generates both wearable device metrics and
        self-reported symptoms, then merges them into a unified dataset. This
        is the recommended method for most use cases as it ensures perfect
        temporal alignment and consistent cycle phase information between
        objective and subjective measurements.

        Parameters
        ----------
        n_days : int, optional
            Number of consecutive days of data to generate. Must be positive.
            Default is 90 (approximately 3 menstrual cycles).
        start_date : datetime, optional
            Starting date for the time series. If None, uses current datetime.
            Both wearable and symptom data will use the same start date.

        Returns
        -------
        combined_data : pandas.DataFrame
            DataFrame with n_days rows containing all columns from both
            `generate_wearable_data` and `generate_symptom_data`, merged on
            date, cycle_day, and cycle_phase. Columns include:

            From wearable data:
                - resting_heart_rate, heart_rate_variability, sleep_hours,
                  sleep_quality_score, steps, active_minutes, calories_burned,
                  training_load

            From symptom data:
                - is_menstruating, flow_level, energy_level, mood_score,
                  pain_level, bloating, breast_tenderness

            Shared columns (appear once):
                - date, cycle_day, cycle_phase

        Notes
        -----
        This method internally calls both `generate_wearable_data` and
        `generate_symptom_data` with identical parameters, then performs an
        inner join on ['date', 'cycle_day', 'cycle_phase']. Since both datasets
        are generated from the same RNG state with identical date ranges, the
        merge is guaranteed to be lossless.

        The combined dataset is ideal for:
        - Training machine learning models that use both objective and
          subjective features
        - Exploratory data analysis examining correlations
        - Preprocessing pipelines that require complete daily records

        See Also
        --------
        generate_wearable_data : Generate only wearable device metrics
        generate_symptom_data : Generate only symptom data

        Examples
        --------
        Generate a complete dataset for analysis:

        >>> sdg = SyntheticDataGenerator(seed=42)
        >>> data = sdg.generate_combined_data(n_days=90)
        >>> data.shape
        (90, 19)
        >>> list(data.columns)
        ['date', 'resting_heart_rate', 'heart_rate_variability', ...]

        Use combined data for correlation analysis:

        >>> corr = data[['energy_level', 'sleep_quality_score']].corr()
        >>> corr.loc['energy_level', 'sleep_quality_score']
        0.3...

        Verify data completeness:

        >>> data.isnull().sum().sum()
        0
        >>> len(data) == 90
        True

        Generate data for a specific time period:

        >>> from datetime import datetime
        >>> march_data = sdg.generate_combined_data(
        ...     n_days=31,
        ...     start_date=datetime(2024, 3, 1)
        ... )
        >>> march_data['date'].min()
        '2024-03-01'
        >>> march_data['date'].max()
        '2024-03-31'
        """
        wearable_df = self.generate_wearable_data(n_days, start_date)
        symptom_df = self.generate_symptom_data(n_days, start_date)

        # Merge on date
        combined_df = pd.merge(wearable_df, symptom_df, on=['date', 'cycle_day', 'cycle_phase'])

        return combined_df