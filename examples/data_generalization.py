"""Example demonstrating synthetic data generation.

This example shows how to use the SyntheticDataGenerator to create
realistic wearable and menstrual cycle symptom data for testing and development.

Run this after installing AstridML:
    python examples/data_generation.py
"""

from astridml import SyntheticDataGenerator
from datetime import datetime


def main():
    print("=" * 70)
    print("Synthetic Data Generation Example")
    print("=" * 70)

    # Create generator with seed for reproducibility
    sdg = SyntheticDataGenerator(seed=42)

    # Example 1: Generate wearable data only
    print("\n[Example 1] Generating wearable device data...")
    wearable_data = sdg.generate_wearable_data(
        n_days=30, start_date=datetime(2024, 1, 1)
    )
    print(f"Generated {len(wearable_data)} days of wearable data")
    print("\nFirst 5 rows:")
    print(wearable_data.head())
    print("\nWearable data columns:", list(wearable_data.columns))

    # Example 2: Generate symptom data only
    print("\n" + "-" * 70)
    print("\n[Example 2] Generating symptom data...")
    symptom_data = sdg.generate_symptom_data(n_days=30, start_date=datetime(2024, 1, 1))
    print(f"Generated {len(symptom_data)} days of symptom data")
    print("\nFirst 5 rows:")
    print(symptom_data.head())
    print("\nSymptom data columns:", list(symptom_data.columns))

    # Example 3: Generate combined data
    print("\n" + "-" * 70)
    print("\n[Example 3] Generating combined data...")
    combined_data = sdg.generate_combined_data(
        n_days=90, start_date=datetime(2024, 1, 1)
    )
    print(f"Generated {len(combined_data)} days of combined data")
    print("\nFirst 3 rows:")
    print(combined_data.head(3))
    print(f"\nTotal columns: {len(combined_data.columns)}")
    print("All columns:", list(combined_data.columns))

    # Example 4: Analyze cycle phases
    print("\n" + "-" * 70)
    print("\n[Example 4] Cycle phase distribution...")
    phase_counts = combined_data["cycle_phase"].value_counts()
    print(phase_counts)

    # Example 5: Analyze correlations
    print("\n" + "-" * 70)
    print("\n[Example 5] Sample correlations between energy and cycle phase...")
    for phase in ["menstrual", "follicular", "ovulatory", "luteal"]:
        phase_data = combined_data[combined_data["cycle_phase"] == phase]
        avg_energy = phase_data["energy_level"].mean()
        avg_pain = phase_data["pain_level"].mean()
        print(f"  {phase:12s}: energy={avg_energy:.2f}/10, pain={avg_pain:.2f}/10")

    print("\n" + "=" * 70)
    print("Data generation examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
