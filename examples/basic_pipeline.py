"""Basic pipeline example demonstrating the complete AstridML workflow.

This example shows how to:
1. Generate synthetic training data
2. Preprocess the data
3. Train a prediction model
4. Make predictions
5. Generate recommendations

Run this after installing AstridML:
    python examples/basic_pipeline.py
"""

from astridml import (
    SyntheticDataGenerator,
    DataPreprocessor,
    SymptomPredictor,
    RecommendationEngine,
)


def main():
    print("=" * 70)
    print("AstridML Basic Pipeline Example")
    print("=" * 70)

    # Step 1: Generate synthetic data
    print("\n[1/5] Generating synthetic data...")
    sdg = SyntheticDataGenerator(seed=42)
    data = sdg.generate_combined_data(n_days=90)
    print(f"      Generated {len(data)} days of data")
    print(f"      Columns: {list(data.columns)[:5]}...")

    # Step 2: Preprocess data
    print("\n[2/5] Preprocessing data...")
    preprocessor = DataPreprocessor()
    target_cols = ["energy_level", "mood_score", "pain_level"]
    X, y, feature_names = preprocessor.fit_transform(data, target_cols)
    print(f"      Features: {len(feature_names)}")
    print(f"      Training samples: {len(X)}")

    # Step 3: Train ML model
    print("\n[3/5] Training prediction model...")
    predictor = SymptomPredictor(input_dim=X.shape[1])
    history = predictor.train(X, y, epochs=50, batch_size=16, verbose=0)
    print(f"      Initial loss: {history['loss'][0]:.4f}")
    print(f"      Final loss: {history['loss'][-1]:.4f}")

    # Step 4: Make predictions
    print("\n[4/5] Making predictions...")
    recent_X = X[-1:]
    predictions = predictor.predict(recent_X)
    print(f"      Predicted energy level: {predictions[0][0]:.2f}/10")
    print(f"      Predicted mood score: {predictions[0][1]:.2f}/10")
    print(f"      Predicted pain level: {predictions[0][2]:.2f}/10")

    # Step 5: Generate recommendations
    print("\n[5/5] Generating recommendations...")
    recommender = RecommendationEngine()

    current_row = data.iloc[-1]
    current_state = {
        "cycle_phase": current_row["cycle_phase"],
        "energy_level": current_row["energy_level"],
        "pain_level": current_row["pain_level"],
        "mood_score": current_row["mood_score"],
        "sleep_quality_score": current_row["sleep_quality_score"],
        "heart_rate_variability": current_row["heart_rate_variability"],
        "heart_rate_variability_rolling_7d": current_row["heart_rate_variability"],
    }

    prediction_dict = {
        "energy_level": float(predictions[0][0]),
        "mood_score": float(predictions[0][1]),
        "pain_level": float(predictions[0][2]),
    }

    recommendations = recommender.generate_recommendations(current_state, prediction_dict)

    print(f"\n      Current State:")
    print(f"      - Cycle Phase: {current_state['cycle_phase']}")
    print(f"      - Energy: {current_state['energy_level']:.1f}/10")
    print(f"      - Mood: {current_state['mood_score']:.1f}/10")
    print(f"      - Pain: {current_state['pain_level']:.1f}/10")

    print(recommender.format_recommendations(recommendations))

    print("\n" + "=" * 70)
    print("Pipeline completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
