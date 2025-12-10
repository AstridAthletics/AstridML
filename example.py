"""Example usage of AstridML pipeline."""

from astridml import (
    SyntheticDataGenerator,
    DataPreprocessor,
    SymptomPredictor,
    RecommendationEngine,
)


def main():
    """Demonstrate the complete AstridML pipeline."""
    print("=" * 60)
    print("AstridML - Female Athlete Health Optimization Pipeline")
    print("=" * 60)

    # 1. Generate synthetic data
    print("\n1. Generating synthetic data...")
    sdg = SyntheticDataGenerator(seed=42)
    data = sdg.generate_combined_data(n_days=90)
    print(f"   Generated {len(data)} days of data")
    print(f"   Columns: {', '.join(data.columns[:10])}...")

    # 2. Preprocess data
    print("\n2. Preprocessing data...")
    preprocessor = DataPreprocessor()
    target_cols = ["energy_level", "mood_score", "pain_level"]

    X_train, y_train, feature_names = preprocessor.fit_transform(data, target_cols)
    print(f"   Features: {len(feature_names)}")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Sample features: {feature_names[:5]}...")

    # 3. Train ML model
    print("\n3. Training machine learning model...")
    predictor = SymptomPredictor(input_dim=X_train.shape[1])
    history = predictor.train(X_train, y_train, epochs=50, batch_size=16, verbose=0)
    print(f"   Initial loss: {history['loss'][0]:.4f}")
    print(f"   Final loss: {history['loss'][-1]:.4f}")

    # 4. Evaluate model
    print("\n4. Evaluating model...")
    metrics = predictor.evaluate(X_train, y_train)
    print(f"   MAE: {metrics['mae']:.4f}")
    print(f"   MSE: {metrics['mse']:.4f}")

    # 5. Make predictions on recent data
    print("\n5. Making predictions on recent data...")
    recent_X = X_train[-1:]
    predictions = predictor.predict(recent_X)
    print(f"   Predicted energy level: {predictions[0][0]:.2f}/10")
    print(f"   Predicted mood score: {predictions[0][1]:.2f}/10")
    print(f"   Predicted pain level: {predictions[0][2]:.2f}/10")

    # 6. Generate recommendations
    print("\n6. Generating personalized recommendations...")
    recommender = RecommendationEngine()

    # Get current state from most recent data
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

    print(f"\n   Current Cycle Phase: {current_state['cycle_phase'].upper()}")
    print(f"   Current Energy: {current_state['energy_level']:.1f}/10")
    print(f"   Current Mood: {current_state['mood_score']:.1f}/10")
    print(f"   Current Pain: {current_state['pain_level']:.1f}/10")

    formatted = recommender.format_recommendations(recommendations)
    print(formatted)

    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()