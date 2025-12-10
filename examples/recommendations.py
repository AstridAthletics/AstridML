"""Example demonstrating the recommendation engine.

This example shows how to generate personalized nutrition, recovery, and
performance recommendations based on cycle phase and current symptoms.

Run this after installing AstridML:
    python examples/recommendations.py
"""

from astridml import RecommendationEngine


def print_scenario(scenario_name, current_state, predictions=None):
    """Helper function to print recommendations for a scenario."""
    print("\n" + "=" * 70)
    print(f"Scenario: {scenario_name}")
    print("=" * 70)

    print(f"\nCurrent State:")
    print(f"  Cycle Phase: {current_state['cycle_phase']}")
    print(f"  Energy Level: {current_state['energy_level']:.1f}/10")
    print(f"  Mood Score: {current_state['mood_score']:.1f}/10")
    print(f"  Pain Level: {current_state['pain_level']:.1f}/10")
    print(f"  Sleep Quality: {current_state['sleep_quality_score']:.1f}/100")
    print(f"  HRV: {current_state['heart_rate_variability']:.1f} ms")

    recommender = RecommendationEngine()
    recommendations = recommender.generate_recommendations(current_state, predictions)
    print(recommender.format_recommendations(recommendations))


def main():
    print("=" * 70)
    print("AstridML Recommendation Engine Examples")
    print("=" * 70)

    # Scenario 1: Menstrual phase with high pain
    print_scenario(
        "Menstrual Phase - High Pain Day",
        {
            "cycle_phase": "menstrual",
            "energy_level": 3.5,
            "pain_level": 7.0,
            "mood_score": 5.0,
            "sleep_quality_score": 60.0,
            "heart_rate_variability": 55.0,
            "heart_rate_variability_rolling_7d": 65.0,
        },
    )

    # Scenario 2: Follicular phase with high energy
    print_scenario(
        "Follicular Phase - High Energy Day",
        {
            "cycle_phase": "follicular",
            "energy_level": 8.5,
            "pain_level": 1.0,
            "mood_score": 8.5,
            "sleep_quality_score": 85.0,
            "heart_rate_variability": 75.0,
            "heart_rate_variability_rolling_7d": 65.0,
        },
    )

    # Scenario 3: Ovulatory phase - peak performance
    print_scenario(
        "Ovulatory Phase - Peak Performance Window",
        {
            "cycle_phase": "ovulatory",
            "energy_level": 9.0,
            "pain_level": 0.5,
            "mood_score": 9.0,
            "sleep_quality_score": 90.0,
            "heart_rate_variability": 80.0,
            "heart_rate_variability_rolling_7d": 65.0,
        },
    )

    # Scenario 4: Luteal phase with poor recovery
    print_scenario(
        "Luteal Phase - Poor Recovery",
        {
            "cycle_phase": "luteal",
            "energy_level": 4.0,
            "pain_level": 3.5,
            "mood_score": 4.5,
            "sleep_quality_score": 55.0,
            "heart_rate_variability": 50.0,
            "heart_rate_variability_rolling_7d": 65.0,
        },
    )

    # Scenario 5: With predictions
    print_scenario(
        "Luteal Phase - With Tomorrow's Predictions",
        {
            "cycle_phase": "luteal",
            "energy_level": 6.0,
            "pain_level": 2.0,
            "mood_score": 6.5,
            "sleep_quality_score": 70.0,
            "heart_rate_variability": 60.0,
            "heart_rate_variability_rolling_7d": 65.0,
        },
        predictions={
            "energy_level": 3.5,  # Predicting low energy tomorrow
            "mood_score": 5.0,
            "pain_level": 4.5,
        },
    )

    print("\n" + "=" * 70)
    print("Recommendation examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
