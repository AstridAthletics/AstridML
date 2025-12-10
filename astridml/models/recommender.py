"""Recommendation engine for nutrition, recovery, and performance advice."""

from typing import Dict, List, Optional


class RecommendationEngine:
    """
    Provides personalized recommendations based on current state and predictions.

    Generates actionable advice for nutrition, recovery, and training based on
    menstrual cycle phase, predicted symptoms, and performance metrics.
    """

    # Recommendation templates based on cycle phase and symptoms
    NUTRITION_RECOMMENDATIONS = {
        "menstrual": {
            "low_energy": [
                "Increase iron-rich foods (lean meats, spinach, legumes) to combat potential iron loss",
                "Focus on complex carbohydrates for sustained energy",
                "Stay well-hydrated to reduce bloating",
            ],
            "high_pain": [
                "Include anti-inflammatory foods (omega-3 rich fish, berries, turmeric)",
                "Reduce caffeine and salt intake to minimize cramping",
                "Consider magnesium-rich foods (dark chocolate, nuts, avocado)",
            ],
            "normal": [
                "Maintain balanced nutrition with adequate protein and healthy fats",
                "Include calcium-rich foods for bone health",
            ],
        },
        "follicular": {
            "high_energy": [
                "This is a great time for higher calorie intake to support intense training",
                "Increase protein intake to support muscle building (aim for 1.6-2.2g/kg body weight)",
                "Focus on pre-workout carbs for optimal performance",
            ],
            "normal": [
                "Maintain balanced macronutrients to support increased training capacity",
                "Stay well-hydrated, especially around intense training sessions",
            ],
        },
        "ovulatory": {
            "high_energy": [
                "Continue high protein intake to maximize training adaptations",
                "Ensure adequate carbohydrate intake for peak performance potential",
                "Consider timing carbs around high-intensity workouts",
            ],
            "normal": [
                "Maintain consistent nutrition to support performance",
                "Focus on nutrient-dense whole foods",
            ],
        },
        "luteal": {
            "low_energy": [
                "Increase complex carbohydrates to combat fatigue",
                "Include more frequent, smaller meals to stabilize blood sugar",
                "Prioritize foods rich in B vitamins for energy metabolism",
            ],
            "high_pain": [
                "Focus on anti-inflammatory foods",
                "Reduce processed foods and excess sodium",
                "Include foods rich in vitamin B6 (poultry, fish, bananas)",
            ],
            "normal": [
                "Slightly increase overall calorie intake to match metabolic changes",
                "Include more warming, comfort foods if cravings increase",
            ],
        },
    }

    RECOVERY_RECOMMENDATIONS = {
        "high_pain": [
            "Prioritize rest and gentle movement (yoga, walking)",
            "Use heat therapy for cramping relief",
            "Ensure 8-9 hours of quality sleep",
            "Consider reducing high-impact activities",
        ],
        "low_energy": [
            "Increase sleep duration by 30-60 minutes if possible",
            "Include recovery activities like foam rolling and stretching",
            "Consider a rest day or active recovery session",
            "Practice stress-reduction techniques (meditation, breathing exercises)",
        ],
        "poor_sleep": [
            "Establish consistent sleep schedule",
            "Avoid screens 1 hour before bed",
            "Create a cool, dark sleeping environment",
            "Consider magnesium supplementation (consult healthcare provider)",
        ],
        "high_hrv": [
            "Your body is well-recovered - good time for intense training",
            "Maintain current recovery protocols",
            "Consider pushing training intensity this week",
        ],
        "low_hrv": [
            "Prioritize recovery over intensity today",
            "Reduce training load by 20-30%",
            "Focus on sleep quality and stress management",
            "Consider a deload week if HRV remains low",
        ],
    }

    PERFORMANCE_RECOMMENDATIONS = {
        "follicular": [
            "Excellent phase for high-intensity interval training",
            "Good time for heavy strength training and PRs",
            "Consider scheduling competitions or key workouts during this phase",
            "Your body is primed for building fitness",
        ],
        "ovulatory": [
            "Peak performance potential - ideal for competitions or max efforts",
            "Maintain high training intensity",
            "Good recovery capacity supports back-to-back hard sessions",
            "Confidence and coordination may be at their best",
        ],
        "luteal": [
            "Shift toward maintenance or moderate intensity",
            "Focus on technique and skill work",
            "Longer warm-ups may be beneficial",
            "Listen to your body - some days will feel better than others",
        ],
        "menstrual": [
            "Honor your body's needs - reduce intensity if needed",
            "Lower-impact activities may feel better (swimming, cycling)",
            "Shorter, less intense sessions are still beneficial",
            "This is a good time for active recovery and mobility work",
        ],
    }

    def __init__(self):
        """Initialize the recommendation engine."""
        pass

    def _assess_energy_level(self, energy: float) -> str:
        """Classify energy level."""
        if energy < 4:
            return "low_energy"
        elif energy > 7:
            return "high_energy"
        return "normal"

    def _assess_pain_level(self, pain: float) -> str:
        """Classify pain level."""
        if pain > 5:
            return "high_pain"
        return "normal"

    def _assess_sleep_quality(self, sleep_score: float) -> str:
        """Classify sleep quality."""
        if sleep_score < 60:
            return "poor_sleep"
        return "normal"

    def _assess_hrv(self, hrv: float, hrv_avg: float) -> str:
        """Classify HRV relative to baseline."""
        if hrv > hrv_avg * 1.1:
            return "high_hrv"
        elif hrv < hrv_avg * 0.9:
            return "low_hrv"
        return "normal"

    def generate_recommendations(
        self, current_data: Dict, predictions: Optional[Dict] = None
    ) -> Dict[str, List[str]]:
        """
        Generate personalized recommendations.

        Args:
            current_data: Dictionary containing current state:
                - cycle_phase: Current menstrual cycle phase
                - energy_level: Current energy (1-10)
                - pain_level: Current pain (0-10)
                - mood_score: Current mood (1-10)
                - sleep_quality_score: Recent sleep quality (0-100)
                - heart_rate_variability: Current HRV
                - heart_rate_variability_rolling_7d: 7-day average HRV
            predictions: Optional dictionary with predicted values

        Returns:
            Dictionary with recommendation categories and lists of recommendations
        """
        recommendations = {"nutrition": [], "recovery": [], "performance": []}

        cycle_phase = current_data.get("cycle_phase", "follicular")
        energy = current_data.get("energy_level", 5)
        pain = current_data.get("pain_level", 0)
        sleep_score = current_data.get("sleep_quality_score", 75)
        hrv = current_data.get("heart_rate_variability", 65)
        hrv_avg = current_data.get("heart_rate_variability_rolling_7d", 65)

        # Nutrition recommendations based on phase and symptoms
        energy_state = self._assess_energy_level(energy)
        pain_state = self._assess_pain_level(pain)

        phase_nutrition = self.NUTRITION_RECOMMENDATIONS.get(cycle_phase, {})

        if pain_state == "high_pain" and "high_pain" in phase_nutrition:
            recommendations["nutrition"].extend(phase_nutrition["high_pain"])
        elif energy_state == "low_energy" and "low_energy" in phase_nutrition:
            recommendations["nutrition"].extend(phase_nutrition["low_energy"])
        elif energy_state == "high_energy" and "high_energy" in phase_nutrition:
            recommendations["nutrition"].extend(phase_nutrition["high_energy"])
        else:
            recommendations["nutrition"].extend(phase_nutrition.get("normal", []))

        # Recovery recommendations
        if pain_state == "high_pain":
            recommendations["recovery"].extend(self.RECOVERY_RECOMMENDATIONS["high_pain"])

        if energy_state == "low_energy":
            recommendations["recovery"].extend(self.RECOVERY_RECOMMENDATIONS["low_energy"])

        sleep_state = self._assess_sleep_quality(sleep_score)
        if sleep_state == "poor_sleep":
            recommendations["recovery"].extend(self.RECOVERY_RECOMMENDATIONS["poor_sleep"])

        hrv_state = self._assess_hrv(hrv, hrv_avg)
        if hrv_state in self.RECOVERY_RECOMMENDATIONS:
            recommendations["recovery"].extend(self.RECOVERY_RECOMMENDATIONS[hrv_state])

        # Performance recommendations based on cycle phase
        if cycle_phase in self.PERFORMANCE_RECOMMENDATIONS:
            recommendations["performance"].extend(self.PERFORMANCE_RECOMMENDATIONS[cycle_phase])

        # Add prediction-based recommendations if available
        if predictions:
            pred_energy = predictions.get("energy_level", energy)
            if pred_energy < 4:
                recommendations["recovery"].insert(
                    0,
                    f"Tomorrow's predicted energy is low ({pred_energy:.1f}/10) - plan for lighter activity",
                )

        return recommendations

    def format_recommendations(self, recommendations: Dict[str, List[str]]) -> str:
        """
        Format recommendations as readable text.

        Args:
            recommendations: Dictionary of recommendations by category

        Returns:
            Formatted string
        """
        output = []

        for category, items in recommendations.items():
            if items:
                output.append(f"\n{category.upper()} RECOMMENDATIONS:")
                output.append("-" * 40)
                for i, item in enumerate(items, 1):
                    output.append(f"{i}. {item}")

        return "\n".join(output)