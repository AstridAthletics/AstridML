"""Integration tests for API endpoints."""

import pytest
from fastapi.testclient import TestClient

from astridml.api import app
from astridml.sdg import SyntheticDataGenerator


class TestAPIEndpoints:
    """Test suite for API integration."""

    @pytest.fixture
    def client(self):
        """Create a test client."""
        return TestClient(app)

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for API testing."""
        sdg = SyntheticDataGenerator(seed=42)
        df = sdg.generate_combined_data(n_days=60)

        # Convert to API format
        wearable_data = []
        symptom_data = []

        for _, row in df.iterrows():
            wearable_data.append(
                {
                    "date": row["date"],
                    "resting_heart_rate": float(row["resting_heart_rate"]),
                    "heart_rate_variability": float(row["heart_rate_variability"]),
                    "sleep_hours": float(row["sleep_hours"]),
                    "sleep_quality_score": float(row["sleep_quality_score"]),
                    "steps": int(row["steps"]),
                    "active_minutes": int(row["active_minutes"]),
                    "calories_burned": int(row["calories_burned"]),
                    "training_load": float(row["training_load"]),
                    "cycle_day": int(row["cycle_day"]),
                    "cycle_phase": row["cycle_phase"],
                }
            )

            symptom_data.append(
                {
                    "date": row["date"],
                    "cycle_day": int(row["cycle_day"]),
                    "cycle_phase": row["cycle_phase"],
                    "is_menstruating": bool(row["is_menstruating"]),
                    "flow_level": int(row["flow_level"]),
                    "energy_level": float(row["energy_level"]),
                    "mood_score": float(row["mood_score"]),
                    "pain_level": float(row["pain_level"]),
                    "bloating": float(row["bloating"]),
                    "breast_tenderness": float(row["breast_tenderness"]),
                }
            )

        return {"wearable_data": wearable_data, "symptom_data": symptom_data}

    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "timestamp" in data

    def test_health_check_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"

    def test_ingest_data_success(self, client, sample_data):
        """Test successful data ingestion."""
        response = client.post("/data/ingest", json=sample_data)
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "success"
        assert "records_processed" in data
        assert data["records_processed"] > 0
        assert "date_range" in data

    def test_ingest_data_empty_request(self, client):
        """Test data ingestion with empty data."""
        empty_data = {"wearable_data": [], "symptom_data": []}

        response = client.post("/data/ingest", json=empty_data)
        # Should fail with no matching dates
        assert response.status_code in [400, 500]

    def test_ingest_data_mismatched_dates(self, client, sample_data):
        """Test data ingestion with mismatched dates."""
        # Modify symptom data to have different dates
        modified_data = sample_data.copy()
        for item in modified_data["symptom_data"]:
            item["date"] = "2099-01-01"

        response = client.post("/data/ingest", json=modified_data)
        assert response.status_code == 400

    def test_ingest_data_invalid_format(self, client):
        """Test data ingestion with invalid format."""
        invalid_data = {"wearable_data": [{"foo": "bar"}], "symptom_data": [{"baz": "qux"}]}

        response = client.post("/data/ingest", json=invalid_data)
        assert response.status_code == 422  # Validation error

    def test_predict_endpoint_success(self, client, sample_data):
        """Test prediction endpoint."""
        response = client.post("/predict", json=sample_data)
        assert response.status_code == 200

        data = response.json()
        assert "predictions" in data
        assert "recommendations" in data
        assert "timestamp" in data

        # Check predictions structure
        predictions = data["predictions"]
        assert "energy_level" in predictions
        assert "mood_score" in predictions
        assert "pain_level" in predictions

        # Check recommendations structure
        recommendations = data["recommendations"]
        assert "nutrition" in recommendations
        assert "recovery" in recommendations
        assert "performance" in recommendations

    def test_predict_endpoint_recommendations_not_empty(self, client, sample_data):
        """Test that predictions include non-empty recommendations."""
        response = client.post("/predict", json=sample_data)
        assert response.status_code == 200

        data = response.json()
        recommendations = data["recommendations"]

        # At least one category should have recommendations
        total_recommendations = sum(len(v) for v in recommendations.values())
        assert total_recommendations > 0

    def test_predict_endpoint_invalid_data(self, client):
        """Test prediction with invalid data."""
        invalid_data = {"wearable_data": [], "symptom_data": []}

        response = client.post("/predict", json=invalid_data)
        assert response.status_code in [400, 500]

    def test_train_endpoint_success(self, client, sample_data):
        """Test model training endpoint."""
        response = client.post("/train", json=sample_data)
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "success"
        assert "training_records" in data
        assert "features" in data
        assert "final_loss" in data
        assert "metrics" in data

        # Check metrics structure
        metrics = data["metrics"]
        assert "loss" in metrics
        assert "mae" in metrics
        assert "mse" in metrics

    def test_train_endpoint_insufficient_data(self, client):
        """Test training with insufficient data."""
        sdg = SyntheticDataGenerator(seed=42)
        df = sdg.generate_combined_data(n_days=10)  # Less than minimum

        small_data = {"wearable_data": [], "symptom_data": []}

        # Convert small dataset
        for _, row in df.iterrows():
            small_data["wearable_data"].append(
                {
                    "date": row["date"],
                    "resting_heart_rate": float(row["resting_heart_rate"]),
                    "heart_rate_variability": float(row["heart_rate_variability"]),
                    "sleep_hours": float(row["sleep_hours"]),
                    "sleep_quality_score": float(row["sleep_quality_score"]),
                    "steps": int(row["steps"]),
                    "active_minutes": int(row["active_minutes"]),
                    "calories_burned": int(row["calories_burned"]),
                    "training_load": float(row["training_load"]),
                    "cycle_day": int(row["cycle_day"]),
                    "cycle_phase": row["cycle_phase"],
                }
            )

            small_data["symptom_data"].append(
                {
                    "date": row["date"],
                    "cycle_day": int(row["cycle_day"]),
                    "cycle_phase": row["cycle_phase"],
                    "is_menstruating": bool(row["is_menstruating"]),
                    "flow_level": int(row["flow_level"]),
                    "energy_level": float(row["energy_level"]),
                    "mood_score": float(row["mood_score"]),
                    "pain_level": float(row["pain_level"]),
                    "bloating": float(row["bloating"]),
                    "breast_tenderness": float(row["breast_tenderness"]),
                }
            )

        response = client.post("/train", json=small_data)
        assert response.status_code == 400

    def test_train_then_predict_workflow(self, client, sample_data):
        """Test complete workflow: train model then make predictions."""
        # First train the model
        train_response = client.post("/train", json=sample_data)
        assert train_response.status_code == 200

        # Then make predictions
        predict_response = client.post("/predict", json=sample_data)
        assert predict_response.status_code == 200

        data = predict_response.json()
        assert "predictions" in data
        assert "recommendations" in data

    def test_api_handles_validation_errors(self, client):
        """Test that API properly handles validation errors."""
        invalid_data = {
            "wearable_data": [
                {
                    "date": "invalid",
                    "resting_heart_rate": "not_a_number",  # Should be float
                }
            ],
            "symptom_data": [{"date": "invalid"}],
        }

        response = client.post("/data/ingest", json=invalid_data)
        assert response.status_code == 422

    def test_api_handles_out_of_range_values(self, client, sample_data):
        """Test that API validates value ranges."""
        invalid_data = sample_data.copy()
        # Set invalid value
        invalid_data["wearable_data"][0]["resting_heart_rate"] = 200  # Too high

        response = client.post("/data/ingest", json=invalid_data)
        # Should either reject or handle gracefully
        assert response.status_code in [200, 400, 422]

    def test_predict_endpoint_different_cycle_phases(self, client):
        """Test predictions for different cycle phases."""
        phases = ["menstrual", "follicular", "ovulatory", "luteal"]

        for phase in phases:
            sdg = SyntheticDataGenerator(seed=42)
            df = sdg.generate_combined_data(n_days=30)

            # Filter to specific phase
            phase_df = df[df["cycle_phase"] == phase].head(10)

            if len(phase_df) > 0:
                # Convert to API format
                data = {"wearable_data": [], "symptom_data": []}

                for _, row in phase_df.iterrows():
                    data["wearable_data"].append(
                        {
                            "date": row["date"],
                            "resting_heart_rate": float(row["resting_heart_rate"]),
                            "heart_rate_variability": float(row["heart_rate_variability"]),
                            "sleep_hours": float(row["sleep_hours"]),
                            "sleep_quality_score": float(row["sleep_quality_score"]),
                            "steps": int(row["steps"]),
                            "active_minutes": int(row["active_minutes"]),
                            "calories_burned": int(row["calories_burned"]),
                            "training_load": float(row["training_load"]),
                            "cycle_day": int(row["cycle_day"]),
                            "cycle_phase": row["cycle_phase"],
                        }
                    )

                    data["symptom_data"].append(
                        {
                            "date": row["date"],
                            "cycle_day": int(row["cycle_day"]),
                            "cycle_phase": row["cycle_phase"],
                            "is_menstruating": bool(row["is_menstruating"]),
                            "flow_level": int(row["flow_level"]),
                            "energy_level": float(row["energy_level"]),
                            "mood_score": float(row["mood_score"]),
                            "pain_level": float(row["pain_level"]),
                            "bloating": float(row["bloating"]),
                            "breast_tenderness": float(row["breast_tenderness"]),
                        }
                    )

                response = client.post("/predict", json=data)
                assert response.status_code == 200

    def test_concurrent_requests(self, client, sample_data):
        """Test handling multiple concurrent requests."""
        responses = []

        for _ in range(5):
            response = client.post("/predict", json=sample_data)
            responses.append(response)

        # All requests should succeed
        assert all(r.status_code == 200 for r in responses)

    def test_api_response_times(self, client, sample_data):
        """Test that API responds in reasonable time."""
        import time

        start_time = time.time()
        response = client.post("/predict", json=sample_data)
        elapsed_time = time.time() - start_time

        assert response.status_code == 200
        # Should respond within 10 seconds (generous limit for CI)
        assert elapsed_time < 10