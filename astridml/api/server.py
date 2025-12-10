"""FastAPI server for data ingestion and ML inference."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import pandas as pd
from datetime import datetime

from astridml.dpm import DataPreprocessor
from astridml.models import SymptomPredictor, RecommendationEngine


app = FastAPI(
    title="AstridML API",
    description="Machine learning pipeline for female athlete health optimization",
    version="0.1.0",
)


class WearableData(BaseModel):
    """Schema for wearable device data."""

    date: str = Field(..., description="Date in YYYY-MM-DD format")
    resting_heart_rate: float = Field(..., ge=30, le=120)
    heart_rate_variability: float = Field(..., ge=0, le=200)
    sleep_hours: float = Field(..., ge=0, le=24)
    sleep_quality_score: float = Field(..., ge=0, le=100)
    steps: int = Field(..., ge=0)
    active_minutes: int = Field(..., ge=0)
    calories_burned: int = Field(..., ge=0)
    training_load: float = Field(..., ge=0)
    cycle_day: int = Field(..., ge=1, le=28)
    cycle_phase: str = Field(..., pattern="^(menstrual|follicular|ovulatory|luteal)$")


class SymptomData(BaseModel):
    """Schema for menstrual cycle symptom data."""

    date: str = Field(..., description="Date in YYYY-MM-DD format")
    cycle_day: int = Field(..., ge=1, le=28)
    cycle_phase: str = Field(..., pattern="^(menstrual|follicular|ovulatory|luteal)$")
    is_menstruating: bool
    flow_level: int = Field(..., ge=0, le=5)
    energy_level: float = Field(..., ge=1, le=10)
    mood_score: float = Field(..., ge=1, le=10)
    pain_level: float = Field(..., ge=0, le=10)
    bloating: float = Field(..., ge=0, le=10)
    breast_tenderness: float = Field(..., ge=0, le=10)


class CombinedDataInput(BaseModel):
    """Schema for combined wearable and symptom data."""

    wearable_data: List[WearableData]
    symptom_data: List[SymptomData]


class PredictionResponse(BaseModel):
    """Schema for prediction response."""

    predictions: Dict[str, float]
    recommendations: Dict[str, List[str]]
    timestamp: str


class HealthStatus(BaseModel):
    """Schema for health check response."""

    status: str
    version: str
    timestamp: str


# Global instances (in production, these would be loaded from saved models)
preprocessor = DataPreprocessor()
predictor: Optional[SymptomPredictor] = None
recommender = RecommendationEngine()


@app.get("/", response_model=HealthStatus)
async def root():
    """Root endpoint returning API status."""
    return {"status": "healthy", "version": "0.1.0", "timestamp": datetime.now().isoformat()}


@app.get("/health", response_model=HealthStatus)
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "0.1.0", "timestamp": datetime.now().isoformat()}


@app.post("/data/ingest")
async def ingest_data(data: CombinedDataInput) -> Dict:
    """
    Ingest wearable and symptom data.

    This endpoint receives data from wearables and manual symptom logs,
    processes it through the DPM, and stores it for training.
    """
    try:
        # Convert to DataFrames
        wearable_df = pd.DataFrame([item.model_dump() for item in data.wearable_data])
        symptom_df = pd.DataFrame([item.model_dump() for item in data.symptom_data])

        # Merge data
        combined_df = pd.merge(
            wearable_df, symptom_df, on=["date", "cycle_day", "cycle_phase"], how="inner"
        )

        if combined_df.empty:
            raise HTTPException(
                status_code=400, detail="No matching dates between wearable and symptom data"
            )

        return {
            "status": "success",
            "records_processed": len(combined_df),
            "date_range": {"start": combined_df["date"].min(), "end": combined_df["date"].max()},
            "timestamp": datetime.now().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictionResponse)
async def predict(data: CombinedDataInput):
    """
    Make predictions and generate recommendations.

    This endpoint processes input data, makes predictions about future symptoms,
    and generates personalized recommendations.
    """
    try:
        # Convert to DataFrames
        wearable_df = pd.DataFrame([item.model_dump() for item in data.wearable_data])
        symptom_df = pd.DataFrame([item.model_dump() for item in data.symptom_data])

        # Merge data
        combined_df = pd.merge(
            wearable_df, symptom_df, on=["date", "cycle_day", "cycle_phase"], how="inner"
        )

        if combined_df.empty:
            raise HTTPException(
                status_code=400, detail="No matching dates between wearable and symptom data"
            )

        # Get current state (most recent record)
        current_row = combined_df.iloc[-1]
        current_state = current_row.to_dict()

        # Make predictions if model is available
        predictions_dict = {}
        if predictor is not None and predictor.model is not None:
            # Use the global preprocessor that was fitted during training
            # The preprocessor now handles different cycle phases consistently
            if not preprocessor.is_fitted:
                raise HTTPException(
                    status_code=400,
                    detail="Model has not been trained yet. Please train the model first.",
                )

            X, _ = preprocessor.transform(combined_df)

            # Predict on most recent data
            pred = predictor.predict(X[-1:])

            predictions_dict = {
                "energy_level": (
                    float(pred[0][0]) if pred.shape[1] > 0 else current_state["energy_level"]
                ),
                "mood_score": (
                    float(pred[0][1]) if pred.shape[1] > 1 else current_state["mood_score"]
                ),
                "pain_level": (
                    float(pred[0][2]) if pred.shape[1] > 2 else current_state["pain_level"]
                ),
            }
        else:
            # Use current values if no model
            predictions_dict = {
                "energy_level": current_state["energy_level"],
                "mood_score": current_state["mood_score"],
                "pain_level": current_state["pain_level"],
            }

        # Generate recommendations
        recommendations = recommender.generate_recommendations(current_state, predictions_dict)

        return {
            "predictions": predictions_dict,
            "recommendations": recommendations,
            "timestamp": datetime.now().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train")
async def train_model(data: CombinedDataInput) -> Dict:
    """
    Train the ML model on provided data.

    This endpoint trains the symptom prediction model using historical data.
    In production, this would be a background task.
    """
    global predictor, preprocessor

    try:
        # Convert to DataFrames
        wearable_df = pd.DataFrame([item.model_dump() for item in data.wearable_data])
        symptom_df = pd.DataFrame([item.model_dump() for item in data.symptom_data])

        # Merge data
        combined_df = pd.merge(
            wearable_df, symptom_df, on=["date", "cycle_day", "cycle_phase"], how="inner"
        )

        if len(combined_df) < 30:
            raise HTTPException(
                status_code=400,
                detail="Insufficient data for training (minimum 30 records required)",
            )

        # Preprocess
        target_cols = ["energy_level", "mood_score", "pain_level"]
        X, y, feature_names = preprocessor.fit_transform(combined_df, target_cols)

        # Initialize and train model
        predictor = SymptomPredictor(input_dim=X.shape[1])
        history = predictor.train(X, y, epochs=50, batch_size=16, verbose=0)

        # Evaluate on training data (in production, use separate test set)
        metrics = predictor.evaluate(X, y)

        return {
            "status": "success",
            "training_records": len(combined_df),
            "features": len(feature_names),
            "final_loss": float(history["loss"][-1]),
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
