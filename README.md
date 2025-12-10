# AstridML

Machine learning pipeline for female athlete health optimization, helping athletes understand their menstrual cycle and optimize performance, recovery, and nutrition.

## Overview

Female athletes are a heavily underserved population in sports science research. Less than 6% of sports science research has been conducted exclusively on female athletes, leaving critical gaps in understanding how to optimize performance and avoid injury. Studies suggest correlations between certain injuries and menstrual cycle phases, as well as psychological changes throughout the cycle that impact athletic performance and mental health.

**AstridML** addresses this gap by providing a machine learning pipeline that:
- Analyzes wearable training data and menstrual cycle symptoms
- Predicts future symptoms and performance metrics
- Generates personalized recommendations for nutrition, recovery, and training

## Features

- **Synthetic Data Generation**: Create realistic training and menstrual cycle data for testing and development
- **Data Preprocessing**: Clean, validate, and engineer features from raw data
- **ML Prediction Model**: Neural network-based symptom and performance forecasting
- **Recommendation Engine**: Personalized advice based on cycle phase and predicted state
- **REST API**: FastAPI-based endpoints for data ingestion and inference
- **Comprehensive Testing**: Unit and integration tests with property-based testing

## Repo structure

```
  AstridML/
  ├── astridml/          # Main package
  │   ├── sdg/           # Synthetic data generator
  │   ├── dpm/           # Data preprocessing
  │   ├── models/        # ML models and recommender
  │   └── api/           # FastAPI server
  ├── tests/             # Comprehensive test suite
  │   ├── unit/          # Unit tests
  │   └── integration/   # Integration tests
  ├── examples/          # Example scripts
  ├── .github/workflows/ # CI/CD
  ├── example.py         # Quick demo script
  └── pyproject.toml     # Package configuration
```

## Installation

### Using uv (recommended)

```bash
# Clone the repository
git clone https://github.com/your-username/AstridML.git
cd AstridML

# Create virtual environment and install
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"
```

### Using pip

```bash
pip install -e ".[dev]"
```

## Quick Start

### Command Line Example

Run the basic pipeline example:

```bash
python example.py
```

For more detailed examples, see the `examples/` directory:

```bash
# Basic end-to-end pipeline
python examples/basic_pipeline.py

# Data generation examples
python examples/data_generation.py

# Recommendation engine scenarios
python examples/recommendations.py
```

See [examples/README.md](examples/README.md) for detailed documentation of each example.

### Python API

```python
from astridml import (
    SyntheticDataGenerator,
    DataPreprocessor,
    SymptomPredictor,
    RecommendationEngine
)

# 1. Generate synthetic data
sdg = SyntheticDataGenerator(seed=42)
data = sdg.generate_combined_data(n_days=90)

# 2. Preprocess data
preprocessor = DataPreprocessor()
target_cols = ['energy_level', 'mood_score', 'pain_level']
X, y, features = preprocessor.fit_transform(data, target_cols)

# 3. Train model
predictor = SymptomPredictor(input_dim=X.shape[1])
history = predictor.train(X, y, epochs=50, batch_size=16)

# 4. Make predictions
predictions = predictor.predict(X[-1:])

# 5. Generate recommendations
recommender = RecommendationEngine()
current_state = {
    'cycle_phase': 'follicular',
    'energy_level': 7.5,
    'pain_level': 2.0,
    'mood_score': 8.0,
    'sleep_quality_score': 80.0,
    'heart_rate_variability': 65.0,
    'heart_rate_variability_rolling_7d': 65.0
}

recommendations = recommender.generate_recommendations(current_state)
print(recommender.format_recommendations(recommendations))
```

### REST API

Start the API server:

```bash
uvicorn astridml.api.server:app --reload
```

Visit `http://localhost:8000/docs` for interactive API documentation.

Example API usage:

```python
import requests

# Ingest data
response = requests.post(
    "http://localhost:8000/data/ingest",
    json={
        "wearable_data": [...],
        "symptom_data": [...]
    }
)

# Get predictions and recommendations
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "wearable_data": [...],
        "symptom_data": [...]
    }
)

predictions = response.json()
print(predictions['recommendations'])
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=astridml --cov-report=html

# Run specific test file
pytest tests/unit/test_sdg.py

# Run property-based tests
pytest tests/unit/test_sdg.py -v
```

### Code Quality

```bash
# Format code
black astridml/ tests/

# Lint code
ruff check astridml/ tests/

# Type checking (if mypy is installed)
mypy astridml/
```

### Building Package

```bash
# Build distribution
uv pip install build
python -m build

# Install locally
pip install dist/astridml-0.1.0-py3-none-any.whl
```

## Architecture

### Components

1. **Synthetic Data Generator (SDG)**
   - Location: `astridml/sdg/`
   - Simulates wearable training data and menstrual cycle symptoms
   - Includes realistic correlations between cycle phases and performance metrics

2. **Data Preprocessing Module (DPM)**
   - Location: `astridml/dpm/`
   - Cleans and validates data
   - Engineers features (rolling averages, trends, cycle phase encoding)
   - Normalizes features for ML models

3. **ML Models**
   - Location: `astridml/models/`
   - `SymptomPredictor`: Neural network for symptom prediction
   - `RecommendationEngine`: Rule-based system for personalized advice

4. **API**
   - Location: `astridml/api/`
   - FastAPI-based REST endpoints
   - Handles data ingestion, training, and inference

### Data Flow

```
Raw Data → DPM (preprocessing) → ML Model (prediction) → Recommendations
```

## Testing Strategy

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test API endpoints and component interactions
- **Property-Based Tests**: Use Hypothesis for generative testing
- **Coverage**: Aim for >80% code coverage

## Technology Stack

- **Language**: Python 3.9+
- **ML/Data**: NumPy, pandas, SciPy, scikit-learn, TensorFlow
- **API**: FastAPI, Uvicorn, Pydantic
- **Testing**: pytest, pytest-cov, Hypothesis
- **CI/CD**: GitHub Actions
- **Package Management**: uv
