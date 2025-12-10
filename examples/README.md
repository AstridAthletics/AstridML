# AstridML Examples

This directory contains example scripts demonstrating various features of the AstridML pipeline.

## Installation

Before running these examples, ensure you have installed AstridML:

```bash
# Clone the repository
git clone https://github.com/your-username/AstridML.git
cd AstridML

# Install with uv (recommended)
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .

# Or install with pip
pip install -e .
```

## Available Examples

### 1. Basic Pipeline (`basic_pipeline.py`)

Demonstrates the complete AstridML workflow from data generation to recommendations.

```bash
python examples/basic_pipeline.py
```

**What it does:**
- Generates 90 days of synthetic training data
- Preprocesses and engineers features
- Trains a neural network prediction model
- Makes predictions on recent data
- Generates personalized recommendations

**Expected output:** Training progress, predictions, and recommendations based on cycle phase

### 2. Data Generation (`data_generation.py`)

Shows how to use the SyntheticDataGenerator to create realistic test data.

```bash
python examples/data_generation.py
```

**What it does:**
- Generates wearable device data (heart rate, HRV, sleep, etc.)
- Generates menstrual cycle symptom data
- Creates combined datasets
- Analyzes cycle phase distributions and correlations

**Expected output:** Sample data tables and statistical summaries

### 3. Recommendations (`recommendations.py`)

Explores the recommendation engine with various cycle phases and symptom scenarios.

```bash
python examples/recommendations.py
```

**What it does:**
- Demonstrates recommendations for different cycle phases
- Shows how recommendations adapt to symptoms (pain, energy, recovery)
- Includes scenarios with predictions for future symptoms

**Expected output:** Personalized nutrition, recovery, and performance recommendations for each scenario

## Using the Examples

Each example is self-contained and can be run independently. They demonstrate:

- **Basic Pipeline**: End-to-end workflow for new users
- **Data Generation**: How to create test data for development
- **Recommendations**: How the recommendation engine adapts to different situations
