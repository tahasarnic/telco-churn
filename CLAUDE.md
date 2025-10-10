# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a telco customer churn prediction ML project that uses DVC for data versioning and MLflow for experiment tracking. The project follows a typical ML pipeline structure with data preprocessing, feature engineering, and model training.

## Project Structure

- `data/data.csv` - Raw telco customer data (977KB, ~7043 records with 21 features)
- `data/cleaned_data.csv` - Preprocessed data with TotalCharges converted to numeric and Churn encoded to binary
- `eda.ipynb` - Exploratory data analysis notebook with detailed feature analysis
- `preprocessing.py` - Data cleaning script (converts TotalCharges to numeric, encodes Churn)
- `train.py` - Model training script (currently empty)
- `outputs.dvc` - DVC tracking for model outputs directory

## Development Environment

**Python Version**: 3.11+ (managed via `.python-version`)

**Package Management**: Uses `uv` for dependency management (see `uv.lock`). Dependencies are also listed in `requirements.txt` and `pyproject.toml`.

**Install dependencies**:
```bash
uv sync
# or
pip install -r requirements.txt
```

**Activate virtual environment**:
```bash
source .venv/bin/activate
```

## Key Dependencies

- `pandas>=2.3.3` - Data manipulation
- `scikit-learn>=1.7.2` - ML algorithms
- `feature-engine>=1.9.3` - Feature engineering transformations
- `hydra-core>=1.3.2` - Configuration management (not yet implemented)
- `pydantic>=2.11.9` - Data validation (not yet implemented)
- `dvc>=3.63.0` - Data and model versioning
- `mlflow>=3.4.0` - Experiment tracking
- `matplotlib>=3.10.6`, `seaborn>=0.13.2` - Visualization

## Data Pipeline

### Current State

The project is in early development. DVC has been initialized but the pipeline (`dvc.yaml`) has not been created yet.

**Git status shows**:
- DVC configuration and data tracking files were deleted in working tree
- `cleaned_data.csv` exists but is not tracked by DVC
- `preprocessing.py` has been modified

### Data Preprocessing Requirements (from EDA)

Based on analysis in `eda.ipynb`, the following preprocessing is needed:

1. **TotalCharges**: Convert from string to numeric (handles blank values as NaN), impute missing values with median
2. **Categorical encoding**: Label encoding for categorical variables (customerID should be dropped)
3. **Feature scaling**: RobustScaler for continuous features (tenure, MonthlyCharges)
4. **Target encoding**: Churn column encoded as binary (No=0, Yes=1)

**Feature types identified**:
- Continuous: `tenure`, `MonthlyCharges` (need scaling)
- Discrete: `SeniorCitizen`
- Categorical: 18 features including `gender`, `Partner`, `Contract`, `PaymentMethod`, etc.
- High cardinality: `customerID` (7043 unique - should be dropped as identifier)

## Common Commands

**Run data preprocessing**:
```bash
python preprocessing.py
```

**Launch Jupyter for EDA**:
```bash
jupyter notebook eda.ipynb
# or
jupyter lab
```

**Run model training**:
```bash
python train.py  # Not yet implemented
```

**DVC operations** (when pipeline is set up):
```bash
dvc repro              # Reproduce pipeline
dvc dag                # View pipeline DAG
dvc push               # Push data to remote storage
dvc pull               # Pull data from remote storage
```

**MLflow tracking** (when implemented):
```bash
mlflow ui              # Launch MLflow UI at http://localhost:5000
```

## Architecture Notes

### Planned ML Pipeline

The project appears to be setting up a structured ML pipeline with:
1. Configuration management via Hydra (not yet configured)
2. Data validation via Pydantic (not yet implemented)
3. Feature engineering using feature-engine library
4. DVC for reproducible data pipelines
5. MLflow for experiment tracking

### Feature Engineering Strategy

From the EDA notebook analysis, the recommended transformations are:
- `SimpleImputer` or `MeanMedianImputer` for TotalCharges
- `OrdinalEncoder` or `LabelEncoder` for categorical variables
- `RobustScaler` for continuous variables (chosen over StandardScaler/MinMaxScaler)
- Drop `customerID` before training

No outlier treatment or additional missing value imputation needed for other features.
