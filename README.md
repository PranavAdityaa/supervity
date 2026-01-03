# Telco Customer Churn Prediction

A machine learning solution to predict which telecom customers will churn and explain the reasons behind each prediction using SHAP interpretability.

## Problem Statement

Customer churn is a critical business challenge in the telecommunications industry. This project aims to:
- Predict which customers are likely to leave
- Provide explainable reasons for each prediction
- Enable targeted retention strategies based on customer-specific insights

## Dataset

The project uses the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) from Kaggle, containing 7,043 customer records with features including:
- Demographics (age, gender, senior citizen status)
- Services (phone, internet, streaming, security, technical support)
- Account information (tenure, contract type, billing method)
- Charges (monthly and total charges)

## Tech Stack

- **Language**: Python 3.8+
- **Model**: LightGBM (gradient boosting classifier)
- **Explainability**: SHAP (SHapley Additive exPlanations)
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib
- **Optional API**: FastAPI for production deployment

## Getting Started

### Installation

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd telco-churn-prediction
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place `WA_Fn-UseC_-Telco-Customer-Churn.csv` in a `data/` folder

### Running the Project

**Train the model:**
```bash
python train.py
```

This will:
- Load and preprocess the data
- Train a LightGBM classifier
- Evaluate performance on test set
- Save the trained model

**Generate predictions with explanations:**
```bash
python predict.py
```

This will:
- Generate churn predictions for all customers
- Create SHAP waterfall visualizations explaining 3 example predictions
- Export results to CSV format

## What is SHAP?

SHAP (SHapley Additive exPlanations) provides interpretable explanations for machine learning predictions. For each customer, it shows which features push the prediction toward churn (risk factors) and which features reduce the risk (protective factors).

## Project Goals

- Build an accurate churn prediction model
- Generate SHAP waterfall explanations for customer cohorts
- Create a CSV file with churn scores for all customers
- Optional: Deploy as a REST API for real-time predictions

## Status

Initial setup and core infrastructure. Development in progress.

## Contact

For questions or contributions, please reach out to the team.
