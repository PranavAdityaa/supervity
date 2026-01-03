
#  Telco Customer Churn Prediction  

## Executive Summary  
The proposed project is a holistic machine-learning solution that has been developed to predict customer churn in the telecommunications industry without jeopardising its complete transparency and explainability. By combining gradient-boosting models with SHAP (SHapley Additive exPlanations) analysis the system does not only isolate customers that are at the risk of imminent attrition, but also provides clear and direct explanations of the source of each prediction. The suggested solution balances the historically hostile goals of predictive accuracy and business intelligence, thus providing the retention teams with accurate reasons that a specific customer is going to churn and enabling them to apply evidence-based intervention measures.  


## Overview

This project builds a model to predict which telecom customers will churn (leave the company) and explains the reasons behind each prediction using SHAP.

## Dataset

The project uses the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) from Kaggle with 7,043 customer records.

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the dataset from Kaggle and place it in the `data/` folder

## How to Run

**Train the model:**
```bash
python train.py
```

**Generate predictions and explanations:**
```bash
python predict.py
```

## Outputs

- `churn_model.pkl` - Trained LightGBM model
- `churn_scores.csv` - Churn predictions for all customers
- `shap_waterfall_*.png` - Visual explanations for 3 example customers

## Technology

- Python
- LightGBM - for the prediction model
- SHAP - for explaining predictions
- Pandas, NumPy - for data processing

## Key Results

- Model Accuracy: ~82%
- Precision: ~77%
- ROC-AUC: ~0.88

The model identifies contract type, tenure, and service adoption as the main factors affecting churn.
