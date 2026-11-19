"""
Inference and Prediction Pipeline Module.
Provides functions for single-customer inference, batch prediction,
risk tier categorization, and retention recommendations.
"""

import os
import json
import logging
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import joblib

from src.features import FeatureEngineer
from src.data_preprocessing import clean_customer_data

logger = logging.getLogger(__name__)

# Expected input columns
EXPECTED_COLUMNS = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
    'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
    'MonthlyCharges', 'TotalCharges'
]


def load_pipeline_and_metadata(models_dir: str = "models") -> Tuple[Any, float, Dict[str, Any]]:
    """
    Loads the trained pipeline, decision threshold, and metadata.
    """
    pipeline_path = os.path.join(models_dir, "churn_pipeline.joblib")
    threshold_path = os.path.join(models_dir, "threshold.joblib")
    metadata_path = os.path.join(models_dir, "model_metadata.json")

    if not os.path.exists(pipeline_path):
        raise FileNotFoundError(f"Model pipeline not found at {pipeline_path}. Please train model first.")

    pipeline = joblib.load(pipeline_path)

    threshold = 0.5
    if os.path.exists(threshold_path):
        threshold = float(joblib.load(threshold_path))

    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

    return pipeline, threshold, metadata


def categorize_risk_tier(
    prob: float,
    low_thresh: float = 0.30,
    high_thresh: float = 0.60
) -> str:
    """
    Categorizes churn probability into risk levels:
    - Low Risk: prob < low_thresh
    - Medium Risk: low_thresh <= prob < high_thresh
    - High Risk: prob >= high_thresh
    """
    if prob < low_thresh:
        return "Low"
    elif prob < high_thresh:
        return "Medium"
    else:
        return "High"


def generate_retention_recommendations(
    customer: Dict[str, Any],
    prob: float,
    risk_level: str
) -> List[Dict[str, str]]:
    """
    Generates intelligent, actionable customer retention recommendations
    tailored to the customer's contract, tenure, billing, and services.
    """
    recommendations = []

    contract = str(customer.get('Contract', '')).strip().lower()
    tenure = float(customer.get('tenure', 0))
    monthly = float(customer.get('MonthlyCharges', 0.0))
    internet = str(customer.get('InternetService', '')).strip().lower()
    security = str(customer.get('OnlineSecurity', '')).strip().lower()
    support = str(customer.get('TechSupport', '')).strip().lower()
    payment = str(customer.get('PaymentMethod', '')).strip().lower()

    # Rule 1: Contract type
    if 'month' in contract:
        recommendations.append({
            'strategy': 'Contract Migration Incentive',
            'action': 'Offer 15% discount or 2-month bill credit on migrating to a 1-year contract guarantee.',
            'priority': 'High' if risk_level in ['High', 'Medium'] else 'Medium'
        })

    # Rule 2: Tenure vulnerability
    if tenure <= 12:
        recommendations.append({
            'strategy': 'Early-Lifecycle Concierge Onboarding',
            'action': 'Schedule dedicated account check-in at 30/60 days to resolve initial service friction.',
            'priority': 'High'
        })

    # Rule 3: Fiber optic service without support
    if 'fiber' in internet and (security == 'no' or support == 'no'):
        recommendations.append({
            'strategy': 'Premium Care & Security Bundle',
            'action': 'Provide complimentary 6-month Tech Support and Online Security package.',
            'priority': 'High' if risk_level == 'High' else 'Medium'
        })

    # Rule 4: Payment friction
    if 'electronic check' in payment:
        recommendations.append({
            'strategy': 'Auto-Pay Enrollment Campaign',
            'action': 'Offer a one-time $10 account credit to set up automated credit card or bank transfer.',
            'priority': 'Medium'
        })

    # Rule 5: High monthly spend
    if monthly > 80.0:
        recommendations.append({
            'strategy': 'Value Optimization & Tariff Review',
            'action': 'Conduct account audit to eliminate redundant charges and offer bundle packaging.',
            'priority': 'Medium'
        })

    # Rule 6: Low risk customers
    if risk_level == 'Low' and not recommendations:
        recommendations.append({
            'strategy': 'Loyalty Rewards & Appreciation',
            'action': 'Enroll in VIP loyalty tier with periodic courtesy speed-boosts or anniversary rewards.',
            'priority': 'Low'
        })

    return recommendations


def predict_single_customer(
    customer_dict: Dict[str, Any],
    pipeline: Any,
    decision_threshold: float = 0.5,
    low_risk_thresh: float = 0.30,
    high_risk_thresh: float = 0.60
) -> Dict[str, Any]:
    """
    Executes end-to-end prediction for an individual customer.
    Returns probability, prediction label, risk tier, and recommendations.
    """
    df_raw = pd.DataFrame([customer_dict])
    df_clean = clean_customer_data(df_raw, is_training=False)

    # Predict probability
    prob = float(pipeline.predict_proba(df_clean)[0, 1])
    is_churn = bool(prob >= decision_threshold)
    prediction_label = "Likely to Churn" if is_churn else "Likely to Stay"
    risk_level = categorize_risk_tier(prob, low_risk_thresh, high_risk_thresh)

    recommendations = generate_retention_recommendations(customer_dict, prob, risk_level)

    return {
        'churn_probability': prob,
        'churn_percentage': round(prob * 100, 2),
        'prediction': prediction_label,
        'is_churn': is_churn,
        'risk_level': risk_level,
        'decision_threshold': decision_threshold,
        'recommendations': recommendations
    }


def predict_batch_customers(
    df: pd.DataFrame,
    pipeline: Any,
    decision_threshold: float = 0.5,
    low_risk_thresh: float = 0.30,
    high_risk_thresh: float = 0.60
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Executes batch predictions for multiple customer records in a DataFrame.
    Validates column presence, handles missing values, and returns summary stats.
    """
    # Verify at least minimal essential columns
    essential_cols = ['tenure', 'MonthlyCharges', 'Contract']
    missing_essential = [c for c in essential_cols if c not in df.columns]
    if missing_essential:
        raise ValueError(f"Uploaded CSV is missing critical columns: {', '.join(missing_essential)}")

    # Clean data (preserves original for export)
    df_clean = clean_customer_data(df, is_training=False)

    # Get predictions
    probabilities = pipeline.predict_proba(df_clean)[:, 1]
    predictions = (probabilities >= decision_threshold).astype(int)

    risk_levels = [
        categorize_risk_tier(p, low_risk_thresh, high_risk_thresh)
        for p in probabilities
    ]

    result_df = df.copy()
    result_df['Churn_Probability'] = np.round(probabilities, 4)
    result_df['Prediction'] = np.where(predictions == 1, 'Likely to Churn', 'Likely to Stay')
    result_df['Risk_Level'] = risk_levels

    # Summary statistics
    total = len(result_df)
    high_count = int((result_df['Risk_Level'] == 'High').sum())
    med_count = int((result_df['Risk_Level'] == 'Medium').sum())
    low_count = int((result_df['Risk_Level'] == 'Low').sum())
    predicted_churners = int((predictions == 1).sum())

    summary = {
        'total_customers': total,
        'predicted_churners': predicted_churners,
        'predicted_churn_rate': round((predicted_churners / total) * 100, 2) if total > 0 else 0.0,
        'high_risk_count': high_count,
        'medium_risk_count': med_count,
        'low_risk_count': low_count,
        'high_risk_pct': round((high_count / total) * 100, 2) if total > 0 else 0.0,
        'avg_churn_probability': round(float(np.mean(probabilities)) * 100, 2) if total > 0 else 0.0
    }

    return result_df, summary
