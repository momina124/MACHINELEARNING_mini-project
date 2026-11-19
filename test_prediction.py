"""
Unit Tests for Customer Churn Inference and Prediction.
Validates single prediction, batch prediction, probability ranges,
risk classification, thresholding, and recommendation engine.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.data_preprocessing import build_full_pipeline, clean_customer_data
from src.predict import (
    categorize_risk_tier,
    generate_retention_recommendations,
    predict_single_customer,
    predict_batch_customers
)


@pytest.fixture
def mock_trained_pipeline():
    """Builds and fits a fast mock pipeline for testing inference logic."""
    np.random.seed(42)
    raw_df = pd.DataFrame([
        {
            'gender': 'Female', 'SeniorCitizen': '0', 'Partner': 'Yes', 'Dependents': 'No',
            'tenure': 2, 'PhoneService': 'Yes', 'MultipleLines': 'No', 'InternetService': 'Fiber optic',
            'OnlineSecurity': 'No', 'OnlineBackup': 'No', 'DeviceProtection': 'No', 'TechSupport': 'No',
            'StreamingTV': 'No', 'StreamingMovies': 'No', 'Contract': 'Month-to-month',
            'PaperlessBilling': 'Yes', 'PaymentMethod': 'Electronic check',
            'MonthlyCharges': 70.0, 'TotalCharges': 140.0, 'Churn': 1
        },
        {
            'gender': 'Male', 'SeniorCitizen': '0', 'Partner': 'Yes', 'Dependents': 'Yes',
            'tenure': 60, 'PhoneService': 'Yes', 'MultipleLines': 'Yes', 'InternetService': 'DSL',
            'OnlineSecurity': 'Yes', 'OnlineBackup': 'Yes', 'DeviceProtection': 'Yes', 'TechSupport': 'Yes',
            'StreamingTV': 'Yes', 'StreamingMovies': 'Yes', 'Contract': 'Two year',
            'PaperlessBilling': 'No', 'PaymentMethod': 'Bank transfer (automatic)',
            'MonthlyCharges': 80.0, 'TotalCharges': 4800.0, 'Churn': 0
        },
        {
            'gender': 'Female', 'SeniorCitizen': '1', 'Partner': 'No', 'Dependents': 'No',
            'tenure': 10, 'PhoneService': 'Yes', 'MultipleLines': 'Yes', 'InternetService': 'Fiber optic',
            'OnlineSecurity': 'No', 'OnlineBackup': 'No', 'DeviceProtection': 'Yes', 'TechSupport': 'No',
            'StreamingTV': 'Yes', 'StreamingMovies': 'Yes', 'Contract': 'Month-to-month',
            'PaperlessBilling': 'Yes', 'PaymentMethod': 'Electronic check',
            'MonthlyCharges': 95.0, 'TotalCharges': 950.0, 'Churn': 1
        },
        {
            'gender': 'Male', 'SeniorCitizen': '0', 'Partner': 'No', 'Dependents': 'No',
            'tenure': 45, 'PhoneService': 'Yes', 'MultipleLines': 'No', 'InternetService': 'DSL',
            'OnlineSecurity': 'Yes', 'OnlineBackup': 'No', 'DeviceProtection': 'Yes', 'TechSupport': 'Yes',
            'StreamingTV': 'No', 'StreamingMovies': 'No', 'Contract': 'One year',
            'PaperlessBilling': 'No', 'PaymentMethod': 'Credit card (automatic)',
            'MonthlyCharges': 55.0, 'TotalCharges': 2475.0, 'Churn': 0
        }
    ])

    X = raw_df.drop(columns=['Churn'])
    y = raw_df['Churn']

    clf = LogisticRegression(random_state=42)
    pipeline = build_full_pipeline(clf)
    pipeline.fit(X, y)
    return pipeline


def test_risk_tier_categorization():
    """Validate configurable low, medium, and high risk threshold tiers."""
    # Default thresholds: < 0.30 is Low, 0.30 - 0.60 is Medium, >= 0.60 is High
    assert categorize_risk_tier(0.15) == "Low"
    assert categorize_risk_tier(0.299) == "Low"
    assert categorize_risk_tier(0.30) == "Medium"
    assert categorize_risk_tier(0.45) == "Medium"
    assert categorize_risk_tier(0.599) == "Medium"
    assert categorize_risk_tier(0.60) == "High"
    assert categorize_risk_tier(0.95) == "High"

    # Custom thresholds
    assert categorize_risk_tier(0.25, low_thresh=0.20, high_thresh=0.50) == "Medium"
    assert categorize_risk_tier(0.55, low_thresh=0.20, high_thresh=0.50) == "High"


def test_predict_single_customer_probability_bounds(mock_trained_pipeline):
    """Validate probability is strictly within [0.0, 1.0] and output schema is correct."""
    customer = {
        'gender': 'Female',
        'SeniorCitizen': 0,
        'Partner': 'Yes',
        'Dependents': 'No',
        'tenure': 4,
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'InternetService': 'Fiber optic',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'No',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'No',
        'StreamingMovies': 'No',
        'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 75.5,
        'TotalCharges': 302.0
    }

    res = predict_single_customer(customer, mock_trained_pipeline, decision_threshold=0.5)

    assert 0.0 <= res['churn_probability'] <= 1.0
    assert 0.0 <= res['churn_percentage'] <= 100.0
    assert res['prediction'] in ['Likely to Churn', 'Likely to Stay']
    assert res['risk_level'] in ['Low', 'Medium', 'High']
    assert isinstance(res['recommendations'], list)
    assert len(res['recommendations']) > 0


def test_batch_prediction(mock_trained_pipeline):
    """Validate batch inference on multi-record DataFrames."""
    batch_df = pd.DataFrame([
        {
            'customerID': 'CUST-001',
            'gender': 'Male', 'SeniorCitizen': 0, 'Partner': 'No', 'Dependents': 'No',
            'tenure': 2, 'PhoneService': 'Yes', 'MultipleLines': 'No', 'InternetService': 'Fiber optic',
            'OnlineSecurity': 'No', 'OnlineBackup': 'No', 'DeviceProtection': 'No', 'TechSupport': 'No',
            'StreamingTV': 'No', 'StreamingMovies': 'No', 'Contract': 'Month-to-month',
            'PaperlessBilling': 'Yes', 'PaymentMethod': 'Electronic check',
            'MonthlyCharges': 75.0, 'TotalCharges': 150.0
        },
        {
            'customerID': 'CUST-002',
            'gender': 'Female', 'SeniorCitizen': 0, 'Partner': 'Yes', 'Dependents': 'Yes',
            'tenure': 65, 'PhoneService': 'Yes', 'MultipleLines': 'Yes', 'InternetService': 'DSL',
            'OnlineSecurity': 'Yes', 'OnlineBackup': 'Yes', 'DeviceProtection': 'Yes', 'TechSupport': 'Yes',
            'StreamingTV': 'Yes', 'StreamingMovies': 'Yes', 'Contract': 'Two year',
            'PaperlessBilling': 'No', 'PaymentMethod': 'Bank transfer (automatic)',
            'MonthlyCharges': 85.0, 'TotalCharges': 5525.0
        }
    ])

    results, summary = predict_batch_customers(batch_df, mock_trained_pipeline, decision_threshold=0.5)

    assert 'Churn_Probability' in results.columns
    assert 'Prediction' in results.columns
    assert 'Risk_Level' in results.columns
    assert len(results) == 2
    assert summary['total_customers'] == 2
    assert (summary['high_risk_count'] + summary['medium_risk_count'] + summary['low_risk_count']) == 2


def test_batch_prediction_missing_essential_columns(mock_trained_pipeline):
    """Ensure meaningful error is raised when required columns are absent."""
    invalid_df = pd.DataFrame([{'gender': 'Female', 'Partner': 'Yes'}])
    with pytest.raises(ValueError, match="missing critical columns"):
        predict_batch_customers(invalid_df, mock_trained_pipeline)


def test_retention_recommendations_logic():
    """Verify targeted actions are triggered by high-risk customer attributes."""
    high_risk_cust = {
        'Contract': 'Month-to-month',
        'tenure': 3,
        'InternetService': 'Fiber optic',
        'OnlineSecurity': 'No',
        'TechSupport': 'No',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 95.0
    }
    recs = generate_retention_recommendations(high_risk_cust, prob=0.75, risk_level='High')
    strategies = [r['strategy'] for r in recs]

    assert 'Contract Migration Incentive' in strategies
    assert 'Early-Lifecycle Concierge Onboarding' in strategies
    assert 'Premium Care & Security Bundle' in strategies
    assert 'Auto-Pay Enrollment Campaign' in strategies
