"""
Unit Tests for Data Preprocessing and Feature Engineering.
Validates cleaning, transformations, edge cases (tenure=0, missing TotalCharges),
customerID preservation in analytics vs removal in modeling,
and handling of unknown categorical levels.
"""

import pytest
import numpy as np
import pandas as pd

from src.features import engineer_features, FeatureEngineer
from src.data_preprocessing import clean_customer_data, build_preprocessor, prepare_features_and_target, build_full_pipeline


@pytest.fixture
def sample_raw_data():
    """Returns a representative sample DataFrame mimicking IBM Telco dataset."""
    return pd.DataFrame([
        {
            'customerID': '0001-TESTA',
            'gender': 'Female',
            'SeniorCitizen': 0,
            'Partner': 'Yes',
            'Dependents': 'No',
            'tenure': 1,
            'PhoneService': 'No',
            'MultipleLines': 'No phone service',
            'InternetService': 'DSL',
            'OnlineSecurity': 'No',
            'OnlineBackup': 'Yes',
            'DeviceProtection': 'No',
            'TechSupport': 'No',
            'StreamingTV': 'No',
            'StreamingMovies': 'No',
            'Contract': 'Month-to-month',
            'PaperlessBilling': 'Yes',
            'PaymentMethod': 'Electronic check',
            'MonthlyCharges': 29.85,
            'TotalCharges': '29.85',
            'Churn': 'No'
        },
        {
            'customerID': '0002-TESTB',
            'gender': 'Male',
            'SeniorCitizen': 1,
            'Partner': 'No',
            'Dependents': 'No',
            'tenure': 0,  # Zero tenure edge case
            'PhoneService': 'Yes',
            'MultipleLines': 'No',
            'InternetService': 'Fiber optic',
            'OnlineSecurity': 'No internet service',
            'OnlineBackup': 'No internet service',
            'DeviceProtection': 'No internet service',
            'TechSupport': 'No internet service',
            'StreamingTV': 'No internet service',
            'StreamingMovies': 'No internet service',
            'Contract': 'Two year',
            'PaperlessBilling': 'No',
            'PaymentMethod': 'Mailed check',
            'MonthlyCharges': 75.0,
            'TotalCharges': ' ',  # Blank string missing TotalCharges
            'Churn': 'No'
        },
        {
            'customerID': '0003-TESTC',
            'gender': 'Male',
            'SeniorCitizen': 0,
            'Partner': 'No',
            'Dependents': 'No',
            'tenure': 55,
            'PhoneService': 'Yes',
            'MultipleLines': 'Yes',
            'InternetService': 'Fiber optic',
            'OnlineSecurity': 'Yes',
            'OnlineBackup': 'Yes',
            'DeviceProtection': 'Yes',
            'TechSupport': 'Yes',
            'StreamingTV': 'Yes',
            'StreamingMovies': 'Yes',
            'Contract': 'One year',
            'PaperlessBilling': 'Yes',
            'PaymentMethod': 'Bank transfer (automatic)',
            'MonthlyCharges': 105.0,
            'TotalCharges': '5775.0',
            'Churn': 'Yes'
        }
    ])


def test_customer_id_preserved_in_analytics(sample_raw_data):
    """Test that customerID is preserved in cleaned dataset for customer tracking."""
    cleaned = clean_customer_data(sample_raw_data, is_training=False)
    assert 'customerID' in cleaned.columns
    assert len(cleaned) == 3


def test_customer_id_dropped_from_predictive_features(sample_raw_data):
    """Test that customerID column is dropped from model features."""
    X, y = prepare_features_and_target(sample_raw_data, is_training=True)
    assert 'customerID' not in X.columns
    assert 'Churn' not in X.columns
    assert len(X) == 3
    assert len(y) == 3


def test_identical_feature_profiles_with_different_ids_preserved():
    """Verify that two distinct customers with identical profiles are not erroneously deleted."""
    dupe_features = pd.DataFrame([
        {
            'customerID': 'CUST-A1', 'gender': 'Male', 'SeniorCitizen': 0, 'Partner': 'No', 'Dependents': 'No',
            'tenure': 1, 'PhoneService': 'Yes', 'MultipleLines': 'No', 'InternetService': 'No',
            'OnlineSecurity': 'No', 'OnlineBackup': 'No', 'DeviceProtection': 'No', 'TechSupport': 'No',
            'StreamingTV': 'No', 'StreamingMovies': 'No', 'Contract': 'Month-to-month',
            'PaperlessBilling': 'Yes', 'PaymentMethod': 'Mailed check', 'MonthlyCharges': 20.0,
            'TotalCharges': '20.0', 'Churn': 'Yes'
        },
        {
            'customerID': 'CUST-A2', 'gender': 'Male', 'SeniorCitizen': 0, 'Partner': 'No', 'Dependents': 'No',
            'tenure': 1, 'PhoneService': 'Yes', 'MultipleLines': 'No', 'InternetService': 'No',
            'OnlineSecurity': 'No', 'OnlineBackup': 'No', 'DeviceProtection': 'No', 'TechSupport': 'No',
            'StreamingTV': 'No', 'StreamingMovies': 'No', 'Contract': 'Month-to-month',
            'PaperlessBilling': 'Yes', 'PaymentMethod': 'Mailed check', 'MonthlyCharges': 20.0,
            'TotalCharges': '20.0', 'Churn': 'Yes'
        }
    ])
    cleaned = clean_customer_data(dupe_features, is_training=True)
    # Both legitimate customers must be preserved
    assert len(cleaned) == 2
    assert set(cleaned['customerID']) == {'CUST-A1', 'CUST-A2'}


def test_tenure_zero_handling(sample_raw_data):
    """Test that zero-tenure records have TotalCharges set to 0.0 and avg_monthly_spend = 0.0."""
    cleaned = clean_customer_data(sample_raw_data, is_training=False)
    assert cleaned.loc[cleaned['tenure'] == 0, 'TotalCharges'].iloc[0] == 0.0

    engineered = engineer_features(cleaned)
    assert engineered.loc[engineered['tenure'] == 0, 'avg_monthly_spend'].iloc[0] == 0.0
    assert engineered.loc[engineered['tenure'] == 0, 'tenure_group'].iloc[0] == '0-12 months'


def test_redundant_categorical_standardization(sample_raw_data):
    """Test that 'No internet service' and 'No phone service' are mapped to 'No'."""
    cleaned = clean_customer_data(sample_raw_data, is_training=False)
    assert cleaned.loc[1, 'OnlineSecurity'] == 'No'
    assert cleaned.loc[1, 'TechSupport'] == 'No'
    assert cleaned.loc[0, 'MultipleLines'] == 'No'


def test_target_encoding(sample_raw_data):
    """Test that 'Churn' (Yes/No) is encoded as integers 1 and 0."""
    cleaned = clean_customer_data(sample_raw_data, is_training=True)
    assert cleaned['Churn'].tolist() == [0, 0, 1]
    assert np.issubdtype(cleaned['Churn'].dtype, np.integer)


def test_feature_engineering_outputs(sample_raw_data):
    """Validate all required engineered features are produced correctly."""
    cleaned = clean_customer_data(sample_raw_data, is_training=False)
    engineered = engineer_features(cleaned)

    assert 'tenure_group' in engineered.columns
    assert engineered.loc[2, 'tenure_group'] == '49+ months'
    assert engineered.loc[0, 'tenure_group'] == '0-12 months'

    assert 'num_services' in engineered.columns
    assert engineered.loc[2, 'num_services'] == 8
    assert engineered.loc[0, 'num_services'] == 1

    assert 'avg_monthly_spend' in engineered.columns
    assert np.isclose(engineered.loc[2, 'avg_monthly_spend'], 105.0)

    assert 'is_monthly_contract' in engineered.columns
    assert engineered.loc[0, 'is_monthly_contract'] == 1
    assert engineered.loc[1, 'is_monthly_contract'] == 0
    assert engineered.loc[2, 'is_monthly_contract'] == 0


def test_preprocessor_handles_unknown_categories(sample_raw_data):
    """Validate that unseen categories do not trigger crashes during inference."""
    cleaned = clean_customer_data(sample_raw_data, is_training=False)
    engineered = engineer_features(cleaned)

    preprocessor = build_preprocessor()
    preprocessor.fit(engineered)

    unseen_df = engineered.copy()
    unseen_df.loc[0, 'PaymentMethod'] = 'Cryptocurrency'
    unseen_df.loc[0, 'Contract'] = 'Lifetime Subscription'

    transformed = preprocessor.transform(unseen_df)
    assert transformed.shape[0] == 3
    assert not np.isnan(transformed).any()
