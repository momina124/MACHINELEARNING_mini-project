"""
Customer Churn Feature Engineering Module.
Implements custom feature generation for the IBM Telco Churn dataset.
Designed to be compatible with scikit-learn Pipeline and ColumnTransformer.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

SERVICE_COLUMNS = [
    'PhoneService',
    'MultipleLines',
    'OnlineSecurity',
    'OnlineBackup',
    'DeviceProtection',
    'TechSupport',
    'StreamingTV',
    'StreamingMovies'
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies domain feature engineering to customer records.
    
    Features engineered:
    1. tenure_group:
       - '0-12 months'
       - '13-24 months'
       - '25-48 months'
       - '49+ months'
    2. num_services:
       - Total count of additional subscribed services.
    3. avg_monthly_spend:
       - TotalCharges / tenure, zero when tenure is zero.
    4. is_monthly_contract:
       - 1 for Month-to-month, 0 otherwise.
    """
    data = df.copy()

    # 1. tenure_group
    tenure_val = pd.to_numeric(data.get('tenure', 0), errors='coerce').fillna(0)
    bins = [-1, 12, 24, 48, np.inf]
    labels = ['0-12 months', '13-24 months', '25-48 months', '49+ months']
    data['tenure_group'] = pd.cut(tenure_val, bins=bins, labels=labels).astype(str)

    # 2. num_services: count active services where value == 'Yes'
    service_count = pd.Series(0, index=data.index)
    for col in SERVICE_COLUMNS:
        if col in data.columns:
            service_count += (data[col].astype(str).str.strip().str.lower() == 'yes').astype(int)
    data['num_services'] = service_count

    # 3. avg_monthly_spend = TotalCharges / tenure (0 when tenure == 0)
    if 'TotalCharges' in data.columns:
        tot_charges = pd.to_numeric(data['TotalCharges'], errors='coerce').fillna(0.0)
    else:
        tot_charges = pd.Series(0.0, index=data.index)
    
    # Avoid zero division
    avg_spend = np.where(tenure_val > 0, tot_charges / tenure_val, 0.0)
    data['avg_monthly_spend'] = np.round(avg_spend, 2)

    # 4. is_monthly_contract
    if 'Contract' in data.columns:
        contract_str = data['Contract'].astype(str).str.strip().str.lower()
        data['is_monthly_contract'] = (contract_str == 'month-to-month').astype(int)
    else:
        data['is_monthly_contract'] = 0

    return data


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer for customer feature engineering.
    Can be seamlessly placed in a Pipeline before ColumnTransformer.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return engineer_features(X)
        elif isinstance(X, np.ndarray):
            raise TypeError("FeatureEngineer expects a pandas DataFrame with named columns.")
        return engineer_features(pd.DataFrame(X))
