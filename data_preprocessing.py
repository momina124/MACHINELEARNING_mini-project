"""
Data Preprocessing Module for Customer Churn Prediction.
Handles data loading, cleaning, validation, redundant categorical standardization,
and scikit-learn ColumnTransformer creation.
Preserves customerID in customer analytics datasets while removing it from predictive features.
"""

import os
import logging
from typing import Tuple, Optional, List
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.features import engineer_features, FeatureEngineer, SERVICE_COLUMNS

logger = logging.getLogger(__name__)

# Standard column lists
NUMERIC_FEATURES = [
    'tenure',
    'MonthlyCharges',
    'TotalCharges',
    'num_services',
    'avg_monthly_spend'
]

CATEGORICAL_FEATURES = [
    'gender',
    'SeniorCitizen',
    'Partner',
    'Dependents',
    'PhoneService',
    'MultipleLines',
    'InternetService',
    'OnlineSecurity',
    'OnlineBackup',
    'DeviceProtection',
    'TechSupport',
    'StreamingTV',
    'StreamingMovies',
    'Contract',
    'PaperlessBilling',
    'PaymentMethod',
    'tenure_group',
    'is_monthly_contract'
]

TARGET_COL = 'Churn'


def load_raw_data(file_path: str) -> pd.DataFrame:
    """Loads raw CSV data from the specified path."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")
    df = pd.read_csv(file_path)
    logger.info(f"Loaded dataset from {file_path} with shape {df.shape}")
    return df


def clean_customer_data(df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
    """
    Performs data cleaning while preserving distinct customer records:
    1. Removes genuinely duplicated complete records (including customerID).
    2. Preserves customerID for customer analytics tracking.
    3. Converts TotalCharges to numeric, handles whitespace/null values.
    4. Fills missing TotalCharges with 0.0 when tenure == 0.
    5. Standardizes redundant categorical values ('No internet service'/'No phone service' -> 'No').
    6. Standardizes SeniorCitizen to string category.
    7. Encodes target variable 'Churn' (Yes=1, No=0) when is_training=True.
    """
    data = df.copy()

    # 1. Genuine duplicate detection on complete record before touching customerID
    if is_training:
        initial_len = len(data)
        data = data.drop_duplicates()
        if len(data) < initial_len:
            logger.info(f"Dropped {initial_len - len(data)} genuinely duplicate records.")

    # 2. TotalCharges numeric conversion & zero-tenure handling
    if 'TotalCharges' in data.columns:
        data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
        if 'tenure' in data.columns:
            tenure_mask = (pd.to_numeric(data['tenure'], errors='coerce').fillna(0) == 0)
            data.loc[tenure_mask & data['TotalCharges'].isna(), 'TotalCharges'] = 0.0
        data['TotalCharges'] = data['TotalCharges'].fillna(0.0)

    # 3. Standardize SeniorCitizen
    if 'SeniorCitizen' in data.columns:
        data['SeniorCitizen'] = data['SeniorCitizen'].astype(str)

    # 4. Standardize redundant categorical values
    redundant_service_cols = [
        'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    for col in redundant_service_cols:
        if col in data.columns:
            data[col] = data[col].replace({
                'No internet service': 'No',
                'No phone service': 'No'
            })

    # 5. Target encoding if present
    if TARGET_COL in data.columns:
        if not pd.api.types.is_numeric_dtype(data[TARGET_COL]):
            data[TARGET_COL] = data[TARGET_COL].astype(str).str.strip().map({
                'Yes': 1, 'No': 0, 'yes': 1, 'no': 0, '1': 1, '0': 0, 'True': 1, 'False': 0
            })
        if is_training:
            data = data.dropna(subset=[TARGET_COL])
            data[TARGET_COL] = data[TARGET_COL].astype(int)

    return data


def build_preprocessor() -> ColumnTransformer:
    """
    Builds a Scikit-Learn ColumnTransformer for numerical scaling and categorical one-hot encoding.
    Ensures handle_unknown='ignore' and dense output.
    """
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERIC_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ],
        remainder='drop'
    )
    return preprocessor


def prepare_features_and_target(
    df: pd.DataFrame,
    is_training: bool = True
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Full preparation pipeline:
    1. Cleans the data (preserving customer records).
    2. Applies feature engineering.
    3. Separates features X and target y, dropping customerID and Churn from X.
    """
    cleaned_df = clean_customer_data(df, is_training=is_training)
    engineered_df = engineer_features(cleaned_df)

    cols_to_drop = []
    if TARGET_COL in engineered_df.columns:
        cols_to_drop.append(TARGET_COL)
        y = engineered_df[TARGET_COL]
    else:
        y = None

    if 'customerID' in engineered_df.columns:
        cols_to_drop.append('customerID')

    X = engineered_df.drop(columns=cols_to_drop)
    return X, y


class CustomerDataCleaner(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer for customer data cleaning and standardization.
    Enables raw DataFrames (with customerID, raw TotalCharges, etc.) to pass directly through Pipeline.
    Safely drops customerID from model features without altering row counts.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            df_out = clean_customer_data(X, is_training=False)
        else:
            df_out = clean_customer_data(pd.DataFrame(X), is_training=False)
        
        # Drop customerID if present from model features
        if 'customerID' in df_out.columns:
            df_out = df_out.drop(columns=['customerID'])
        return df_out


def build_full_pipeline(classifier) -> Pipeline:
    """
    Constructs an end-to-end Pipeline that takes raw customer records and yields predictions:
    1. Cleaner: drops customerID, converts TotalCharges, standardizes service categories.
    2. FeatureEngineer: creates tenure_group, num_services, avg_monthly_spend, is_monthly_contract.
    3. Preprocessor: scales numeric columns and one-hot encodes categorical columns.
    4. Classifier: trained model.
    """
    return Pipeline([
        ('cleaner', CustomerDataCleaner()),
        ('features', FeatureEngineer()),
        ('preprocessor', build_preprocessor()),
        ('classifier', classifier)
    ])
