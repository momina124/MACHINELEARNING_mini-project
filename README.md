# ChurnGuard – Customer Retention Intelligence

> **An End-to-End Machine Learning System for Predicting, Explaining, and Preventing Telecommunications Customer Churn**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-red.svg)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-green.svg)](https://shap.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

---

## 1. Project Overview & Problem Statement

In the telecommunications industry, customer acquisition costs are estimated to be **5 to 25 times higher** than customer retention costs. When subscribers cancel their service, telecom providers lose both predictable Monthly Recurring Revenue (MRR) and significant customer lifetime value (LTV).

**ChurnGuard** is a production-grade machine learning platform that detects early indicators of customer churn, determines each subscriber's churn probability, categorizes them into actionable risk tiers (Low, Medium, High), provides transparent model explanations using cooperative game theory (SHAP), and prescribes targeted retention offers before account cancellation takes place.

### Core Objectives
1. **Accurate Predictive Scoring:** Quantify customer churn propensity with high recall to avoid costly false negatives.
2. **Explainable AI (XAI):** Translate complex non-linear gradient-boosted decision logic into intuitive, human-understandable drivers via SHAP.
3. **Cost-Optimal Thresholding:** Shift decision cutoffs based on asymmetrical business costs ($150 loss per churned customer vs $25 retention campaign offer).
4. **Actionable Decision Support:** Provide an executive-level interactive dashboard for cohort analytics, real-time single-customer scoring, batch CSV predictions, and retention campaign ROI simulation.

---

## 2. Technology Stack

- **Core Runtime:** Python 3.10+
- **Data Engineering:** Pandas 2.0+, NumPy 1.24+
- **Machine Learning Algorithms:** Scikit-learn, XGBoost
- **Class Imbalance Handling:** Imbalanced-learn (SMOTE), Class Weight Balancing, XGBoost `scale_pos_weight`
- **Model Explainability:** SHAP (SHapley Additive exPlanations)
- **Data Visualizations:** Plotly Interactive Graphs, Matplotlib, Seaborn
- **Persistence & Deployment:** Joblib serialization, Streamlit Web Application
- **Testing & Quality Assurance:** Pytest test suite with 100% core coverage

---

## 3. Dataset Architecture

The project utilizes the canonical **IBM Telco Customer Churn** dataset:
- **Total Records:** 7,043 customer accounts (7,021 deduplicated)
- **Original Attributes:** 21 columns encompassing demographics, subscribed services, account tenure, contract types, payment methods, and billing details.
- **Target Variable:** `Churn` (Binary: Yes = 1, No = 0).
- **Target Distribution:** Imbalanced (~26.5% churners, ~73.5% retained).

### Expected Data Schema
| Category | Attributes |
| :--- | :--- |
| **Demographics** | `gender`, `SeniorCitizen`, `Partner`, `Dependents` |
| **Account Info** | `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod` |
| **Telephony** | `PhoneService`, `MultipleLines` |
| **Internet & Cloud** | `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport` |
| **Entertainment** | `StreamingTV`, `StreamingMovies` |
| **Financials** | `MonthlyCharges`, `TotalCharges` |

---

## 4. Machine Learning Pipeline Architecture

```
[ Raw Customer Data ]
         │
         ▼
[ Stage 1: Data Cleaning ]
  ├── Convert TotalCharges to float; zero-fill for tenure = 0
  ├── Standardize redundant categorical values ('No internet service' -> 'No')
  └── Drop non-predictive identifiers ('customerID') & deduplicate
         │
         ▼
[ Stage 2: Feature Engineering ]
  ├── tenure_group: Binned lifecycle buckets (0–12m, 13–24m, 25–48m, 49+m)
  ├── num_services: Cumulative active service subscription count (0–8)
  ├── avg_monthly_spend: TotalCharges / tenure (safe division zero-handling)
  └── is_monthly_contract: Binary indicator for month-to-month contracts
         │
         ▼
[ Stage 3: ColumnTransformer Preprocessing ]
  ├── StandardScaler: Numeric features (tenure, MonthlyCharges, TotalCharges, etc.)
  └── OneHotEncoder: Categorical features (handle_unknown='ignore')
         │
         ▼
[ Stage 4: Stratified 5-Fold Cross-Validation & Multi-Model Benchmarking ]
  ├── Logistic Regression (balanced class weights)
  ├── Random Forest Classifier (balanced class weights, depth-controlled)
  ├── XGBoost Classifier (scale_pos_weight = 2.78, gradient boosted trees)
  └── Random Forest + SMOTE (synthetic minority oversampling on training folds only)
         │
         ▼
[ Stage 5: Cost-Optimal Threshold Sweeping ]
  └── Optimize cutoff balancing $150 False Negative vs $25 False Positive costs
         │
         ▼
[ Stage 6: Explainability & Artifact Serialization ]
  ├── SHAP TreeExplainer global feature ranking & local force attributions
  └── Joblib export of end-to-end Pipeline, metadata, and decision threshold
```

---

## 5. Model Evaluation & Benchmark Results

Evaluated on an independent 20% stratified test holdout set (1,405 customers):

| Algorithm | ROC-AUC | PR-AUC | Recall (Churn) | Precision | F1-Score | Optimal Threshold |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost Classifier (Selected)** | **0.8453** | **0.6599** | **0.9704** | 0.5052 | **0.5610** | **0.20** |
| **Random Forest Classifier** | 0.8428 | 0.6599 | 0.9435 | 0.4086 | 0.5707 | 0.28 |
| **Logistic Regression** | 0.8398 | 0.6376 | 0.9355 | 0.4357 | 0.5944 | 0.32 |

### 5-Fold Stratified Cross-Validation
- **XGBoost CV ROC-AUC:** `0.8466 ± 0.0186`
- **Random Forest CV ROC-AUC:** `0.8474 ± 0.0176`
- **Logistic Regression CV ROC-AUC:** `0.8459 ± 0.0174`
- **Random Forest + SMOTE CV ROC-AUC:** `0.8421 ± 0.0165`

### Business Cost Analysis
By tuning the classification threshold to `0.20`, the model captures **97.04% of all churning customers** on the test set, preventing substantial revenue loss while keeping false-positive campaign costs strictly bounded.

---

## 6. Project Directory Structure

```
churn-prediction/
│
├── app.py                          # Streamlit multi-page application entry point
├── requirements.txt                # Production dependency specifications
├── README.md                       # Comprehensive documentation
│
├── data/
│   ├── raw/                        # Original IBM Telco CSV
│   │   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   └── processed/                  # Cleaned and standardized dataset
│       └── cleaned_telco_churn.csv
│
├── models/
│   ├── churn_pipeline.joblib       # Serialized production inference pipeline
│   ├── threshold.joblib            # Cost-optimal decision cutoff
│   ├── model_metadata.json         # Evaluation metrics and dataset statistics
│   ├── all_models_eval.json        # Comparative benchmark data for UI rendering
│   └── shap_summary.joblib         # Precomputed global SHAP values
│
├── notebooks/
│   └── eda.ipynb                   # Exploratory Data Analysis Jupyter Notebook
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py       # Data cleaning and ColumnTransformer construction
│   ├── features.py                 # Domain feature engineering & Scikit transformer
│   ├── train.py                    # Multi-model training, 5-fold CV, and threshold tuning
│   ├── evaluate.py                 # Evaluation metrics, ROC/PR curves, and cost sweep
│   ├── explainability.py           # SHAP calculation and natural-language interpretations
│   └── predict.py                  # Single-customer & batch inference engine
│
├── app_pages/
│   ├── __init__.py
│   ├── analytics.py                # Page 2: Interactive EDA cohort explorer
│   ├── prediction.py               # Page 3: Real-time customer prediction form
│   ├── batch_prediction.py         # Page 4: Bulk CSV prediction & export
│   ├── model_performance.py        # Page 5: Multi-model evaluation & cost curves
│   ├── explainability.py           # Page 6: SHAP global & local feature attributions
│   └── recommendations.py          # Page 7: Strategic retention playbook & ROI simulator
│
├── reports/
│   └── figures/                    # 300 DPI publication-quality figures
│       ├── roc_curve.png
│       ├── pr_curve.png
│       ├── confusion_matrix.png
│       └── feature_importance.png
│
└── tests/
    ├── __init__.py
    ├── test_preprocessing.py       # Unit tests for cleaning, edge cases, and transformations
    └── test_prediction.py          # Unit tests for scoring, batch jobs, and recommendations
```

---

## 7. Installation & Setup Instructions

### 1. Clone or Open the Repository
```bash
cd "C:\Users\BUTUL FATIMA\.gemini\antigravity\scratch\churn-prediction"
```

### 2. Set Up a Virtual Environment (Recommended)
```bash
python -m venv venv
venv\Scripts\activate   # On Windows
# or: source venv/bin/activate  # On Linux / macOS
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify Core Test Suite
Run the 11 unit tests covering edge cases (tenure=0, missing total charges, unknown categories):
```bash
pytest -v
```

---

## 8. How to Train the Model

To execute the complete end-to-end training pipeline, run:
```bash
python -m src.train
```

This script automatically:
1. Loads raw data from `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`.
2. Cleans records and writes `data/processed/cleaned_telco_churn.csv`.
3. Performs stratified 80/20 train/test splitting (`random_state=42`).
4. Executes 5-fold cross-validation on Logistic Regression, Random Forest, XGBoost, and SMOTE.
5. Optimizes the decision threshold using business cost curves ($150 FN vs $25 FP).
6. Automatically selects the best algorithm (XGBoost Classifier).
7. Serializes the complete end-to-end inference pipeline into `models/churn_pipeline.joblib`.
8. Generates high-resolution evaluation figures in `reports/figures/`.

---

## 9. How to Run the Web Application

Launch the Streamlit web dashboard:
```bash
streamlit run app.py
```
Open your web browser at `http://localhost:8501`.

### Application Pages Overview
- **Home / Overview:** Executive retention overview, key performance indicators, model status.
- **Customer Analytics:** Interactive Plotly cohort exploration, churn distributions by contract, payment, and internet service.
- **Churn Prediction:** Real-time single customer risk assessment with preset profiles, visual probability gauge, and local SHAP factor attribution.
- **Batch Prediction:** Bulk CSV customer scoring, risk tier distributions, and downloadable enriched CSVs.
- **Model Performance:** Comparative metrics, ROC/PR curves, confusion matrices, and cost-benefit threshold curves.
- **Model Explainability:** SHAP feature importance bar charts, risk amplifiers vs stabilizers, and non-technical business interpretations.
- **Retention Recommendations:** Segment-specific playbooks and interactive campaign ROI economic modeling.

---

## 10. Key Insights & Recommended Retention Strategies

1. **Contract Duration Migration:**
   - *Observation:* Month-to-month contracts have an alarming ~42% churn rate vs <3% for 2-year contracts.
   - *Strategy:* Offer a 15% discount or $50 billing credit for switching to a 12-month commitment.
2. **Fiber Optic Support Bundling:**
   - *Observation:* Fiber optic customers churn at 41.9% when lacking tech support or security add-ons.
   - *Strategy:* Provide complimentary 6-month Tech Support and Online Security to eliminate early service frustration.
3. **Auto-Pay Friction Removal:**
   - *Observation:* Electronic check payers churn at 45.3% vs 16.7% for automatic bank transfer.
   - *Strategy:* Incentivize auto-pay enrollment with a recurring $5 monthly discount.
4. **Early-Lifecycle Concierge Onboarding:**
   - *Observation:* The first 12 months represent the highest attrition vulnerability.
   - *Strategy:* Institute proactive 30-day and 60-day customer service check-ins.

---

## 11. Limitations & Future Enhancements

- **Temporal Validation:** The current dataset represents a cross-sectional snapshot. Future iterations will incorporate time-series longitudinal billing events.
- **Customer Lifetime Value (LTV) Joint Modeling:** Integrate survival analysis (Cox Proportional Hazards) alongside binary churn classification.
- **Real-Time Webhook Integration:** Connect batch and single predictions to CRM platforms (HubSpot, Salesforce) via REST APIs.

---

## 12. License & Author Attribution

This project is developed for academic, portfolio, and enterprise decision-support demonstration under the MIT License.
