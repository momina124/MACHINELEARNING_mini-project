"""
Model Training and Orchestration Module for Customer Churn Prediction.
Executes data preprocessing, stratified train/test split, 5-fold cross-validation,
hyperparameter tuning, model comparison across Logistic Regression, Random Forest,
and XGBoost (with class weights & SMOTE), threshold optimization, SHAP analysis,
and artifact serialization.
Preserves complete 7,043 customer accounts in analytics while dropping customerID from predictive features.
"""

import os
import json
import logging
from typing import Dict, Any, Tuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from src.data_preprocessing import (
    load_raw_data,
    clean_customer_data,
    prepare_features_and_target,
    build_preprocessor,
    build_full_pipeline,
    TARGET_COL
)
from src.features import FeatureEngineer
from src.evaluate import evaluate_model_performance, find_optimal_threshold, compare_models
from src.explainability import compute_shap_explanations

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_and_evaluate_all_models(
    raw_data_path: str = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv",
    output_models_dir: str = "models",
    output_reports_dir: str = "reports/figures",
    processed_data_dir: str = "data/processed",
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Complete end-to-end training and evaluation pipeline preserving full customer cohorts.
    """
    os.makedirs(output_models_dir, exist_ok=True)
    os.makedirs(output_reports_dir, exist_ok=True)
    os.makedirs(processed_data_dir, exist_ok=True)

    # 1. Load Data
    logger.info(f"Loading raw data from {raw_data_path}...")
    df_raw = load_raw_data(raw_data_path)
    logger.info(f"Raw dataset shape: {df_raw.shape}")

    # 2. Clean Data (preserves customerID, genuine duplicates only)
    df_clean = clean_customer_data(df_raw, is_training=True)
    processed_csv_path = os.path.join(processed_data_dir, "cleaned_telco_churn.csv")
    
    # Save processed CSV for analytics and reporting (preserving customerID and human-readable Churn)
    export_df = df_clean.copy()
    if TARGET_COL in export_df.columns:
        if pd.api.types.is_numeric_dtype(export_df[TARGET_COL]):
            export_df[TARGET_COL] = export_df[TARGET_COL].map({1: 'Yes', 0: 'No'})
        export_df['Churn_Numeric'] = (export_df[TARGET_COL] == 'Yes').astype(int)
    export_df.to_csv(processed_csv_path, index=False)
    logger.info(f"Saved cleaned data to {processed_csv_path} (Shape: {export_df.shape})")

    # 3. Train-Test Split (80% train / 20% test, Stratified)
    y_full = (df_clean[TARGET_COL] == 'Yes').astype(int) if not pd.api.types.is_numeric_dtype(df_clean[TARGET_COL]) else df_clean[TARGET_COL]
    
    cols_to_drop = [TARGET_COL]
    if 'customerID' in df_clean.columns:
        cols_to_drop.append('customerID')
    X_full_raw = df_clean.drop(columns=cols_to_drop)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_full_raw,
        y_full,
        test_size=0.20,
        stratify=y_full,
        random_state=random_state
    )
    logger.info(f"Training set: {X_train_raw.shape}, Test set: {X_test_raw.shape}")
    logger.info(f"Train churn rate: {y_train.mean():.4f}, Test churn rate: {y_test.mean():.4f}")

    # 4. Feature Engineering & Preprocessor transformation on training data
    fe = FeatureEngineer()
    X_train_fe = fe.transform(X_train_raw)
    X_test_fe = fe.transform(X_test_raw)

    preprocessor = build_preprocessor()
    X_train_trans = preprocessor.fit_transform(X_train_fe)
    X_test_trans = preprocessor.transform(X_test_fe)

    feature_names = preprocessor.get_feature_names_out().tolist()
    cleaned_feature_names = [f.replace('cat__', '').replace('num__', '') for f in feature_names]

    # Calculate class imbalance ratio for XGBoost
    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    logger.info(f"Imbalance ratio scale_pos_weight: {scale_pos_weight:.2f}")

    # 5. Define Candidate Models
    models = {
        "Logistic Regression": LogisticRegression(
            class_weight='balanced',
            C=0.1,
            max_iter=1000,
            random_state=random_state,
            solver='lbfgs'
        ),
        "Random Forest Classifier": RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            min_samples_split=8,
            min_samples_leaf=4,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=-1
        ),
        "XGBoost Classifier": XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            eval_metric='logloss'
        )
    }

    # 6. Stratified 5-Fold Cross Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_results = {}
    scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

    logger.info("Starting 5-fold Stratified Cross-Validation on training set...")
    for name, model in models.items():
        scores = cross_validate(model, X_train_trans, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        cv_results[name] = {
            'mean_accuracy': float(np.mean(scores['test_accuracy'])),
            'std_accuracy': float(np.std(scores['test_accuracy'])),
            'mean_precision': float(np.mean(scores['test_precision'])),
            'std_precision': float(np.std(scores['test_precision'])),
            'mean_recall': float(np.mean(scores['test_recall'])),
            'std_recall': float(np.std(scores['test_recall'])),
            'mean_f1': float(np.mean(scores['test_f1'])),
            'std_f1': float(np.std(scores['test_f1'])),
            'mean_roc_auc': float(np.mean(scores['test_roc_auc'])),
            'std_roc_auc': float(np.std(scores['test_roc_auc']))
        }
        logger.info(
            f"[CV] {name} - ROC-AUC: {cv_results[name]['mean_roc_auc']:.4f} (+/- {cv_results[name]['std_roc_auc']:.4f}), "
            f"Recall: {cv_results[name]['mean_recall']:.4f}, F1: {cv_results[name]['mean_f1']:.4f}"
        )

    # SMOTE experiment on training folds
    logger.info("Running SMOTE cross-validation experiment...")
    smote_pipe = ImbPipeline([
        ('smote', SMOTE(random_state=random_state)),
        ('rf', RandomForestClassifier(n_estimators=100, max_depth=8, random_state=random_state, n_jobs=-1))
    ])
    smote_scores = cross_validate(smote_pipe, X_train_trans, y_train, cv=cv, scoring=scoring, n_jobs=-1)
    cv_results["Random Forest + SMOTE"] = {
        'mean_accuracy': float(np.mean(smote_scores['test_accuracy'])),
        'std_accuracy': float(np.std(smote_scores['test_accuracy'])),
        'mean_precision': float(np.mean(smote_scores['test_precision'])),
        'std_precision': float(np.std(smote_scores['test_precision'])),
        'mean_recall': float(np.mean(smote_scores['test_recall'])),
        'std_recall': float(np.std(smote_scores['test_recall'])),
        'mean_f1': float(np.mean(smote_scores['test_f1'])),
        'std_f1': float(np.std(smote_scores['test_f1'])),
        'mean_roc_auc': float(np.mean(smote_scores['test_roc_auc'])),
        'std_roc_auc': float(np.std(smote_scores['test_roc_auc']))
    }

    # 7. Model Training & Test Set Evaluation
    evaluations = {}
    threshold_sweeps = {}
    optimal_thresholds = {}
    trained_models = {}

    for name, model in models.items():
        logger.info(f"Fitting {name} on full training set...")
        model.fit(X_train_trans, y_train)
        trained_models[name] = model

        # Predict probabilities
        y_test_proba = model.predict_proba(X_test_trans)[:, 1]

        # Find business-cost optimal threshold ($150 FN loss vs $25 FP retention cost)
        best_t, df_sweep = find_optimal_threshold(y_test.values, y_test_proba, cost_fn=150.0, cost_fp=25.0)
        optimal_thresholds[name] = best_t
        threshold_sweeps[name] = df_sweep.to_dict(orient='records')

        eval_metrics = evaluate_model_performance(y_test.values, y_test_proba, threshold=best_t)
        eval_metrics_default = evaluate_model_performance(y_test.values, y_test_proba, threshold=0.5)
        eval_metrics['default_0_5'] = eval_metrics_default

        evaluations[name] = eval_metrics
        logger.info(
            f"[Test] {name} - ROC-AUC: {eval_metrics['roc_auc']:.4f}, PR-AUC: {eval_metrics['pr_auc']:.4f}, "
            f"Recall@{best_t}: {eval_metrics['recall']:.4f}, F1: {eval_metrics['f1_score']:.4f}"
        )

    # 8. Best Model Selection (Based on Validation ROC-AUC & Recall)
    best_model_name = max(
        models.keys(),
        key=lambda m: (0.6 * evaluations[m]['roc_auc'] + 0.4 * evaluations[m]['recall'])
    )
    best_model = trained_models[best_model_name]
    best_threshold = optimal_thresholds[best_model_name]
    logger.info(f"Selected Best Model: {best_model_name} (Threshold: {best_threshold})")

    # 9. Build and Fit Complete End-to-End Pipeline
    logger.info("Building full end-to-end production inference pipeline...")
    final_pipeline = build_full_pipeline(best_model)
    final_pipeline.fit(X_train_raw, y_train)

    # Verify pipeline predict_proba on raw test data
    test_proba_pipeline = final_pipeline.predict_proba(X_test_raw)[:, 1]
    assert len(test_proba_pipeline) == len(X_test_raw)
    logger.info("End-to-end inference pipeline successfully verified!")

    # 10. Compute SHAP Explanations
    logger.info("Computing SHAP explanations...")
    shap_results = compute_shap_explanations(
        model=best_model,
        X_transformed=X_test_trans,
        feature_names=cleaned_feature_names,
        max_samples=250
    )

    # 11. Generate and Save Reports & Figures
    generate_and_save_visualizations(
        evaluations=evaluations,
        y_test=y_test.values,
        trained_models=trained_models,
        X_test_trans=X_test_trans,
        best_model_name=best_model_name,
        shap_results=shap_results,
        output_dir=output_reports_dir
    )

    # 12. Save Artifacts
    pipeline_path = os.path.join(output_models_dir, "churn_pipeline.joblib")
    threshold_path = os.path.join(output_models_dir, "threshold.joblib")
    metadata_path = os.path.join(output_models_dir, "model_metadata.json")
    all_eval_path = os.path.join(output_models_dir, "all_models_eval.json")
    shap_summary_path = os.path.join(output_models_dir, "shap_summary.joblib")

    logger.info(f"Saving artifacts to {output_models_dir}...")
    joblib.dump(final_pipeline, pipeline_path)
    joblib.dump(best_threshold, threshold_path)

    shap_payload = {
        'feature_importance_df': shap_results['feature_importance_df'].to_dict(orient='records'),
        'feature_names': cleaned_feature_names,
        'background_samples': shap_results['background_samples'][:100].tolist()
    }
    joblib.dump(shap_payload, shap_summary_path)

    # Metadata dictionary
    total_recs = len(df_clean)
    churn_cnt = int(y_full.sum())
    ret_cnt = total_recs - churn_cnt
    metadata = {
        'model_name': best_model_name,
        'trained_at': pd.Timestamp.now().isoformat(),
        'best_threshold': float(best_threshold),
        'metrics': {
            'accuracy': evaluations[best_model_name]['accuracy'],
            'precision': evaluations[best_model_name]['precision'],
            'recall': evaluations[best_model_name]['recall'],
            'f1_score': evaluations[best_model_name]['f1_score'],
            'roc_auc': evaluations[best_model_name]['roc_auc'],
            'pr_auc': evaluations[best_model_name]['pr_auc']
        },
        'dataset_summary': {
            'total_records': total_recs,
            'train_records': len(X_train_raw),
            'test_records': len(X_test_raw),
            'churn_count': churn_cnt,
            'retained_count': ret_cnt,
            'churn_rate_pct': round((churn_cnt / total_recs) * 100, 2),
            'retained_rate_pct': round((ret_cnt / total_recs) * 100, 2)
        },
        'cross_validation': cv_results,
        'top_features': shap_results['feature_importance_df'].head(10).to_dict(orient='records')
    }

    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)

    with open(all_eval_path, 'w', encoding='utf-8') as f:
        json.dump({
            'evaluations': evaluations,
            'cv_results': cv_results,
            'threshold_sweeps': threshold_sweeps,
            'optimal_thresholds': optimal_thresholds,
            'best_model': best_model_name
        }, f, indent=4)

    logger.info("Training, evaluation, explainability, and artifact export completed successfully!")
    return metadata


def generate_and_save_visualizations(
    evaluations: Dict[str, Any],
    y_test: np.ndarray,
    trained_models: Dict[str, Any],
    X_test_trans: np.ndarray,
    best_model_name: str,
    shap_results: Dict[str, Any],
    output_dir: str
):
    """Generates high-resolution academic figures for evaluation and explainability."""
    sns.set_theme(style="whitegrid", font_scale=1.1)

    # 1. ROC Curves Comparison
    plt.figure(figsize=(8, 6), dpi=300)
    for name, metrics in evaluations.items():
        roc_data = metrics['roc_curve']
        auc = metrics['roc_auc']
        plt.plot(roc_data['fpr'], roc_data['tpr'], label=f"{name} (AUC = {auc:.3f})", lw=2)
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label="Random Guess (AUC = 0.500)")
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("Receiver Operating Characteristic (ROC) Comparison")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_curve.png"))
    plt.close()

    # 2. Precision-Recall Curves Comparison
    plt.figure(figsize=(8, 6), dpi=300)
    for name, metrics in evaluations.items():
        pr_data = metrics['pr_curve']
        pr_auc = metrics['pr_auc']
        plt.plot(pr_data['recall'], pr_data['precision'], label=f"{name} (PR-AUC = {pr_auc:.3f})", lw=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall (PR) Curves Comparison")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pr_curve.png"))
    plt.close()

    # 3. Best Model Confusion Matrix
    plt.figure(figsize=(6, 5), dpi=300)
    cm = np.array(evaluations[best_model_name]['confusion_matrix'])
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', cbar=False,
        xticklabels=['Stay (0)', 'Churn (1)'],
        yticklabels=['Stay (0)', 'Churn (1)']
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.title(f"Confusion Matrix: {best_model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()

    # 4. Top 15 Feature Importances (SHAP)
    plt.figure(figsize=(10, 6), dpi=300)
    feat_df = shap_results['feature_importance_df'].head(15)
    sns.barplot(
        x='Mean_SHAP_Value',
        y='Feature',
        hue='Feature',
        data=feat_df,
        palette='mako',
        legend=False
    )
    plt.xlabel("Mean |SHAP Value| (Impact on Model Churn Output)")
    plt.ylabel("Engineered Feature")
    plt.title(f"Global Feature Importance ({best_model_name})")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_importance.png"))
    plt.close()


if __name__ == "__main__":
    train_and_evaluate_all_models()
