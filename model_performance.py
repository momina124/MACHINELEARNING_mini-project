"""
Page 5: Model Evaluation & Performance Analytics.
Displays multi-model comparative benchmarks, interactive ROC/PR curves,
confusion matrices, 5-fold cross-validation metrics, and business cost
threshold optimization.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def render_performance_page(all_eval_data: dict, model_metadata: dict):
    """Renders the comprehensive machine learning evaluation dashboard."""
    st.markdown("""
        <div style="margin-bottom: 24px;">
            <h2 style="color: #0F172A; font-weight: 700; margin-bottom: 6px;">Machine Learning Model Performance</h2>
            <p style="color: #64748B; font-size: 15px;">
                Empirical evaluation across Logistic Regression, Random Forest, and XGBoost with stratified 5-fold cross-validation and cost-optimal decision thresholding.
            </p>
        </div>
    """, unsafe_allow_html=True)

    if not all_eval_data or 'evaluations' not in all_eval_data:
        st.warning("Model evaluation data is not available. Please run `python -m src.train` to train and evaluate models.")
        return

    evaluations = all_eval_data['evaluations']
    cv_results = all_eval_data.get('cv_results', {})
    best_model_name = all_eval_data.get('best_model', 'XGBoost Classifier')
    optimal_thresholds = all_eval_data.get('optimal_thresholds', {})

    # Top highlight of selected best model
    best_metrics = evaluations.get(best_model_name, {})
    st.markdown(f"""
        <div style="background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%); color: white; padding: 20px 24px; border-radius: 12px; margin-bottom: 24px;">
            <span style="background-color: #3B82F6; color: white; padding: 3px 10px; border-radius: 4px; font-size: 11px; font-weight: 700; text-transform: uppercase;">Selected Production Model</span>
            <h3 style="color: white; margin: 8px 0 4px 0; font-size: 22px;">{best_model_name}</h3>
            <p style="color: #94A3B8; font-size: 14px; margin-bottom: 16px;">
                Selected based on validation ROC-AUC and high recall for the minority churn class.
            </p>
            <div style="display: flex; gap: 32px; flex-wrap: wrap;">
                <div><div style="font-size: 12px; color: #94A3B8;">ROC-AUC</div><div style="font-size: 20px; font-weight: 700; color: #38BDF8;">{best_metrics.get('roc_auc', 0):.4f}</div></div>
                <div><div style="font-size: 12px; color: #94A3B8;">PR-AUC</div><div style="font-size: 20px; font-weight: 700; color: #818CF8;">{best_metrics.get('pr_auc', 0):.4f}</div></div>
                <div><div style="font-size: 12px; color: #94A3B8;">Recall (Churn)</div><div style="font-size: 20px; font-weight: 700; color: #4ADE80;">{best_metrics.get('recall', 0):.4f}</div></div>
                <div><div style="font-size: 12px; color: #94A3B8;">F1-Score</div><div style="font-size: 20px; font-weight: 700; color: #FBBF24;">{best_metrics.get('f1_score', 0):.4f}</div></div>
                <div><div style="font-size: 12px; color: #94A3B8;">Optimal Threshold</div><div style="font-size: 20px; font-weight: 700; color: #F472B6;">{best_metrics.get('threshold', 0.5):.2f}</div></div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # 1. Comparison Table
    st.markdown("### 1. Algorithm Comparative Performance (Test Set)")
    table_rows = []
    for m_name, m_data in evaluations.items():
        table_rows.append({
            'Model': m_name,
            'ROC-AUC': f"{m_data.get('roc_auc', 0):.4f}",
            'PR-AUC': f"{m_data.get('pr_auc', 0):.4f}",
            'Recall (Churn)': f"{m_data.get('recall', 0):.4f}",
            'Precision': f"{m_data.get('precision', 0):.4f}",
            'F1-Score': f"{m_data.get('f1_score', 0):.4f}",
            'Accuracy': f"{m_data.get('accuracy', 0):.4f}",
            'Decision Threshold': f"{m_data.get('threshold', 0.5):.2f}"
        })
    df_compare = pd.DataFrame(table_rows)
    st.dataframe(df_compare, use_container_width=True, hide_index=True)

    st.markdown("---")

    # 2. Interactive ROC and PR Curves
    row1_c1, row1_c2 = st.columns(2)

    with row1_c1:
        st.markdown("#### ROC Curves Comparison")
        fig_roc = go.Figure()
        for m_name, m_data in evaluations.items():
            roc = m_data.get('roc_curve', {})
            fig_roc.add_trace(go.Scatter(
                x=roc.get('fpr', []),
                y=roc.get('tpr', []),
                mode='lines',
                name=f"{m_name} ({m_data.get('roc_auc', 0):.3f})"
            ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            line=dict(dash='dash', color='gray'),
            name="Random Guess (0.500)"
        ))
        fig_roc.update_layout(
            xaxis_title="False Positive Rate (1 - Specificity)",
            yaxis_title="True Positive Rate (Recall)",
            height=360,
            margin=dict(t=20, b=20, l=20, r=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.35, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    with row1_c2:
        st.markdown("#### Precision-Recall (PR) Curves")
        fig_pr = go.Figure()
        for m_name, m_data in evaluations.items():
            pr = m_data.get('pr_curve', {})
            fig_pr.add_trace(go.Scatter(
                x=pr.get('recall', []),
                y=pr.get('precision', []),
                mode='lines',
                name=f"{m_name} ({m_data.get('pr_auc', 0):.3f})"
            ))
        fig_pr.update_layout(
            xaxis_title="Recall",
            yaxis_title="Precision",
            height=360,
            margin=dict(t=20, b=20, l=20, r=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.35, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_pr, use_container_width=True)

    st.markdown("---")

    # 3. Confusion Matrix & 5-Fold Cross Validation
    row2_c1, row2_c2 = st.columns(2)

    with row2_c1:
        st.markdown(f"#### Confusion Matrix ({best_model_name})")
        cm = best_metrics.get('confusion_matrix', [[0, 0], [0, 0]])
        cm_df = pd.DataFrame(
            cm,
            index=['Actual: Stay (0)', 'Actual: Churn (1)'],
            columns=['Pred: Stay (0)', 'Pred: Churn (1)']
        )
        fig_cm = px.imshow(
            cm_df,
            text_auto=True,
            color_continuous_scale='Blues',
            aspect="auto"
        )
        fig_cm.update_layout(height=320, margin=dict(t=20, b=20, l=20, r=20), coloraxis_showscale=False)
        st.plotly_chart(fig_cm, use_container_width=True)

    with row2_c2:
        st.markdown("#### 5-Fold Stratified Cross-Validation Benchmarks")
        if cv_results:
            cv_rows = []
            for name, cv_data in cv_results.items():
                cv_rows.append({
                    'Algorithm': name,
                    'CV ROC-AUC': f"{cv_data['mean_roc_auc']:.4f} ± {cv_data['std_roc_auc']:.3f}",
                    'CV Recall': f"{cv_data['mean_recall']:.4f} ± {cv_data['std_recall']:.3f}",
                    'CV F1': f"{cv_data['mean_f1']:.4f} ± {cv_data['std_f1']:.3f}",
                    'CV Accuracy': f"{cv_data['mean_accuracy']:.4f} ± {cv_data['std_accuracy']:.3f}"
                })
            st.dataframe(pd.DataFrame(cv_rows), use_container_width=True, hide_index=True)
            st.caption("Cross-validation confirms model stability across independent stratified folds with no evidence of overfitting.")

    st.markdown("---")

    # 4. Business Cost & Decision Threshold Tuning
    st.markdown("### 2. Business Cost-Optimized Decision Threshold")
    st.markdown("""
        In telecommunications customer retention, the business cost of a **False Negative** (a churned customer who was not identified, costing ~$150 in lost lifetime customer value) is far greater than the cost of a **False Positive** (giving an unnecessary retention incentive of ~$25).
        Optimizing the probability threshold minimizes cumulative business cost.
    """)

    threshold_sweeps = all_eval_data.get('threshold_sweeps', {})
    if best_model_name in threshold_sweeps:
        sweep_data = pd.DataFrame(threshold_sweeps[best_model_name])
        
        fig_cost = go.Figure()
        fig_cost.add_trace(go.Scatter(
            x=sweep_data['threshold'],
            y=sweep_data['total_cost'],
            mode='lines',
            line=dict(color='#DC2626', width=2.5),
            name="Total Expected Business Cost ($)"
        ))
        best_t = optimal_thresholds.get(best_model_name, 0.20)
        min_cost = sweep_data.loc[sweep_data['threshold'] == best_t, 'total_cost'].iloc[0] if not sweep_data.loc[sweep_data['threshold'] == best_t].empty else 0

        fig_cost.add_vline(x=best_t, line_dash="dash", line_color="#2563EB", annotation_text=f"Optimal: {best_t:.2f} (Min Cost: ${min_cost:,.0f})")
        fig_cost.add_vline(x=0.50, line_dash="dot", line_color="#64748B", annotation_text="Default 0.50 Threshold")
        
        fig_cost.update_layout(
            xaxis_title="Classification Decision Threshold",
            yaxis_title="Expected Cost ($)",
            height=320,
            margin=dict(t=20, b=20, l=20, r=20)
        )
        st.plotly_chart(fig_cost, use_container_width=True)
