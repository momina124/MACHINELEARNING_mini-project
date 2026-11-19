"""
Page 4: Batch Customer Churn Prediction.
Enables batch CSV file uploads, validates schema, applies vector inference
via the trained pipeline, categorizes risk distributions, and allows
download of enriched results with risk levels and probability scores.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from src.predict import predict_batch_customers


def render_batch_prediction_page(pipeline, decision_threshold: float, low_risk_thresh: float = 0.30, high_risk_thresh: float = 0.60):
    """Renders the batch prediction interface."""
    st.markdown("""
        <div style="margin-bottom: 24px;">
            <h2 style="color: #0F172A; font-weight: 700; margin-bottom: 6px;">Batch Customer Churn Prediction</h2>
            <p style="color: #64748B; font-size: 15px;">
                Upload a customer dataset CSV to score churn propensity, flag high-risk accounts, and export enriched retention intelligence.
            </p>
        </div>
    """, unsafe_allow_html=True)

    if pipeline is None:
        st.error("Trained machine learning pipeline is not available. Please train the model first.")
        return

    # Sample template generator for quick user testing
    col_dl, col_blank = st.columns([1, 2])
    with col_dl:
        sample_template_df = pd.DataFrame([
            {
                'customerID': 'SAMPLE-7001', 'gender': 'Female', 'SeniorCitizen': 0, 'Partner': 'No',
                'Dependents': 'No', 'tenure': 2, 'PhoneService': 'Yes', 'MultipleLines': 'No',
                'InternetService': 'Fiber optic', 'OnlineSecurity': 'No', 'OnlineBackup': 'No',
                'DeviceProtection': 'No', 'TechSupport': 'No', 'StreamingTV': 'Yes', 'StreamingMovies': 'Yes',
                'Contract': 'Month-to-month', 'PaperlessBilling': 'Yes', 'PaymentMethod': 'Electronic check',
                'MonthlyCharges': 85.50, 'TotalCharges': 171.00
            },
            {
                'customerID': 'SAMPLE-7002', 'gender': 'Male', 'SeniorCitizen': 0, 'Partner': 'Yes',
                'Dependents': 'Yes', 'tenure': 60, 'PhoneService': 'Yes', 'MultipleLines': 'Yes',
                'InternetService': 'DSL', 'OnlineSecurity': 'Yes', 'OnlineBackup': 'Yes',
                'DeviceProtection': 'Yes', 'TechSupport': 'Yes', 'StreamingTV': 'No', 'StreamingMovies': 'No',
                'Contract': 'Two year', 'PaperlessBilling': 'No', 'PaymentMethod': 'Credit card (automatic)',
                'MonthlyCharges': 55.20, 'TotalCharges': 3312.00
            },
            {
                'customerID': 'SAMPLE-7003', 'gender': 'Male', 'SeniorCitizen': 1, 'Partner': 'No',
                'Dependents': 'No', 'tenure': 14, 'PhoneService': 'Yes', 'MultipleLines': 'Yes',
                'InternetService': 'Fiber optic', 'OnlineSecurity': 'No', 'OnlineBackup': 'Yes',
                'DeviceProtection': 'No', 'TechSupport': 'No', 'StreamingTV': 'No', 'StreamingMovies': 'No',
                'Contract': 'Month-to-month', 'PaperlessBilling': 'Yes', 'PaymentMethod': 'Electronic check',
                'MonthlyCharges': 79.80, 'TotalCharges': 1117.20
            },
            {
                'customerID': 'SAMPLE-7004', 'gender': 'Female', 'SeniorCitizen': 0, 'Partner': 'Yes',
                'Dependents': 'No', 'tenure': 36, 'PhoneService': 'No', 'MultipleLines': 'No phone service',
                'InternetService': 'DSL', 'OnlineSecurity': 'Yes', 'OnlineBackup': 'No',
                'DeviceProtection': 'Yes', 'TechSupport': 'Yes', 'StreamingTV': 'Yes', 'StreamingMovies': 'No',
                'Contract': 'One year', 'PaperlessBilling': 'No', 'PaymentMethod': 'Bank transfer (automatic)',
                'MonthlyCharges': 45.30, 'TotalCharges': 1630.80
            }
        ])
        sample_csv = sample_template_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Sample Batch Template (CSV)",
            data=sample_csv,
            file_name="telco_sample_batch_template.csv",
            mime="text/csv"
        )

    # File uploader
    uploaded_file = st.file_uploader("Upload Customer CSV File", type=["csv"])

    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded {len(input_df):,} customer records.")

            with st.spinner("Executing machine learning predictions across batch records..."):
                results_df, summary = predict_batch_customers(
                    df=input_df,
                    pipeline=pipeline,
                    decision_threshold=decision_threshold,
                    low_risk_thresh=low_risk_thresh,
                    high_risk_thresh=high_risk_thresh
                )

            # Display KPI Summary Cards
            st.markdown("### Batch Prediction Overview")
            k1, k2, k3, k4, k5 = st.columns(5)
            with k1:
                st.metric("Total Records", f"{summary['total_customers']:,}")
            with k2:
                st.metric("Predicted Churners", f"{summary['predicted_churners']:,}")
            with k3:
                st.metric("High Risk Accounts", f"{summary['high_risk_count']:,}", f"{summary['high_risk_pct']}% of cohort", delta_color="inverse")
            with k4:
                st.metric("Medium Risk", f"{summary['medium_risk_count']:,}")
            with k5:
                st.metric("Low Risk", f"{summary['low_risk_count']:,}")

            # Risk Distribution Breakdown Chart
            st.markdown("<br>", unsafe_allow_html=True)
            col_chart, col_empty = st.columns([1, 1])
            with col_chart:
                risk_dist = results_df['Risk_Level'].value_counts().reset_index()
                risk_dist.columns = ['Risk_Level', 'Count']
                fig_risk = px.pie(
                    risk_dist,
                    names='Risk_Level',
                    values='Count',
                    color='Risk_Level',
                    color_discrete_map={'High': '#EF4444', 'Medium': '#F59E0B', 'Low': '#10B981'},
                    hole=0.45,
                    title="Risk Tier Distribution"
                )
                fig_risk.update_layout(height=260, margin=dict(t=30, b=10, l=10, r=10))
                st.plotly_chart(fig_risk, use_container_width=True)

            # Prediction Results Table
            st.markdown("### Scored Customer Records")
            display_cols = [c for c in ['customerID', 'Prediction', 'Churn_Probability', 'Risk_Level', 'Contract', 'tenure', 'MonthlyCharges', 'PaymentMethod'] if c in results_df.columns]
            st.dataframe(results_df[display_cols], use_container_width=True, height=350)

            # Export Button
            csv_output = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Full Prediction Results (CSV)",
                data=csv_output,
                file_name="churn_guard_batch_predictions.csv",
                mime="text/csv",
                type="primary"
            )

        except Exception as e:
            st.error(f"Error processing batch dataset: {str(e)}")
            st.info("Tip: Ensure your CSV contains required columns like 'tenure', 'MonthlyCharges', and 'Contract'. You can download the sample template above.")
