"""
Page 3: Interactive Single Customer Churn Prediction.
Provides a comprehensive form for real-time customer profiling,
executes inference through the trained scikit-learn pipeline,
evaluates risk tiers, generates local SHAP factor attributions,
and displays targeted customer retention recommendations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from src.predict import predict_single_customer, categorize_risk_tier
from src.explainability import explain_single_customer


def render_prediction_page(pipeline, decision_threshold: float, model_metadata: dict, shap_data: dict, low_risk_thresh: float = 0.30, high_risk_thresh: float = 0.60):
    """Renders the real-time single customer inference engine."""
    st.markdown("""
        <div style="margin-bottom: 24px;">
            <h2 style="color: #0F172A; font-weight: 700; margin-bottom: 6px;">Customer Churn Intelligence Engine</h2>
            <p style="color: #64748B; font-size: 15px;">
                Enter customer contract, service, and billing attributes to predict churn probability, evaluate risk level, and inspect driving factors.
            </p>
        </div>
    """, unsafe_allow_html=True)

    if pipeline is None:
        st.error("Trained machine learning pipeline is not available. Please run `python -m src.train` first.")
        return

    # Quick Presets Selector
    preset = st.selectbox(
        "Load Quick Demonstration Profile:",
        [
            "Custom Manual Profile",
            "High Risk: New Fiber Optic Customer (Month-to-month, Electronic Check)",
            "Low Risk: Long-Term Loyal Customer (2-Year Contract, Auto-Pay)",
            "Medium Risk: Mid-Tenure Fiber Customer with Technical Support Gaps"
        ]
    )

    # Defaults based on preset
    if preset == "High Risk: New Fiber Optic Customer (Month-to-month, Electronic Check)":
        d_gender, d_senior, d_partner, d_dep = "Female", 0, "No", "No"
        d_tenure, d_contract, d_pay, d_paperless = 3, "Month-to-month", "Electronic check", "Yes"
        d_phone, d_mult = "Yes", "No"
        d_internet, d_sec, d_backup, d_device, d_supp, d_tv, d_mov = "Fiber optic", "No", "No", "No", "No", "Yes", "Yes"
        d_monthly, d_total = 89.50, 268.50
    elif preset == "Low Risk: Long-Term Loyal Customer (2-Year Contract, Auto-Pay)":
        d_gender, d_senior, d_partner, d_dep = "Male", 0, "Yes", "Yes"
        d_tenure, d_contract, d_pay, d_paperless = 62, "Two year", "Bank transfer (automatic)", "No"
        d_phone, d_mult = "Yes", "Yes"
        d_internet, d_sec, d_backup, d_device, d_supp, d_tv, d_mov = "DSL", "Yes", "Yes", "Yes", "Yes", "No", "No"
        d_monthly, d_total = 58.20, 3608.40
    elif preset == "Medium Risk: Mid-Tenure Fiber Customer with Technical Support Gaps":
        d_gender, d_senior, d_partner, d_dep = "Male", 1, "Yes", "No"
        d_tenure, d_contract, d_pay, d_paperless = 18, "Month-to-month", "Credit card (automatic)", "Yes"
        d_phone, d_mult = "Yes", "Yes"
        d_internet, d_sec, d_backup, d_device, d_supp, d_tv, d_mov = "Fiber optic", "No", "Yes", "No", "No", "Yes", "No"
        d_monthly, d_total = 84.10, 1513.80
    else:
        d_gender, d_senior, d_partner, d_dep = "Female", 0, "No", "No"
        d_tenure, d_contract, d_pay, d_paperless = 12, "Month-to-month", "Electronic check", "Yes"
        d_phone, d_mult = "Yes", "No"
        d_internet, d_sec, d_backup, d_device, d_supp, d_tv, d_mov = "Fiber optic", "No", "No", "No", "No", "No", "No"
        d_monthly, d_total = 70.00, 840.00

    # Input Form organized in 4 cards
    with st.form("customer_prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("##### 1. Demographics & Account Identity")
            c_gender = st.selectbox("Customer Gender", ["Female", "Male"], index=0 if d_gender == "Female" else 1)
            c_senior = st.selectbox("Senior Citizen Status", ["0 (No)", "1 (Yes)"], index=d_senior)
            senior_val = 1 if "1" in c_senior else 0
            c_partner = st.selectbox("Has Partner", ["No", "Yes"], index=0 if d_partner == "No" else 1)
            c_dependents = st.selectbox("Has Dependents", ["No", "Yes"], index=0 if d_dep == "No" else 1)

            st.markdown("##### 2. Account & Contract Setup")
            c_tenure = st.slider("Customer Tenure (Months with Provider)", min_value=0, max_value=72, value=d_tenure)
            c_contract = st.selectbox("Contract Terms", ["Month-to-month", "One year", "Two year"],
                                      index=["Month-to-month", "One year", "Two year"].index(d_contract))
            c_payment = st.selectbox(
                "Payment Method",
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
                index=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"].index(d_pay)
            )
            c_paperless = st.selectbox("Paperless Billing", ["Yes", "No"], index=0 if d_paperless == "Yes" else 1)

        with col2:
            st.markdown("##### 3. Subscribed Telecommunication Services")
            c_phone = st.selectbox("Phone Service", ["Yes", "No"], index=0 if d_phone == "Yes" else 1)
            c_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"],
                                   index=["No", "Yes", "No phone service"].index(d_mult))
            c_internet = st.selectbox("Internet Service Infrastructure", ["Fiber optic", "DSL", "No"],
                                      index=["Fiber optic", "DSL", "No"].index(d_internet))
            
            sc1, sc2 = st.columns(2)
            with sc1:
                c_security = st.selectbox("Online Security Add-on", ["No", "Yes"], index=0 if d_sec == "No" else 1)
                c_backup = st.selectbox("Online Cloud Backup", ["No", "Yes"], index=0 if d_backup == "No" else 1)
                c_device = st.selectbox("Device Protection", ["No", "Yes"], index=0 if d_device == "No" else 1)
            with sc2:
                c_tech = st.selectbox("Premium Tech Support", ["No", "Yes"], index=0 if d_supp == "No" else 1)
                c_tv = st.selectbox("Streaming TV Service", ["No", "Yes"], index=0 if d_tv == "No" else 1)
                c_movies = st.selectbox("Streaming Movie Service", ["No", "Yes"], index=0 if d_mov == "No" else 1)

            st.markdown("##### 4. Monthly & Historical Charges")
            c_monthly = st.number_input("Monthly Charges ($)", min_value=15.0, max_value=150.0, value=float(d_monthly), step=0.50)
            
            calc_default = round(c_tenure * c_monthly, 2)
            c_total = st.number_input("Cumulative Total Charges ($)", min_value=0.0, max_value=10000.0, value=float(calc_default if d_total == 0 else d_total), step=10.0)

        submit_btn = st.form_submit_button("Predict Churn Risk", type="primary", use_container_width=True)

    # Prediction Output
    if submit_btn:
        customer_dict = {
            'gender': c_gender,
            'SeniorCitizen': senior_val,
            'Partner': c_partner,
            'Dependents': c_dependents,
            'tenure': c_tenure,
            'PhoneService': c_phone,
            'MultipleLines': c_lines,
            'InternetService': c_internet,
            'OnlineSecurity': c_security,
            'OnlineBackup': c_backup,
            'DeviceProtection': c_device,
            'TechSupport': c_tech,
            'StreamingTV': c_tv,
            'StreamingMovies': c_movies,
            'Contract': c_contract,
            'PaperlessBilling': c_paperless,
            'PaymentMethod': c_payment,
            'MonthlyCharges': c_monthly,
            'TotalCharges': c_total
        }

        # Run actual prediction
        result = predict_single_customer(
            customer_dict=customer_dict,
            pipeline=pipeline,
            decision_threshold=decision_threshold,
            low_risk_thresh=low_risk_thresh,
            high_risk_thresh=high_risk_thresh
        )

        prob = result['churn_probability']
        pct = result['churn_percentage']
        prediction_label = result['prediction']
        risk_level = result['risk_level']
        recommendations = result['recommendations']

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### Prediction Assessment & Risk Analysis")

        # Visual Scoreboard
        res_col1, res_col2 = st.columns([1.2, 1.8])

        with res_col1:
            # Color based on risk level
            if risk_level == "High":
                badge_bg = "#FEF2F2"
                badge_border = "#EF4444"
                badge_text = "#DC2626"
            elif risk_level == "Medium":
                badge_bg = "#FFFBEB"
                badge_border = "#F59E0B"
                badge_text = "#D97706"
            else:
                badge_bg = "#ECFDF5"
                badge_border = "#10B981"
                badge_text = "#059669"

            st.markdown(f"""
                <div style="background-color: {badge_bg}; border: 1.5px solid {badge_border}; border-radius: 12px; padding: 20px; text-align: center; margin-bottom: 16px;">
                    <span style="font-size: 13px; font-weight: 700; color: #64748B; text-transform: uppercase; letter-spacing: 0.5px;">Customer Retention Outlook</span>
                    <h2 style="color: {badge_text}; margin: 8px 0; font-size: 26px;">{prediction_label}</h2>
                    <div style="font-size: 15px; font-weight: 600; color: {badge_text};">
                        Risk Level: <strong>{risk_level} Risk</strong>
                    </div>
                    <div style="font-size: 12px; color: #64748B; margin-top: 6px;">
                        Decision Threshold: {decision_threshold:.2f} | Churn Probability: {pct:.1f}%
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # Plotly Gauge Chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pct,
                number={'suffix': "%", 'font': {'size': 32, 'color': '#0F172A'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#CBD5E1"},
                    'bar': {'color': "#1E293B", 'thickness': 0.25},
                    'bgcolor': "white",
                    'borderwidth': 1,
                    'bordercolor': "#E2E8F0",
                    'steps': [
                        {'range': [0, low_risk_thresh * 100], 'color': '#D1FAE5'},
                        {'range': [low_risk_thresh * 100, high_risk_thresh * 100], 'color': '#FEF3C7'},
                        {'range': [high_risk_thresh * 100, 100], 'color': '#FEE2E2'}
                    ],
                    'threshold': {
                        'line': {'color': "#DC2626", 'width': 3},
                        'thickness': 0.75,
                        'value': decision_threshold * 100
                    }
                }
            ))
            fig_gauge.update_layout(height=230, margin=dict(t=10, b=10, l=20, r=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with res_col2:
            st.markdown("##### Model Attribution & Contributing Factors")

            # Compute local SHAP explanation
            try:
                # Transform vector through pipeline steps up to classifier
                preproc_steps = pipeline.named_steps
                cleaner = preproc_steps.get('cleaner')
                features = preproc_steps.get('features')
                preprocessor = preproc_steps.get('preprocessor')
                classifier = preproc_steps.get('classifier')

                raw_df = pd.DataFrame([customer_dict])
                cleaned_df = cleaner.transform(raw_df) if cleaner else raw_df
                fe_df = features.transform(cleaned_df) if features else cleaned_df
                transformed_vector = preprocessor.transform(fe_df)
                feature_names = preprocessor.get_feature_names_out().tolist()
                clean_names = [f.replace('cat__', '').replace('num__', '') for f in feature_names]

                local_explanation = explain_single_customer(
                    model=classifier,
                    customer_vector=transformed_vector,
                    feature_names=clean_names,
                    original_customer_data=customer_dict
                )

                # Narrative explanation
                st.info(local_explanation['narrative'])

                # Top Factors Chart
                factors = local_explanation['factor_df'].head(8)
                factors['Color'] = factors['SHAP_Value'].apply(lambda v: '#EF4444' if v > 0 else '#10B981')

                fig_factors = go.Figure(go.Bar(
                    x=factors['SHAP_Value'],
                    y=factors['Feature'],
                    orientation='h',
                    marker_color=factors['Color'],
                    text=factors['SHAP_Value'].apply(lambda v: f"{v:+.3f}"),
                    textposition='outside'
                ))
                fig_factors.update_layout(
                    title="Key Local Drivers (Positive = Pushes Churn, Negative = Pushes Retention)",
                    xaxis_title="SHAP Attribution Value",
                    yaxis_title="",
                    height=280,
                    margin=dict(t=30, b=20, l=10, r=20)
                )
                st.plotly_chart(fig_factors, use_container_width=True)

            except Exception as e:
                st.info(f"Summary Insight: This customer exhibits {risk_level} churn risk driven by their {c_contract} commitment and payment settings.")

        # Recommendations Box
        st.markdown("---")
        st.markdown("#### Recommended Retention Strategies")
        if recommendations:
            for rec in recommendations:
                priority_color = "#DC2626" if rec['priority'] == 'High' else "#D97706" if rec['priority'] == 'Medium' else "#2563EB"
                st.markdown(f"""
                    <div style="background-color: #F8FAFC; border-left: 4px solid {priority_color}; padding: 12px 16px; border-radius: 6px; margin-bottom: 10px;">
                        <span style="background-color: {priority_color}; color: white; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 700; text-transform: uppercase;">{rec['priority']} Priority</span>
                        <strong style="margin-left: 8px; color: #0F172A; font-size: 14px;">{rec['strategy']}</strong>
                        <p style="margin: 6px 0 0 0; color: #475569; font-size: 13.5px;">{rec['action']}</p>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.success("No critical interventions needed. Customer profile shows strong brand loyalty and low churn propensity.")
