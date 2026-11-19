"""
Page 2: Customer Analytics & Exploratory Data Analysis.
Provides interactive business analytics and exploratory visualizations
using Plotly for the IBM Telco Customer Churn dataset.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def render_analytics_page(df: pd.DataFrame):
    """Renders the Customer Analytics dashboard with interactive filtering and Plotly charts."""
    st.markdown("""
        <div style="margin-bottom: 24px;">
            <h2 style="color: #0F172A; font-weight: 700; margin-bottom: 6px;">Customer Retention Analytics</h2>
            <p style="color: #64748B; font-size: 15px;">
                Explore behavioral trends, demographic distributions, and service friction points driving customer churn.
            </p>
        </div>
    """, unsafe_allow_html=True)

    if df is None or df.empty:
        st.warning("Customer dataset is not loaded. Please ensure the Telco dataset is present.")
        return

    # Standardize Churn into 'Yes' and 'No' for human-readable dashboard analytics
    df = df.copy()
    is_churn_global = (df['Churn'] == 'Yes') | (df['Churn'] == 1) | (df['Churn'] == '1')
    df['Churn'] = np.where(is_churn_global, 'Yes', 'No')
    df['Churn_Numeric'] = is_churn_global.astype(int)

    # Filter Sidebar / Controls
    with st.expander("Filter Customer Cohorts", expanded=True):
        col_f1, col_f2, col_f3, col_f4 = st.columns(4)

        with col_f1:
            contract_options = ["All"] + sorted(list(df['Contract'].dropna().unique()))
            selected_contract = st.selectbox("Contract Type", contract_options)

        with col_f2:
            internet_options = ["All"] + sorted(list(df['InternetService'].dropna().unique()))
            selected_internet = st.selectbox("Internet Service", internet_options)

        with col_f3:
            payment_options = ["All"] + sorted(list(df['PaymentMethod'].dropna().unique()))
            selected_payment = st.selectbox("Payment Method", payment_options)

        with col_f4:
            senior_options = ["All", "Senior Citizen (Yes)", "Non-Senior (No)"]
            selected_senior = st.selectbox("Senior Citizen", senior_options)

    # Apply filters
    filtered_df = df.copy()
    if selected_contract != "All":
        filtered_df = filtered_df[filtered_df['Contract'] == selected_contract]
    if selected_internet != "All":
        filtered_df = filtered_df[filtered_df['InternetService'] == selected_internet]
    if selected_payment != "All":
        filtered_df = filtered_df[filtered_df['PaymentMethod'] == selected_payment]
    if selected_senior == "Senior Citizen (Yes)":
        filtered_df = filtered_df[filtered_df['SeniorCitizen'].astype(str).str.strip().isin(['1', '1.0', 'Yes', 'yes'])]
    elif selected_senior == "Non-Senior (No)":
        filtered_df = filtered_df[filtered_df['SeniorCitizen'].astype(str).str.strip().isin(['0', '0.0', 'No', 'no'])]

    # Filtered KPI summary (strictly dynamic calculations)
    total_cohort = len(filtered_df)
    is_churn_cohort = filtered_df['Churn'] == 'Yes'
    churn_count = int(is_churn_cohort.sum())
    retained_count = int((~is_churn_cohort).sum())
    cohort_churn_rate = (churn_count / total_cohort * 100) if total_cohort > 0 else 0.0
    cohort_retained_rate = (retained_count / total_cohort * 100) if total_cohort > 0 else 0.0

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Filtered Cohort", f"{total_cohort:,}")
    with c2:
        st.metric("Retained Customers", f"{retained_count:,}", f"{cohort_retained_rate:.1f}% of cohort")
    with c3:
        st.metric("Churned Customers", f"{churn_count:,}", f"{cohort_churn_rate:.1f}% of cohort", delta_color="inverse")
    with c4:
        st.metric("Cohort Churn Rate", f"{cohort_churn_rate:.1f}%")

    st.markdown("---")

    # Row 1: Churn Distribution & Churn by Contract
    row1_col1, row1_col2 = st.columns(2)

    with row1_col1:
        st.markdown("#### Overall Churn Ratio")
        churn_counts = filtered_df['Churn'].value_counts().reset_index()
        churn_counts.columns = ['Status', 'Count']
        fig_donut = px.pie(
            churn_counts,
            values='Count',
            names='Status',
            hole=0.55,
            color='Status',
            color_discrete_map={'No': '#1E3A8A', 'Yes': '#EF4444'},
        )
        fig_donut.update_traces(
            textposition='inside',
            textinfo='percent+label',
            marker=dict(line=dict(color='#FFFFFF', width=2))
        )
        fig_donut.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=320, showlegend=False)
        st.plotly_chart(fig_donut, use_container_width=True)

    with row1_col2:
        st.markdown("#### Churn Rate by Contract Commitment")
        contract_summary = filtered_df.groupby('Contract')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100).reset_index()
        contract_summary.columns = ['Contract', 'Churn_Rate']
        fig_contract = px.bar(
            contract_summary,
            x='Contract',
            y='Churn_Rate',
            color='Contract',
            color_discrete_sequence=['#3B82F6', '#6366F1', '#8B5CF6'],
            text=contract_summary['Churn_Rate'].apply(lambda v: f"{v:.1f}%")
        )
        fig_contract.update_layout(
            yaxis_title="Churn Rate (%)",
            xaxis_title="",
            margin=dict(t=20, b=20, l=20, r=20),
            height=320,
            showlegend=False
        )
        st.plotly_chart(fig_contract, use_container_width=True)

    # Row 2: Internet Service & Payment Method Analysis
    row2_col1, row2_col2 = st.columns(2)

    with row2_col1:
        st.markdown("#### Churn Rate by Internet Service")
        internet_summary = filtered_df.groupby('InternetService')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100).reset_index()
        internet_summary.columns = ['InternetService', 'Churn_Rate']
        fig_internet = px.bar(
            internet_summary,
            x='InternetService',
            y='Churn_Rate',
            color='InternetService',
            color_discrete_sequence=['#0284C7', '#EF4444', '#10B981'],
            text=internet_summary['Churn_Rate'].apply(lambda v: f"{v:.1f}%")
        )
        fig_internet.update_layout(
            yaxis_title="Churn Rate (%)",
            xaxis_title="",
            margin=dict(t=20, b=20, l=20, r=20),
            height=320,
            showlegend=False
        )
        st.plotly_chart(fig_internet, use_container_width=True)

    with row2_col2:
        st.markdown("#### Churn Rate by Payment Method")
        pay_summary = filtered_df.groupby('PaymentMethod')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100).reset_index()
        pay_summary.columns = ['PaymentMethod', 'Churn_Rate']
        pay_summary = pay_summary.sort_values(by='Churn_Rate', ascending=True)
        fig_pay = px.bar(
            pay_summary,
            y='PaymentMethod',
            x='Churn_Rate',
            orientation='h',
            color='Churn_Rate',
            color_continuous_scale='Reds',
            text=pay_summary['Churn_Rate'].apply(lambda v: f"{v:.1f}%")
        )
        fig_pay.update_layout(
            xaxis_title="Churn Rate (%)",
            yaxis_title="",
            margin=dict(t=20, b=20, l=20, r=20),
            height=320,
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_pay, use_container_width=True)

    # Row 3: Tenure Density & Monthly Charges
    row3_col1, row3_col2 = st.columns(2)

    with row3_col1:
        st.markdown("#### Tenure Distribution: Retained vs Churned")
        fig_tenure = px.histogram(
            filtered_df,
            x='tenure',
            color='Churn',
            barmode='overlay',
            nbins=36,
            color_discrete_map={'No': '#1E3A8A', 'Yes': '#EF4444'},
            opacity=0.65
        )
        fig_tenure.update_layout(
            xaxis_title="Tenure (Months)",
            yaxis_title="Customer Count",
            margin=dict(t=20, b=20, l=20, r=20),
            height=320,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_tenure, use_container_width=True)

    with row3_col2:
        st.markdown("#### Monthly Charges Distribution")
        fig_charges = px.box(
            filtered_df,
            x='Churn',
            y='MonthlyCharges',
            color='Churn',
            color_discrete_map={'No': '#1E3A8A', 'Yes': '#EF4444'},
            points="outliers"
        )
        fig_charges.update_layout(
            yaxis_title="Monthly Charges ($)",
            xaxis_title="Churn Status",
            margin=dict(t=20, b=20, l=20, r=20),
            height=320,
            showlegend=False
        )
        st.plotly_chart(fig_charges, use_container_width=True)

    # Row 4: Customer Segmentation Scatter Analysis
    st.markdown("#### Customer Segmentation: Tenure vs Monthly Spend")
    sample_size = min(800, len(filtered_df))
    sample_df = filtered_df.sample(sample_size, random_state=42) if len(filtered_df) > sample_size else filtered_df
    fig_scatter = px.scatter(
        sample_df,
        x='tenure',
        y='MonthlyCharges',
        color='Churn',
        color_discrete_map={'No': '#2563EB', 'Yes': '#DC2626'},
        hover_data=['Contract', 'PaymentMethod', 'InternetService'],
        opacity=0.75,
        title="Sampled Cohort Distribution (Tenure vs Monthly Charges)"
    )
    fig_scatter.update_layout(
        xaxis_title="Tenure (Months)",
        yaxis_title="Monthly Charges ($)",
        margin=dict(t=40, b=20, l=20, r=20),
        height=380
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
