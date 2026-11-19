"""
Page 7: Strategic Retention Recommendations & ROI Campaign Simulator.
Provides actionable retention strategies aligned with customer risk tiers
and an interactive campaign ROI modeling engine.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


def render_recommendations_page():
    """Renders the Strategic Customer Retention dashboard and ROI simulator."""
    st.markdown("""
        <div style="margin-bottom: 24px;">
            <h2 style="color: #0F172A; font-weight: 700; margin-bottom: 6px;">Customer Retention Playbook & ROI Simulator</h2>
            <p style="color: #64748B; font-size: 15px;">
                Deploy proven, data-driven retention interventions and simulate campaign economic returns before operational rollout.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Section 1: Strategic Playbook by Segment
    st.markdown("### 1. Retention Intervention Matrix")

    t1, t2, t3 = st.tabs(["High-Risk Interventions", "Medium-Risk Interventions", "Low-Risk Loyalty Tactics"])

    with t1:
        st.markdown("""
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 12px;">
                <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 16px;">
                    <span style="background-color: #EF4444; color: white; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 700;">URGENT</span>
                    <h4 style="margin: 8px 0 4px 0; color: #0F172A;">Contract Migration Incentive</h4>
                    <p style="color: #64748B; font-size: 13.5px; margin-bottom: 8px;"><strong>Target:</strong> Month-to-month subscribers with high churn propensity.</p>
                    <p style="color: #334155; font-size: 13.5px;">Offer a 15% discount for 12 months or a $50 instant bill credit upon switching to a 1-year agreement.</p>
                </div>
                <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 16px;">
                    <span style="background-color: #EF4444; color: white; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 700;">HIGH IMPACT</span>
                    <h4 style="margin: 8px 0 4px 0; color: #0F172A;">Fiber Optic Care & Support Bundle</h4>
                    <p style="color: #64748B; font-size: 13.5px; margin-bottom: 8px;"><strong>Target:</strong> Fiber optic users without tech support or security add-ons.</p>
                    <p style="color: #334155; font-size: 13.5px;">Bundle free 6-month Premium Tech Support and Online Security to safeguard high-margin revenue.</p>
                </div>
                <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 16px;">
                    <span style="background-color: #EF4444; color: white; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 700;">ONBOARDING</span>
                    <h4 style="margin: 8px 0 4px 0; color: #0F172A;">Concierge Welcome Protocol</h4>
                    <p style="color: #64748B; font-size: 13.5px; margin-bottom: 8px;"><strong>Target:</strong> New customers with tenure < 6 months.</p>
                    <p style="color: #334155; font-size: 13.5px;">Execute proactive satisfaction check-ins at 14 and 45 days to eliminate initial friction.</p>
                </div>
                <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 16px;">
                    <span style="background-color: #EF4444; color: white; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 700;">PAYMENT</span>
                    <h4 style="margin: 8px 0 4px 0; color: #0F172A;">Auto-Pay Transition Bonus</h4>
                    <p style="color: #64748B; font-size: 13.5px; margin-bottom: 8px;"><strong>Target:</strong> Electronic check and paper billing customers.</p>
                    <p style="color: #334155; font-size: 13.5px;">Provide a recurring $5/month discount for enrolling in paperless auto-pay via bank or credit card.</p>
                </div>
            </div>
        """, unsafe_allow_html=True)

    with t2:
        st.markdown("""
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 12px;">
                <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 16px;">
                    <span style="background-color: #F59E0B; color: white; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 700;">ENGAGEMENT</span>
                    <h4 style="margin: 8px 0 4px 0; color: #0F172A;">Multi-Product Sticky Bundles</h4>
                    <p style="color: #64748B; font-size: 13.5px; margin-bottom: 8px;"><strong>Target:</strong> Single-service customers (Internet-only or Phone-only).</p>
                    <p style="color: #334155; font-size: 13.5px;">Promote streaming package or cloud backup trial; customers with 4+ services churn 60% less.</p>
                </div>
                <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 16px;">
                    <span style="background-color: #F59E0B; color: white; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 700;">OPTIMIZATION</span>
                    <h4 style="margin: 8px 0 4px 0; color: #0F172A;">Annual Plan Renewal Window</h4>
                    <p style="color: #64748B; font-size: 13.5px; margin-bottom: 8px;"><strong>Target:</strong> 1-year contract customers at month 10–11.</p>
                    <p style="color: #334155; font-size: 13.5px;">Trigger automated early re-contracting offer before natural contract expiration.</p>
                </div>
            </div>
        """, unsafe_allow_html=True)

    with t3:
        st.markdown("""
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 12px;">
                <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 16px;">
                    <span style="background-color: #10B981; color: white; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 700;">LOYALTY</span>
                    <h4 style="margin: 8px 0 4px 0; color: #0F172A;">VIP Anniversary Recognition</h4>
                    <p style="color: #64748B; font-size: 13.5px; margin-bottom: 8px;"><strong>Target:</strong> Loyal subscribers (tenure > 36 months).</p>
                    <p style="color: #334155; font-size: 13.5px;">Grant courtesy broadband speed upgrades or partner streaming perks on annual service anniversaries.</p>
                </div>
                <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 16px;">
                    <span style="background-color: #10B981; color: white; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 700;">ADVOCACY</span>
                    <h4 style="margin: 8px 0 4px 0; color: #0F172A;">Referral & Family Plan Expansion</h4>
                    <p style="color: #64748B; font-size: 13.5px; margin-bottom: 8px;"><strong>Target:</strong> Low-risk multi-line accounts.</p>
                    <p style="color: #334155; font-size: 13.5px;">Provide family member line additions with shared data discounts to deepen brand lock-in.</p>
                </div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Section 2: Interactive Campaign ROI Simulator
    st.markdown("### 2. Retention Campaign ROI Simulator")
    st.markdown("Model financial impact and calculate net economic returns for proposed retention campaigns.")

    sim_c1, sim_c2 = st.columns([1, 1])

    with sim_c1:
        cohort_size = st.slider("Target High-Risk Customer Cohort Size", min_value=100, max_value=5000, value=1000, step=100)
        avg_monthly_rev = st.slider("Average Monthly Spend per Customer ($)", min_value=20.0, max_value=120.0, value=75.0, step=5.0)
        campaign_cost_per_cust = st.slider("Campaign Cost / Incentive per Customer ($)", min_value=5.0, max_value=60.0, value=25.0, step=5.0)
        conversion_rate = st.slider("Expected Retention Intervention Success Rate (%)", min_value=5, max_value=60, value=25, step=5) / 100.0

    # Financial Calculations
    retained_customers = int(cohort_size * conversion_rate)
    total_campaign_cost = cohort_size * campaign_cost_per_cust
    annual_rev_saved = retained_customers * avg_monthly_rev * 12
    net_profit = annual_rev_saved - total_campaign_cost
    roi_pct = (net_profit / total_campaign_cost * 100) if total_campaign_cost > 0 else 0.0

    with sim_c2:
        st.markdown(f"""
            <div style="background-color: #F8FAFC; border: 1.5px solid #CBD5E1; border-radius: 12px; padding: 20px;">
                <h4 style="color: #0F172A; margin-top: 0;">Projected Financial Returns</h4>
                <div style="margin-bottom: 14px;">
                    <span style="font-size: 13px; color: #64748B;">Retained Customers</span>
                    <div style="font-size: 24px; font-weight: 700; color: #1E3A8A;">{retained_customers:,} accounts</div>
                </div>
                <div style="margin-bottom: 14px;">
                    <span style="font-size: 13px; color: #64748B;">Annualized Revenue Preserved (12 Months)</span>
                    <div style="font-size: 24px; font-weight: 700; color: #059669;">${annual_rev_saved:,.0f}</div>
                </div>
                <div style="margin-bottom: 14px;">
                    <span style="font-size: 13px; color: #64748B;">Total Campaign Investment</span>
                    <div style="font-size: 20px; font-weight: 600; color: #DC2626;">${total_campaign_cost:,.0f}</div>
                </div>
                <hr style="margin: 12px 0; border: none; border-top: 1px solid #E2E8F0;">
                <div>
                    <span style="font-size: 13px; color: #64748B;">Net Economic Gain (ROI)</span>
                    <div style="font-size: 26px; font-weight: 800; color: #2563EB;">+${net_profit:,.0f} ({roi_pct:.0f}% ROI)</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
