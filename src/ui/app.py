import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
import sys
import os
import contextlib
import io

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.ml.predict import predict_risk
from src.ml.explain import explain_applicant
from src.talk_to_data.talk_to_data import TalkToData

import zipfile

@st.cache_resource
def initialize_deployment():
    # Run once at application startup to initialize required datasets and database for Streamlit Community Cloud.
    csv_path = Path("models/application_features.csv")
    zip_path = Path("models/application_features.zip")
    
    # Extract CSV if missing
    if not csv_path.exists() and zip_path.exists():
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for file_info in zip_ref.infolist():
                if "application_features.csv" in file_info.filename:
                    file_info.filename = "application_features.csv"
                    zip_ref.extract(file_info, "models/")
                    break
                    
    # Initialize DuckDB if missing
    db_path = Path("data/credit_risk.duckdb")
    if not db_path.exists() and csv_path.exists():
        try:
            from src.talk_to_data.query_runner import initialize_database
            initialize_database()
        except Exception as e:
            print(f"Database initialization failed: {e}")

# Run initialization
initialize_deployment()

# ==========================================
# CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="Credit Risk Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a premium, professional aesthetic
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Background and global styles */
    .stApp {
        background-color: #f8fafc;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
        box-shadow: 2px 0 10px rgba(0,0,0,0.02);
    }
    
    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        color: #0f172a;
        font-weight: 600;
        letter-spacing: -0.02em;
    }
    
    /* Premium Metric Card (Glassmorphism & Shadow) */
    .premium-card {
        background: linear-gradient(145deg, #ffffff, #f8fafc);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        border: 1px solid #e2e8f0;
        transition: all 0.3s ease;
        margin-bottom: 1rem;
    }
    
    .premium-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 20px -5px rgba(0, 0, 0, 0.08), 0 8px 10px -5px rgba(0, 0, 0, 0.04);
        border-color: #cbd5e1;
    }
    
    .premium-card-title {
        color: #64748b;
        font-size: 0.875rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 8px;
    }
    
    .premium-card-value {
        color: #0f172a;
        font-size: 2rem;
        font-weight: 600;
        line-height: 1.2;
    }
    
    .premium-card-subtitle {
        color: #64748b;
        font-size: 0.875rem;
        margin-top: 8px;
    }
    
    /* Risk Band specific styling */
    .risk-low-card { border-top: 4px solid #10b981; }
    .risk-medium-card { border-top: 4px solid #f59e0b; }
    .risk-high-card { border-top: 4px solid #ef4444; }
    
    .text-low { color: #10b981; font-weight: 600; }
    .text-medium { color: #f59e0b; font-weight: 600; }
    .text-high { color: #ef4444; font-weight: 600; }
    
    /* Buttons */
    .stButton>button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s ease;
        background-color: #ffffff;
        border: 1px solid #cbd5e1;
    }
    .stButton>button:hover {
        border-color: #94a3b8;
        background-color: #f1f5f9;
        transform: translateY(-1px);
    }
    
    /* Primary Button override */
    .stButton>button[kind="primary"] {
        background-color: #2563eb;
        color: white;
        border: none;
    }
    .stButton>button[kind="primary"]:hover {
        background-color: #1d4ed8;
        transform: translateY(-1px);
        box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.2);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: transparent !important;
        border-bottom: 1px solid #e2e8f0;
        font-weight: 500;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

# Helper function for rendering premium cards
def render_metric_card(title, value, risk_class="", subtitle=None):
    subtitle_html = f'<div class="premium-card-subtitle">{subtitle}</div>' if subtitle else ""
    st.markdown(f"""
    <div class="premium-card {risk_class}">
        <div class="premium-card-title">{title}</div>
        <div class="premium-card-value">{value}</div>
        {subtitle_html}
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# DATA LOADING
# ==========================================
@st.cache_data
def load_applicant_data():
    try:
        return pd.read_csv("models/application_features.csv")
    except Exception as e:
        return None

@st.cache_data
def load_risk_rules():
    try:
        return pd.read_csv("documents/rules/business_risk_rules.csv")
    except:
        return None

@st.cache_resource
def get_talk_to_data():
    try:
        return TalkToData()
    except:
        return None

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.markdown("""
<div style="text-align: center; margin-bottom: 20px;">
    <h1 style="color: #1e293b; font-size: 1.5rem; font-weight: 700; margin: 0;">CREDIT RISK</h1>
    <h2 style="color: #3b82f6; font-size: 1.2rem; font-weight: 600; margin: 0; letter-spacing: 2px;">INTELLIGENCE</h2>
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown("---")

nav_options = [
    "Overview",
    "EDA & Portfolio Insights",
    "Risk Prediction",
    "Explainability",
    "Business Rules",
    "Talk to Data"
]
selection = st.sidebar.radio("Navigation", nav_options)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="font-size: 0.9rem; color: #475569;">
    <strong>Dataset</strong><br>
    Home Credit Default Risk<br><br>
    <strong>Applicants</strong><br>
    307,511<br><br>
    <strong>Default Rate</strong><br>
    8.07%
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="font-size: 0.9rem; color: #475569;">
    <strong>Model</strong><br>
    LightGBM<br><br>
    <strong>ROC-AUC</strong><br>
    0.7762
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
# Check statuses
dataset_status = "🟢 Connected" if Path("models/application_features.csv").exists() else "🔴 Missing"
model_status = "🟢 Loaded" if Path("models/credit_risk_lightgbm.pkl").exists() else "🔴 Missing"
db_status = "🟢 Connected" if Path("data/credit_risk.duckdb").exists() else "🔴 Missing"

st.sidebar.markdown(f"""
<div style="font-size: 0.9rem; color: #475569;">
    <strong>Status</strong><br>
    Data: {dataset_status}<br>
    Model: {model_status}<br>
    Database: {db_status}
</div>
""", unsafe_allow_html=True)


# ==========================================
# PAGE 1: OVERVIEW
# ==========================================
if selection == "Overview":
    st.markdown("""
    <h1 style='margin-bottom: 0.5rem;'>Credit Risk Intelligence Platform</h1>
    <p style='color: #64748b; font-size: 1.1rem; max-width: 800px; margin-bottom: 2rem;'>
        An explainable AI platform for assessing historical credit risk patterns, predicting default probability, understanding model decisions, and exploring credit data using natural language.
    </p>
    """, unsafe_allow_html=True)
    
    st.markdown("<h3 style='margin-top: 2rem; margin-bottom: 1rem;'>Portfolio Metrics</h3>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric_card("Total Applicants", "307,511")
    with col2:
        render_metric_card("Historical Default Rate", "8.07%")
    with col3:
        render_metric_card("Model ROC-AUC", "0.7762")
    with col4:
        render_metric_card("High-Risk Applicants", "10,914")
        
    st.markdown("<h3 style='margin-top: 2rem; margin-bottom: 1rem;'>Risk Band Summary</h3>", unsafe_allow_html=True)
    rcol1, rcol2, rcol3 = st.columns(3)
    with rcol1:
        render_metric_card("Low Risk", "2.44%", "risk-low-card", "Applicants: 28,085")
    with rcol2:
        render_metric_card("Medium Risk", "7.61%", "risk-medium-card", "Applicants: 22,504")
    with rcol3:
        render_metric_card("High Risk", "23.51%", "risk-high-card", "Applicants: 10,914")
        
    st.markdown("<h3 style='margin-top: 2rem; margin-bottom: 1rem;'>Model Performance Summary</h3>", unsafe_allow_html=True)
    
    perf1, perf2, perf3, perf4 = st.columns(4)
    with perf1:
        render_metric_card("ROC-AUC", "0.7762")
    with perf2:
        render_metric_card("Avg Precision", "0.2727")
    with perf3:
        render_metric_card("Recall @ 0.50", "65.48%")
    with perf4:
        render_metric_card("F1 @ 0.50", "29.69%")
        
    st.info("*Because default is relatively rare, threshold selection involves a trade-off between identifying more defaults and reducing false positives.*")
    
    st.markdown("<h3 style='margin-top: 2rem; margin-bottom: 1rem;'>Platform Capabilities</h3>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px;">
        <div class="premium-card" style="border-left: 4px solid #3b82f6;">
            <div style="font-weight: 600; color: #1e293b; margin-bottom: 8px;">📊 EDA</div>
            <div style="font-size: 0.9rem; color: #64748b;">Explore portfolio patterns and historical trends.</div>
        </div>
        <div class="premium-card" style="border-left: 4px solid #8b5cf6;">
            <div style="font-weight: 600; color: #1e293b; margin-bottom: 8px;">🎯 Risk Prediction</div>
            <div style="font-size: 0.9rem; color: #64748b;">Estimate applicant default probability using LightGBM.</div>
        </div>
        <div class="premium-card" style="border-left: 4px solid #ec4899;">
            <div style="font-weight: 600; color: #1e293b; margin-bottom: 8px;">🔍 Explainability</div>
            <div style="font-size: 0.9rem; color: #64748b;">Understand model drivers using SHAP values.</div>
        </div>
        <div class="premium-card" style="border-left: 4px solid #f59e0b;">
            <div style="font-weight: 600; color: #1e293b; margin-bottom: 8px;">📋 Business Rules</div>
            <div style="font-size: 0.9rem; color: #64748b;">Review interpretable, data-backed risk signals.</div>
        </div>
        <div class="premium-card" style="border-left: 4px solid #10b981;">
            <div style="font-weight: 600; color: #1e293b; margin-bottom: 8px;">💬 Talk to Data</div>
            <div style="font-size: 0.9rem; color: #64748b;">Ask questions using natural language.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# PAGE 2: EDA & PORTFOLIO INSIGHTS
# ==========================================
elif selection == "EDA & Portfolio Insights":
    st.markdown("<h1>EDA & Portfolio Insights</h1>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        render_metric_card("Total Applicants", "307,511")
    with col2:
        render_metric_card("Defaults", "24,825")
    with col3:
        render_metric_card("Default Rate", "8.07%")
    
    st.markdown("<h3 style='margin-top: 2rem; margin-bottom: 1rem;'>Key Business Insights</h3>", unsafe_allow_html=True)
    st.info("""
    1. **Younger applicants** show higher observed default rates.
    2. **External credit scores** show strong relationships with observed default rates.
    3. **Longer employment history** is associated with lower observed default rates.
    4. **Education categories** show different observed default rates.
    5. The target is highly imbalanced, with approximately **8.07% defaults**.
    
    *Note: These insights reflect observed historical patterns and do not claim causality.*
    """)
    
    st.markdown("<h3 style='margin-top: 2rem; margin-bottom: 1rem;'>Visualizations</h3>", unsafe_allow_html=True)
    
    eda_dir = Path("documents/eda")
    if eda_dir.exists():
        charts = {
            "Target Distribution": "01_target_distribution.png",
            "Default by Age": "02_default_by_age.png",
            "Default by External Credit Score 1": "05_ext_source_1_default.png",
            "Default by Employment Duration": "06_default_by_employment.png",
            "Default by Education": "07_default_by_education.png",
            "Credit-Income Relationship": "10_default_by_credit_income_ratio.png"
        }
        
        tab_names = list(charts.keys())
        tabs = st.tabs(tab_names)
        
        for tab, (name, filename) in zip(tabs, charts.items()):
            with tab:
                img_path = eda_dir / filename
                if img_path.exists():
                    # Display the image in a nice container with shadow
                    st.markdown("""
                    <div style="background: white; border-radius: 12px; padding: 20px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); border: 1px solid #e2e8f0; margin-top: 10px;">
                    """, unsafe_allow_html=True)
                    st.image(Image.open(img_path), use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.warning(f"Chart '{filename}' not found.")
    else:
        st.warning("EDA charts directory not found.")


# ==========================================
# PAGE 3: RISK PREDICTION
# ==========================================
elif selection == "Risk Prediction":
    st.markdown("<h1>Risk Prediction</h1>", unsafe_allow_html=True)
    
    df = load_applicant_data()
    if df is not None and not df.empty:
        applicant_ids = df["SK_ID_CURR"].tolist()
        
        col1, col2 = st.columns([1, 2], gap="large")
        with col1:
            st.markdown("<div class='premium-card'>", unsafe_allow_html=True)
            st.markdown("<h3 style='margin-top: 0;'>Select Applicant</h3>", unsafe_allow_html=True)
            selected_id = st.selectbox("Applicant ID", applicant_ids, index=0)
            
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Calculate Risk Prediction", type="primary", use_container_width=True):
                with st.spinner("Calculating risk prediction..."):
                    applicant_data = df[df["SK_ID_CURR"] == selected_id].iloc[0]
                    try:
                        result = predict_risk(applicant_data)
                        
                        st.session_state["current_prediction"] = result
                        st.session_state["current_applicant"] = applicant_data
                    except Exception as e:
                        st.error(f"Prediction failed: {str(e)}")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            if "current_prediction" in st.session_state:
                result = st.session_state["current_prediction"]
                applicant_data = st.session_state["current_applicant"]
                
                if applicant_data["SK_ID_CURR"] == selected_id:
                    st.markdown("<h3 style='margin-top: 0;'>Prediction Output</h3>", unsafe_allow_html=True)
                    
                    risk_band = result['risk_band']
                    prob = result['default_probability'] * 100
                    score = result['risk_score']
                    
                    if risk_band == "Low":
                        band_class = "risk-low-card"
                        text_class = "text-low"
                    elif risk_band == "Medium":
                        band_class = "risk-medium-card"
                        text_class = "text-medium"
                    else:
                        band_class = "risk-high-card"
                        text_class = "text-high"
                    
                    st.markdown(f"""
                    <div class="premium-card {band_class}">
                        <div style="font-size: 1.1rem; color: #64748b; margin-bottom: 5px;">Risk Band</div>
                        <div class="{text_class}" style="font-size: 2.5rem; line-height: 1.1; margin-bottom: 20px;">{risk_band.upper()}</div>
                        <div style="display: flex; gap: 40px;">
                            <div>
                                <div style="font-size: 0.9rem; color: #64748b;">Default Probability</div>
                                <div style="font-size: 1.5rem; font-weight: 600; color: #0f172a;">{prob:.2f}%</div>
                            </div>
                            <div>
                                <div style="font-size: 0.9rem; color: #64748b;">Risk Score</div>
                                <div style="font-size: 1.5rem; font-weight: 600; color: #0f172a;">{score:.2f} <span style="font-size: 1rem; color: #94a3b8;">/ 100</span></div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("""
                    <div style="background-color: #f1f5f9; padding: 15px; border-radius: 8px; margin-top: 15px; border-left: 4px solid #cbd5e1;">
                        <strong>Risk Explanation:</strong><br>
                        <span style="color: #64748b; font-size: 0.9rem;">
                        • <strong>Low:</strong> Probability below 30%<br>
                        • <strong>Medium:</strong> Probability 30%–59.99%<br>
                        • <strong>High:</strong> Probability 60% or higher<br><br>
                        <em>*This prediction is a model-generated risk estimate, not a guarantee of default or repayment.*</em>
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    with st.expander("View Applicant Profile"):
                        display_cols = ['AGE_YEARS', 'CREDIT_INCOME_RATIO', 'EMPLOYMENT_YEARS', 
                                        'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 
                                        'LATE_PAYMENT_RATE', 'PREV_APPLICATION_COUNT']
                        
                        prof_data = {}
                        for col in display_cols:
                            if col in applicant_data:
                                val = applicant_data[col]
                                prof_data[col] = val
                        
                        st.dataframe(pd.Series(prof_data, name="Value").to_frame(), use_container_width=True)
    else:
        st.error("Applicant dataset could not be loaded.")


# ==========================================
# PAGE 4: EXPLAINABILITY
# ==========================================
elif selection == "Explainability":
    st.markdown("<h1>Explainability (SHAP)</h1>", unsafe_allow_html=True)
    
    df = load_applicant_data()
    if df is not None and not df.empty:
        applicant_ids = df["SK_ID_CURR"].tolist()
        
        st.markdown("<div class='premium-card' style='max-width: 500px;'>", unsafe_allow_html=True)
        st.markdown("<h3 style='margin-top: 0;'>Select Applicant</h3>", unsafe_allow_html=True)
        selected_id = st.selectbox("Applicant ID", applicant_ids, index=0, key="shap_applicant", label_visibility="collapsed")
        
        if st.button("Generate Explanation", type="primary"):
            with st.spinner("Generating explanation..."):
                try:
                    # We capture the stdout of explain_applicant if needed, but it saves to disk
                    with contextlib.redirect_stdout(io.StringIO()):
                        explain_applicant(selected_id)
                    
                    st.session_state["shap_generated_for"] = selected_id
                except Exception as e:
                    st.error(f"Failed to generate SHAP explanation: {str(e)}")
        st.markdown("</div>", unsafe_allow_html=True)
                    
        if st.session_state.get("shap_generated_for") == selected_id:
            shap_dir = Path("documents/shap")
            csv_path = shap_dir / f"applicant_{selected_id}_shap.csv"
            img_path = shap_dir / f"applicant_{selected_id}_shap_bar.png"
            
            col1, col2 = st.columns([1.5, 1])
            
            with col1:
                if img_path.exists():
                    st.markdown("""
                    <div style="background: white; border-radius: 12px; padding: 20px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); border: 1px solid #e2e8f0; margin-top: 10px;">
                    """, unsafe_allow_html=True)
                    st.image(Image.open(img_path), caption=f"Top SHAP Features for Applicant {selected_id}", use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                if csv_path.exists():
                    st.markdown("<h3 style='margin-top: 10px;'>Top SHAP Contributors</h3>", unsafe_allow_html=True)
                    shap_df = pd.read_csv(csv_path)
                    
                    display_df = shap_df[['business_feature', 'feature_value', 'shap_value', 'direction']].head(10)
                    display_df.columns = ['Feature', 'Value', 'SHAP Contribution', 'Direction']
                    
                    def highlight_direction(val):
                        color = '#ef4444' if val == 'Increases risk' else '#10b981'
                        return f'color: {color}; font-weight: 500;'
                    
                    st.dataframe(display_df.style.map(highlight_direction, subset=['Direction']), use_container_width=True)
                
            st.markdown("<br>", unsafe_allow_html=True)
            st.info("""
            **IMPORTANT DISCLAIMER:**
            SHAP values show how features influenced this model's prediction for this applicant. They describe model behaviour and should not be interpreted as causal relationships.
            """)
    else:
        st.error("Applicant dataset could not be loaded.")


# ==========================================
# PAGE 5: BUSINESS RULES
# ==========================================
elif selection == "Business Rules":
    st.markdown("<h1>Business Rules</h1>", unsafe_allow_html=True)
    
    st.warning("""
    **IMPORTANT:**
    These are analytical risk signals derived from historical data. They are not automatic approve/reject rules and should not be used as the sole basis for individual lending decisions.
    """)
    
    rules_df = load_risk_rules()
    if rules_df is not None:
        st.markdown("<div class='premium-card' style='padding: 0; overflow: hidden;'>", unsafe_allow_html=True)
        st.dataframe(rules_df, use_container_width=True, height=400)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<h3 style='margin-top: 2rem;'>Rule Categories</h3>", unsafe_allow_html=True)
        
        st.markdown("""
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 16px;">
            <div class="premium-card">
                <div style="font-weight: 600; font-size: 1.1rem; color: #1e293b; margin-bottom: 8px;">💳 External Credit Signals</div>
                <div style="font-size: 0.95rem; color: #64748b;">Strongest relationship with historical default rates. External scores heavily weigh on risk prediction.</div>
            </div>
            <div class="premium-card">
                <div style="font-weight: 600; font-size: 1.1rem; color: #1e293b; margin-bottom: 8px;">💼 Employment</div>
                <div style="font-size: 0.95rem; color: #64748b;">Longer tenure generally correlates with lower default risk and greater financial stability.</div>
            </div>
            <div class="premium-card">
                <div style="font-weight: 600; font-size: 1.1rem; color: #1e293b; margin-bottom: 8px;">📅 Repayment Behaviour</div>
                <div style="font-size: 0.95rem; color: #64748b;">Historical late payments indicate elevated future risk. Past behaviour strongly predicts future behaviour.</div>
            </div>
            <div class="premium-card">
                <div style="font-weight: 600; font-size: 1.1rem; color: #1e293b; margin-bottom: 8px;">👤 Age</div>
                <div style="font-size: 0.95rem; color: #64748b;">Younger demographics historically show different, often higher, risk profiles.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("Business rules file not found. Ensure `documents/rules/business_risk_rules.csv` exists.")


# ==========================================
# PAGE 6: TALK TO DATA
# ==========================================
elif selection == "Talk to Data":
    st.markdown("<h1>Talk to Your Credit Data</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #64748b; font-size: 1.1rem;'>Ask questions about the credit portfolio in plain English.</p>", unsafe_allow_html=True)
    
    st.markdown("""
    <style>
        .chat-container {
            background-color: white;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
            border: 1px solid #e2e8f0;
            margin-bottom: 15px;
        }
        
        .stChatMessage {
            background-color: transparent !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
        
    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("details"):
                with st.expander("View Execution Details"):
                    st.markdown(f"**Generated SQL:**\n```sql\n{msg['details'].get('sql')}\n```")
                    st.markdown(f"**Validation:** {msg['details'].get('validation_msg')}")
                    if msg['details'].get('result') is not None:
                        st.markdown("**Database Result:**")
                        st.dataframe(msg['details'].get('result'))
            
    prompt = st.chat_input("Ask a question (e.g., 'What is the default rate by education type?')")
    
    if prompt:
        st.session_state["chat_history"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("Generating and validating SQL..."):
                ttd = get_talk_to_data()
                if ttd is None:
                    st.error("Talk-to-Data is temporarily unavailable.")
                else:
                    try:
                        response = ttd.ask(prompt)
                        st.markdown(response["answer"])
                        
                        details = {
                            "sql": response.get("sql", "N/A"),
                            "validation_msg": response.get("validation", {}).get("message", "N/A"),
                            "result": response.get("result")
                        }
                        
                        with st.expander("View Execution Details"):
                            st.markdown(f"**Generated SQL:**\n```sql\n{details['sql']}\n```")
                            st.markdown(f"**Validation:** {details['validation_msg']}")
                            if details['result'] is not None:
                                st.markdown("**Database Result:**")
                                st.dataframe(details['result'])
                                
                        st.session_state["chat_history"].append({
                            "role": "assistant", 
                            "content": response["answer"],
                            "details": details
                        })
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")

# ==========================================
# FOOTER
# ==========================================
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="font-size: 0.8rem; color: #94a3b8; text-align: center;">
    <strong>AI-Powered Credit Risk Intelligence Platform</strong><br>
    Built for analytical decision support. Model predictions and historical patterns should be interpreted alongside appropriate business and regulatory review.
</div>
""", unsafe_allow_html=True)
