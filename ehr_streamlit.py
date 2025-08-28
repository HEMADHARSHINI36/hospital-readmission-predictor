# ehr_streamlit_refined_ml_optimized.py

import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
from fpdf import FPDF
from io import BytesIO
import pickle
import os
import re
import altair as alt


# -------------------------
# Load ML model safely
# -------------------------
MODEL_PATH = "xgb_readmission_model.pkl"
ml_model = None

def load_ml_model(model_path):
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        st.sidebar.success("✅ ML model loaded successfully.")
        return model
    else:
        st.sidebar.warning("⚠ ML model not found. Falling back to legacy calculations.")
        return None
        
ml_model = load_ml_model(MODEL_PATH)

# -------------------------
# Load dataset (cached)
# -------------------------
DB_PATH = "ehr_large.db"

@st.cache_data
def load_patients_from_db(db_path):
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql("SELECT * FROM admissions_scored", conn)
    finally:
        conn.close()
    return df

patients_df = load_patients_from_db(DB_PATH)

# -------------------------
# Risk calculation (vectorized)
# -------------------------
patients_df['risk_score'] = (
    0.4 * (patients_df['agefactor'] / 100) +
    0.8 * (patients_df['WBC mean'] / 20000) +
    0.3 * (patients_df['heart rate'] / 200) +
    1.5 * patients_df['diabetes'] +
    1.0 * patients_df['hypertension']
)
patients_df['risk_score'] = 1 / (1 + np.exp(-patients_df['risk_score'])) * 100

def assign_risk_level(risk_score):
    if risk_score >= 75: return "HIGH"
    elif risk_score >= 50: return "MEDIUM"
    else: return "LOW"

def individual_savings(risk_score, cost_per_patient=15000, prevention_success_rate=0.7):
    readmit_prob = risk_score / 100
    return cost_per_patient * readmit_prob * prevention_success_rate

def hospital_impact(df):
    df['expected_saving'] = df['risk_score'].apply(individual_savings)
    total_saving = df['expected_saving'].sum()
    max_penalty_reduction = 26e9 * 0.15
    return min(total_saving, max_penalty_reduction)

patients_df['risk_level'] = patients_df['risk_score'].apply(assign_risk_level)
patients_df['expected_saving'] = patients_df['risk_score'].apply(individual_savings)
patients_df['recommendation'] = patients_df['risk_level'].apply(
    lambda x: "Extended monitoring" if x=="HIGH" else
              "Home care with follow-up" if x=="MEDIUM" else
              "Standard follow-up"
)
overall_impact = hospital_impact(patients_df)

# -------------------------
# ML Readmission Prediction (batch)
# -------------------------
def assign_readmission_flag(prob):
    if prob >= 0.7: return "High Risk"
    elif prob >= 0.4: return "Moderate Risk"
    else: return "Low Risk"

if ml_model:
    features = patients_df[['agefactor','WBC mean','heart rate','diabetes','hypertension']].values
    try:
        preds = ml_model.predict_proba(features)[:,1]
        patients_df['readmit_prob'] = preds
    except:
        # silently fallback to legacy risk
        patients_df['readmit_prob'] = patients_df['risk_score'] / 100
else:
    patients_df['readmit_prob'] = patients_df['risk_score'] / 100


patients_df['readmit_flag'] = patients_df['readmit_prob'].apply(assign_readmission_flag)

# -------------------------
# Generative AI Report
# -------------------------
def generate_structured_report(row):
    report = {}
    
    # Patient Summary with ML readmission probability
    readmit_prob_pct = row.get('readmit_prob', 0) * 100
    readmit_flag = row.get('readmit_flag', 'Low Risk')
    report['Patient Summary'] = (
        f"{row.get('name','Unknown')}, Age {row.get('agefactor',0)}, "
        f"Risk Score: {row.get('risk_score',0):.2f}% ({row.get('risk_level','LOW')}), "
        f"ML Readmission Probability: {readmit_prob_pct:.1f}% ({readmit_flag})"
    )

    # Risk Factors
    factors = []
    if row.get('agefactor',0) > 65: factors.append("Advanced age")
    for c in ['diabetes','hypertension','ckd','copd','cad','stroke','cancer']:
        if row.get(c,0)==1: factors.append(c.capitalize())
    if row.get('WBC mean',0)>11000:
        factors.append(f"High WBC ({row.get('WBC mean',0):,.0f})")
    if row.get('heart rate',0)>100:
        factors.append(f"High HR ({row.get('heart rate',0):.0f} bpm)")
    if row.get('BP-mean',0)>140 or row.get('BP-mean',0)<90:
        factors.append(f"Abnormal BP ({row.get('BP-mean',0):.0f} mmHg)")
    factors.append(f"Readmission Risk: {readmit_flag}")  # optional quick reference
    report['Risk Factors'] = factors if factors else ["None notable"]

    # Medications & Suggestions
    meds=[]
    suggestions=[]
    for m,suggestion in [('antibiotics',"Consider antibiotics if infection suspected"),
                         ('antihypertensives',"Ensure BP control with antihypertensives"),
                         ('insulin',"Manage glucose with insulin or oral agents"),
                         ('statins',"Continue statins for cardiac risk"),
                         ('anticoagulants',"Evaluate need for anticoagulation")]:
        if row.get(m,0): meds.append(m.capitalize())
        elif m=='antibiotics' and row.get('WBC mean',0)>11000:
            suggestions.append("Consider antibiotics for elevated WBC")
    report['Medications & Suggestions'] = {
        'Current Medications': meds if meds else ["None"], 
        'Suggestions': suggestions if suggestions else ["No additional suggestions"]
    }

    # Recommended Interventions
    report['Recommended Interventions'] = row.get('recommendation','Standard follow-up')

    # Notes for Clinicians
    notes=[]
    if row.get('temperature mean',0)>100.4: notes.append("Monitor for fever")
    if row.get('haemoglobin',0)<12: notes.append("Check for anemia")
    report['Notes for Clinicians'] = notes if notes else ["No immediate concerns"]

    return report


# -------------------------
# PDF Generation
# -------------------------
def create_patient_pdf_bytes(patient_row, summary_text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Patient Risk Summary", ln=True, align="C")
    pdf.set_font("Arial", "", 12)
    pdf.ln(10)
    pdf.multi_cell(0, 8, f"Patient Name: {patient_row.get('name','Unknown')}")
    pdf.multi_cell(0, 8, f"Age: {patient_row.get('agefactor',0)}")
    pdf.multi_cell(0, 8, f"Disease: {patient_row.get('disease','Unknown')}")
    pdf.multi_cell(0, 8, f"Risk Score: {patient_row.get('risk_score',0):.2f}% ({patient_row.get('risk_level','LOW')})")
    pdf.multi_cell(0, 8, f"Recommendation: {patient_row.get('recommendation','Standard follow-up')}")
    pdf.ln(5)
    pdf.multi_cell(0, 8, "AI-Generated Summary:")
    pdf.multi_cell(0, 8, summary_text)
    
    # Generate PDF as bytes
    pdf_bytes = pdf.output(dest='S').encode('latin1')  # dest='S' returns PDF as string
    return pdf_bytes

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(layout="wide")
st.title("🏥 Hospital EHR Patient Analysis")

st.sidebar.header("Hospital Parameters")
total_patients = st.sidebar.number_input("Total patients monthly", value=max(1000, len(patients_df)), step=1)
current_read_rate = st.sidebar.number_input("Current readmission rate (0-1)", value=0.15, step=0.01)

# -------------------------
# Patient Selection (Search + Dropdown)
# -------------------------
st.sidebar.subheader("🔍 Patient Selection")

# Create dictionary {patient_id: name}
patient_options = patients_df[['patient_id','name']].values.tolist()
patient_dict = dict(patient_options)

# -------------------------
# Patient Selection (Search + Dropdown, robust)
# -------------------------
st.sidebar.subheader("🔍 Patient Selection")

# Prebuild lookups
patient_ids = patients_df['patient_id'].tolist()
patient_names = patients_df['name'].astype(str).tolist() if 'name' in patients_df.columns else [str(x) for x in patient_ids]
id_to_name = dict(zip(patient_ids, patient_names))

# Build a temporary normalized view for searching
df_search = patients_df.copy()
df_search['patient_id_str'] = df_search['patient_id'].astype(str)
df_search['name_str'] = df_search['name'].astype(str) if 'name' in df_search.columns else df_search['patient_id_str']
# Lowercase helpers for case-insensitive matching
df_search['patient_id_str_l'] = df_search['patient_id_str'].str.lower()
df_search['name_str_l'] = df_search['name_str'].str.lower()
df_search['prefixed_id_l'] = ('patient_' + df_search['patient_id_str']).str.lower()

# Search input
q = st.text_input("Search Patient (ID or Name)", placeholder="e.g., 770, Patient_770, or patient name").strip()
q_l = q.lower()

selected_patient_id = None

if q:
    # 1) Try exact matches (ID, prefixed ID, or name)
    exact_mask = (
        (df_search['patient_id_str_l'] == q_l) |
        (df_search['prefixed_id_l'] == q_l) |
        (df_search['name_str_l'] == q_l)
    )
    exact_hits = df_search.loc[exact_mask]

    if len(exact_hits) == 1:
        selected_patient_id = exact_hits['patient_id'].iloc[0]
        st.success(f"✅ Found exact match: {id_to_name.get(selected_patient_id, selected_patient_id)} (ID: {selected_patient_id})")
    elif len(exact_hits) > 1:
        # Multiple exact hits (rare) -> let user choose
        st.info(f"Found {len(exact_hits)} exact matches. Please select one.")
        options = exact_hits['patient_id'].tolist()
        selected_patient_id = st.selectbox(
            "Select matching patient",
            options,
            format_func=lambda pid: f"{id_to_name.get(pid, str(pid))} (ID: {pid})"
        )
    else:
        # 2) Fuzzy / partial matches
        partial_mask = (
            df_search['patient_id_str_l'].str.contains(q_l, na=False) |
            df_search['prefixed_id_l'].str.contains(q_l, na=False) |
            df_search['name_str_l'].str.contains(q_l, na=False)
        )
        partial_hits = df_search.loc[partial_mask]

        if len(partial_hits) >= 1:
            st.info(f"Found {len(partial_hits)} match(es). Select from the list:")
            options = partial_hits['patient_id'].tolist()
            selected_patient_id = st.selectbox(
                "Select matching patient",
                options,
                format_func=lambda pid: f"{id_to_name.get(pid, str(pid))} (ID: {pid})"
            )
        else:
            st.warning("⚠️ No matching patient found. Showing full list below.")
            selected_patient_id = st.selectbox(
                "Select patient",
                patient_ids,
                format_func=lambda pid: id_to_name.get(pid, str(pid))
            )
else:
    # No query -> show full dropdown
    selected_patient_id = st.selectbox(
        "Select patient",
        patient_ids,
        format_func=lambda pid: id_to_name.get(pid, str(pid))
    )

# Use the chosen patient_id to get the row
patient_row = patients_df.loc[patients_df['patient_id'] == selected_patient_id].iloc[0]

# -------------------------
# Predictive Readmission Alert (Old Style)
# -------------------------
st.markdown("### Predictive Readmission Alert")
if patient_row['readmit_flag'] == "High Risk":
    st.markdown(f"<span style='color:red;font-weight:bold'>⚠ High-Risk ({patient_row['readmit_prob']*100:.1f}%)</span>", unsafe_allow_html=True)
elif patient_row['readmit_flag'] == "Moderate Risk":
    st.markdown(f"<span style='color:orange;font-weight:bold'>⚠ Medium-Risk ({patient_row['readmit_prob']*100:.1f}%)</span>", unsafe_allow_html=True)
else:
    st.markdown(f"<span style='color:green;font-weight:bold'>Low Risk ({patient_row['readmit_prob']*100:.1f}%)</span>", unsafe_allow_html=True)

# -------------------------
# Show patient details
# -------------------------
st.markdown("### Patient Details")
st.markdown(f"Name: {patient_row.get('name','')}")
st.markdown(f"Age: {patient_row.get('agefactor','')}")
st.markdown(f"Disease: {patient_row.get('disease','')}")

# Analyze Patient
if st.button("Analyze Patient"):
    st.markdown("### Patient Risk Assessment")
    st.markdown(f"Risk Score: {patient_row['risk_score']:.2f}%")
    st.markdown(f"Risk Level: {patient_row['risk_level']}")
    st.markdown(f"Recommendation: {patient_row['recommendation']}")

    st.markdown("### Individual Patient Impact")
    st.markdown(f"Expected Money Saved: ${patient_row['expected_saving']:,.2f}")

    st.markdown("### Overall Hospital Impact")
    st.markdown(f"Estimated Overall Savings: ${overall_impact:,.2f}")

    report = generate_structured_report(patient_row)

    readmission_cost = 15000             # Cost per readmission (matches individual_savings)
    prevention_success_rate = 0.7        # Probability that intervention prevents readmission
    high_risk_fraction = 0.2             # Fraction of patients considered high-risk

    patient_saving_sim = readmission_cost * patient_row['risk_score']/100 * prevention_success_rate
    high_risk_patients_sim = total_patients * high_risk_fraction
    hospital_saving_sim = high_risk_patients_sim * prevention_success_rate * readmission_cost
    max_penalty_sim = 26_000_000_000 * 0.15
    hospital_saving_sim = min(hospital_saving_sim, max_penalty_sim)

    st.markdown("### Simulation Results")
    sim_html = f"""
    <div style="font-family: 'Arial', sans-serif; font-size:16px; line-height:1.5;">
        <p><strong>Individual Patient Saving:</strong> ${patient_saving_sim:,.2f}</p>
        <p><strong>Hospital-wide Potential Savings:</strong> ${hospital_saving_sim:,.2f}</p>
    </div>
    """
    st.markdown(sim_html, unsafe_allow_html=True)

    st.markdown("### Possibility of Readmission")
    if patient_row['readmit_flag'] == "High Risk":
        st.markdown(f"<span style='color:red;font-weight:bold'>⚠ High-Risk of Readmission ({patient_row['readmit_prob']*100:.1f}%)</span>", unsafe_allow_html=True)
    elif patient_row['readmit_flag'] == "Medium Risk":
        st.markdown(f"<span style='color:orange;font-weight:bold'>⚠ Medium-Risk of Readmission ({patient_row['readmit_prob']*100:.1f}%)</span>", unsafe_allow_html=True)
    else:
        st.markdown(f"<span style='color:green;font-weight:bold'>Low Risk of Readmission ({patient_row['readmit_prob']*100:.1f}%)</span>", unsafe_allow_html=True)

    # Generative AI Summary
    st.markdown("AI-Generated Patient Summary")
    report = generate_structured_report(patient_row)
    summary_text = ""
    for section, content in report.items():
        summary_text += f"{section}:\n"
        if isinstance(content, dict):
            for k,v in content.items():
                summary_text += f"  {k}: {', '.join(v)}\n"
        elif isinstance(content, list):
            for item in content:
                summary_text += f"  - {item}\n"
        else:
            summary_text += f"  {content}\n"
        summary_text += "\n"

    st.text_area("Summary", value=summary_text, height=300)

    # PDF download button
    pdf_bytes = create_patient_pdf_bytes(patient_row, summary_text)

    st.download_button(
        label="Download Patient Summary PDF",
        data=pdf_bytes,
        file_name=f"{patient_row.get('name','unknown')}_summary.pdf",
        mime="application/pdf"
    )

    # -------------------------
    # SHAP Feature Contributions
    # -------------------------
    if ml_model:
        import shap
        import matplotlib.pyplot as plt

        # Get model feature names
        feature_cols = ml_model.get_booster().feature_names

        # Ensure all features exist in patients_df
        for col in feature_cols:
            if col not in patients_df.columns:
                patients_df[col] = 0  # fill missing features with 0

        # Select features for this patient
        X_patient = patients_df.loc[patients_df['patient_id'] == selected_patient_id, feature_cols]

        # Create SHAP explainer
        explainer = shap.TreeExplainer(ml_model)
        shap_values = explainer.shap_values(X_patient)

        # Convert to DataFrame for bar chart
        shap_df = pd.DataFrame({
            'Feature': feature_cols,
            'Contribution': shap_values[0]  # first (and only) row
        }).sort_values(by='Contribution', key=abs, ascending=False)

        st.markdown("### Feature Contribution to Readmission Risk (SHAP)")
        st.bar_chart(shap_df.set_index('Feature')['Contribution'])

