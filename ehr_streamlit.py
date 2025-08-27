
# ehr_streamlit_refined.py

import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
from fpdf import FPDF
import io

# -------------------------
# Load dataset
# -------------------------
DB_PATH = "mini_ehr.db"

def load_patients_from_db(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql("SELECT * FROM admissions_scored", conn)
    finally:
        conn.close()
    return df

patients_df = load_patients_from_db(DB_PATH)

# -------------------------
# Risk calculation
# -------------------------
def calculate_risk_score(row):
    """
    Calculate risk score using a logistic formula with example weights.
    """
    # Scale features for more meaningful sigmoid
    age_scaled = row['agefactor'] / 100
    wbc_scaled = row['WBC mean'] / 20000
    hr_scaled = row['heart rate'] / 200
    diabetes = row['diabetes']
    hypertension = row['hypertension']

    # Linear combination
    ans = (0.4 * age_scaled + 0.8 * wbc_scaled + 0.3 * hr_scaled + 1.5 * diabetes + 1.0 * hypertension)
    
    # Sigmoid to get risk probability
    risk_prob = 1 / (1 + np.exp(-ans))
    return risk_prob * 100  # percentage

# ---------------------------
# Risk Level Assignment
# ---------------------------
def assign_risk_level(risk_score):
    if risk_score >= 75:
        return "HIGH"
    elif risk_score >= 50:
        return "MEDIUM"
    else:
        return "LOW"

# -------------------------
# Individual penalty reduction
# -------------------------
def individual_savings(risk_score):
    """
    Calculate money saved by preventing readmission for a single patient
    """
    # Base readmission cost
    cost_per_patient = 15000

    # Probability patient would be readmitted
    readmit_prob = risk_score / 100

    # Expected saving for this patient
    expected_saving = cost_per_patient * readmit_prob * 0.7  # 70% prevention success
    return expected_saving


# ---------------------------
# Hospital-wide Impact
# ---------------------------
def hospital_impact(df):
    """
    Sum up all individual savings for overall impact
    """
    df['expected_saving'] = df['risk_score'].apply(individual_savings)
    total_saving = df['expected_saving'].sum()
    max_penalty_reduction = 26e9 * 0.15  # 15% of $26B
    overall_impact = min(total_saving, max_penalty_reduction)
    return overall_impact

# ---------------------------
# Apply to dataset
# ---------------------------
patients_df['risk_score'] = patients_df.apply(calculate_risk_score, axis=1)
patients_df['risk_level'] = patients_df['risk_score'].apply(assign_risk_level)
patients_df['expected_saving'] = patients_df['risk_score'].apply(individual_savings)
patients_df['recommendation'] = patients_df['risk_level'].apply(
    lambda x: "Extended monitoring" if x=="HIGH" else
              "Home care with follow-up" if x=="MEDIUM" else
              "Standard follow-up"
)
overall_impact = hospital_impact(patients_df)

# -------------------------
# Predictive Readmission Alerts
# -------------------------
def readmission_probability(row):
    """
    Simple probability of readmission in next 30 days
    based on risk score and comorbidities.
    """
    base_prob = row['risk_score'] / 100
    comorbidity_factor = (row['diabetes'] + row['hypertension'] +
                          row.get('ckd',0) + row.get('copd',0) +
                          row.get('cad',0) + row.get('stroke',0) +
                          row.get('cancer',0)) * 0.05
    prob = min(1, base_prob + comorbidity_factor)
    return prob

def assign_readmission_flag(prob):
    if prob >= 0.7:
        return "High Risk"
    elif prob >= 0.4:
        return "Moderate Risk"
    else:
        return "Low Risk"

patients_df['readmit_prob'] = patients_df.apply(readmission_probability, axis=1)
patients_df['readmit_flag'] = patients_df['readmit_prob'].apply(assign_readmission_flag)


# -------------------------
# Generative AI (example placeholder)
# -------------------------
def generate_structured_report(row):
    report = {}
    report['Patient Summary'] = f"{row['name']}, Age {row['agefactor']}, Risk Score: {row['risk_score']:.2f}% ({row['risk_level']})"
    
    # Risk Factors
    factors = []
    if row['agefactor'] > 65: factors.append("Advanced age")
    for c in ['diabetes','hypertension','ckd','copd','cad','stroke','cancer']:
        if row.get(c,0)==1: factors.append(c.capitalize())
    if row['WBC mean']>11000: factors.append(f"High WBC ({row['WBC mean']})")
    if row['heart rate']>100: factors.append(f"High HR ({row['heart rate']} bpm)")
    if row['BP-mean']>140 or row['BP-mean']<90: factors.append(f"Abnormal BP ({row['BP-mean']})")
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
        elif m=='antibiotics' and row['WBC mean']>11000: suggestions.append("Consider antibiotics for elevated WBC")
    report['Medications & Suggestions'] = {'Current Medications': meds if meds else ["None"], 
                                          'Suggestions': suggestions if suggestions else ["No additional suggestions"]}

    # Recommended Interventions
    report['Recommended Interventions'] = row['recommendation']

    # Notes for Clinicians
    notes=[]
    if row['temperature mean']>100.4: notes.append("Monitor for fever")
    if row['haemoglobin']<12: notes.append("Check for anemia")
    report['Notes for Clinicians'] = notes if notes else ["No immediate concerns"]

    return report

# -------------------------
# PDF Generation
# -------------------------
def create_patient_pdf(patient_row, summary_text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    
    # Title
    pdf.cell(0, 10, "Patient Risk Summary", ln=True, align="C")
    
    pdf.set_font("Arial", "", 12)
    pdf.ln(10)
    
    # Patient Info
    pdf.multi_cell(0, 8, f"Patient Name: {patient_row['name']}")
    pdf.multi_cell(0, 8, f"Age: {patient_row['agefactor']}")
    pdf.multi_cell(0, 8, f"Disease: {patient_row['disease']}")
    pdf.multi_cell(0, 8, f"Risk Score: {patient_row['risk_score']:.2f}% ({patient_row['risk_level']})")
    pdf.multi_cell(0, 8, f"Recommendation: {patient_row['recommendation']}")
    
    pdf.ln(5)
    pdf.multi_cell(0, 8, "AI-Generated Summary:")
    pdf.multi_cell(0, 8, summary_text)
    
    file_name = f"{patient_row['patient_id']}_summary.pdf"
    pdf.output(file_name)
    return file_name

# -------------------------
# Additional Sidebar for Dynamic Hospital Simulation
# -------------------------

st.sidebar.header("Simulation Parameters")
# High-risk fraction adjustment
high_risk_fraction = st.sidebar.slider(
    "High-risk patient fraction", min_value=0.05, max_value=0.5, value=0.2, step=0.01,
    help="Fraction of patients considered high-risk in the hospital"
)
# Prevention success rate
prevention_success_rate = st.sidebar.slider(
    "Prevention success rate", min_value=0.1, max_value=1.0, value=0.7, step=0.05,
    help="Expected success rate of interventions to prevent readmission"
)
# Readmission cost per patient
readmission_cost = st.sidebar.number_input(
    "Readmission cost per patient ($)", min_value=1000, max_value=50000, value=15000, step=500,
    help="Estimated cost of readmission for each patient"
)

# -------------------------
# Recalculate individual patient savings using simulation parameters
# -------------------------
if st.button("Simulate with Custom Parameters"):
    # Recalculate individual patient impact
    patient_risk_score = calculate_risk_score(patient_row)
    patient_saving_sim = readmission_cost * patient_risk_score/100 * prevention_success_rate

    # Recalculate overall hospital impact
    total_patients_sim = total_patients
    high_risk_patients_sim = total_patients_sim * high_risk_fraction
    hospital_saving_sim = high_risk_patients_sim * prevention_success_rate * readmission_cost
    max_penalty_sim = 26_000_000_000 * 0.15
    hospital_saving_sim = min(hospital_saving_sim, max_penalty_sim)

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(layout="wide")
st.title("🏥 Hospital EHR Patient Analysis")

st.sidebar.header("Hospital Parameters")
total_patients = st.sidebar.number_input("Total patients monthly", value=max(1000, len(patients_df)), step=1)
current_read_rate = st.sidebar.number_input("Current readmission rate (0-1)", value=0.15, step=0.01)


# Patient selection
patient_options = [(row["patient_id"], row["name"]) for idx, row in patients_df.iterrows()]
patient_dict = dict(patient_options)
selected_patient_id = st.selectbox("Select patient", list(patient_dict.keys()), format_func=lambda x: patient_dict[x])
patient_row = patients_df[patients_df["patient_id"] == selected_patient_id].iloc[0]

# Predictive Readmission
patient_row = patients_df[patients_df["patient_id"] == selected_patient_id].iloc[0]
if patient_row['readmit_flag'] == "High Risk": st.warning("🚨 High Readmission Risk: This patient may be readmitted in the next 30 days.")

# Show patient details
patient_row = patients_df[patients_df["patient_id"] == selected_patient_id].iloc[0]
st.markdown("### Patient Details")
st.markdown(f"Name: {patient_row.get('name','')}")
st.markdown(f"Age: {patient_row.get('agefactor','')}")
st.markdown(f"Disease: {patient_row.get('disease','')}")

# Analyze Patient
if st.button("Analyze Patient"):
    risk_score = patient_row['risk_score']
    risk_level = patient_row['risk_level']
    recommendation = patient_row['recommendation']
    expected_saving = patient_row['expected_saving']

    st.markdown("### Patient Risk Assessment")
    st.markdown(f"Risk Score: {patient_row['risk_score']:.2f}%")
    st.markdown(f"Risk Level: {patient_row['risk_level']}")
    st.markdown(f"Recommendation: {patient_row['recommendation']}")

    st.markdown("### Individual Patient Impact")
    st.markdown(f"Expected Money Saved: ${patient_row['expected_saving']:,.2f}")

    st.markdown("### Overall Hospital Impact")
    st.markdown(f"Estimated Overall Savings: ${overall_impact:,.2f}")

    patient_risk_score = risk_score  # use selected patient
    patient_saving_sim = readmission_cost * patient_risk_score/100 * prevention_success_rate
    high_risk_patients_sim = total_patients * high_risk_fraction
    hospital_saving_sim = high_risk_patients_sim * prevention_success_rate * readmission_cost
    max_penalty_sim = 26_000_000_000 * 0.15
    hospital_saving_sim = min(hospital_saving_sim, max_penalty_sim)

    st.markdown("### Simulation Results")
    st.markdown(f"Individual Patient Saving: ${patient_saving_sim:,.2f}")
    st.markdown(f"Hospital-wide Potential Savings: ${hospital_saving_sim:,.2f}")

    st.markdown("### Predictive Readmission Alert")
    if patient_row['readmit_flag'] == "High Risk":
        st.markdown(f"<span style='color:red;font-weight:bold'>⚠ High-Risk of Readmission ({patient_row['readmit_prob']*100:.1f}%)</span>", unsafe_allow_html=True)
    elif patient_row['readmit_flag'] == "Medium Risk":
        st.markdown(f"<span style='color:orange;font-weight:bold'>⚠ Medium-Risk of Readmission ({patient_row['readmit_prob']*100:.1f}%)</span>", unsafe_allow_html=True)
    else:
        st.markdown(f"<span style='color:green;font-weight:bold'>Low Risk of Readmission ({patient_row['readmit_prob']*100:.1f}%)</span>", unsafe_allow_html=True)


    # Generative AI Summary
    st.markdown("### AI-Generated Patient Summary")
    report = generate_structured_report(patient_row)
    # Display as text area (convert dict to formatted string)
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
    

    # PDF download
    pdf_file = create_patient_pdf(patient_row, summary_text)
    st.download_button(
        label="Download Patient Summary PDF",
        data=pdf_file,
        file_name=f"{patient_row['name']}_summary.pdf",
        mime="application/pdf"
    )
# -------------------------
# Natural Language Q&A for Patient Search
# -------------------------

import re

st.markdown("## 🔍 Natural Language Q&A")
query = st.text_input("Ask a question (e.g., 'Show me all diabetic patients >70 with high WBC')")

def parse_query(query, df):
    q = query.lower()
    filtered = df.copy()

    # Age filter
    age_match = re.search(r'>(\d+)', q)
    if age_match:
        age_val = int(age_match.group(1))
        filtered = filtered[filtered['agefactor'] > age_val]

    # Diabetes
    if "diabetic" in q or "diabetes" in q:
        filtered = filtered[filtered['diabetes'] == 1]

    # Hypertension
    if "hypertension" in q or "hypertensive" in q:
        filtered = filtered[filtered['hypertension'] == 1]

    # WBC high
    if "high wbc" in q or "wbc" in q:
        filtered = filtered[filtered['WBC mean'] > 11000]

    # High HR
    if "high heart rate" in q or "tachycardia" in q:
        filtered = filtered[filtered['heart rate'] > 100]

    # Risk Level
    if "high risk" in q:
        filtered = filtered[filtered['risk_level'] == "HIGH"]
    elif "medium risk" in q:
        filtered = filtered[filtered['risk_level'] == "MEDIUM"]
    elif "low risk" in q:
        filtered = filtered[filtered['risk_level'] == "LOW"]

    return filtered

if query:
    results = parse_query(query, patients_df)
    st.markdown(f"### Results ({len(results)} patients found)")
    if not results.empty:
        st.dataframe(results[['patient_id','name','agefactor','disease','risk_score','risk_level','readmit_flag']])
    else:
        st.info("No matching patients found.")
