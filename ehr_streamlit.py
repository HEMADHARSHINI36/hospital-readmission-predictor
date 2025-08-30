
import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
from fpdf import FPDF
from io import BytesIO
import pickle
import os
import altair as alt

# Optional: Ollama AI integration
import ollama

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
    if prob >= 0.7: return "Will be readmitted"
    elif prob >= 0.4: return "May be readmitted"
    else: return "Will not be readmitted"

if ml_model:
    features = patients_df[['agefactor','WBC mean','heart rate','diabetes','hypertension']].values
    try:
        preds = ml_model.predict_proba(features)[:,1]
        patients_df['readmit_prob'] = preds
    except:
        patients_df['readmit_prob'] = patients_df['risk_score'] / 100
else:
    patients_df['readmit_prob'] = patients_df['risk_score'] / 100

patients_df['readmit_flag'] = patients_df['readmit_prob'].apply(assign_readmission_flag)


# -------------------------
# Generative AI Report
# -------------------------
def generate_structured_report(row):
    report = {}
    readmit_prob_pct = row.get('readmit_prob', 0) * 100
    readmit_flag = row.get('readmit_flag', 'Low Risk')
    report['Patient Summary'] = (
        f"{row.get('name','Unknown')}, Age {row.get('agefactor',0)}, "
        f"Risk Score: {row.get('risk_score',0):.2f}% ({row.get('risk_level','LOW')}), "
        f"ML Readmission Probability: {readmit_prob_pct:.1f}% ({readmit_flag})"
    )

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
    factors.append(f"Readmission Risk: {readmit_flag}")
    report['Risk Factors'] = factors if factors else ["None notable"]

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

    report['Recommended Interventions'] = row.get('recommendation','Standard follow-up')

    notes=[]
    if row.get('temperature mean',0)>100.4: notes.append("Monitor for fever")
    if row.get('haemoglobin',0)<12: notes.append("Check for anemia")
    report['Notes for Clinicians'] = notes if notes else ["No immediate concerns"]

    return report


# -------------------------
# PDF Generation
# -------------------------
import re
from fpdf import FPDF

def create_patient_pdf_bytes(patient_row, summary_text):
    # Replace non-latin1 characters with a placeholder
    safe_summary = re.sub(r'[^\x00-\xFF]', '?', str(summary_text))

    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Patient Risk Summary", ln=True, align="C")
    pdf.ln(10)

    # Patient info
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 8, f"Patient Name: {patient_row.get('name','Unknown')}")
    pdf.multi_cell(0, 8, f"Age: {patient_row.get('agefactor',0)}")
    pdf.multi_cell(0, 8, f"Disease: {patient_row.get('disease','Unknown')}")
    pdf.multi_cell(0, 8, f"Risk Score: {patient_row.get('risk_score',0):.2f}% ({patient_row.get('risk_level','LOW')})")
    pdf.multi_cell(0, 8, f"Recommendation: {patient_row.get('recommendation','Standard follow-up')}")
    pdf.ln(5)

    # Summary text
    pdf.multi_cell(0, 8, safe_summary)

    # Return PDF bytes
    return pdf.output(dest='S').encode('latin1', errors='replace')

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
patient_ids = patients_df['patient_id'].tolist()
patient_names = patients_df['name'].astype(str).tolist() if 'name' in patients_df.columns else [str(x) for x in patient_ids]
id_to_name = dict(zip(patient_ids, patient_names))

df_search = patients_df.copy()
df_search['patient_id_str'] = df_search['patient_id'].astype(str)
df_search['name_str'] = df_search['name'].astype(str) if 'name' in df_search.columns else df_search['patient_id_str']
df_search['patient_id_str_l'] = df_search['patient_id_str'].str.lower()
df_search['name_str_l'] = df_search['name_str'].str.lower()
df_search['prefixed_id_l'] = ('patient_' + df_search['patient_id_str']).str.lower()

q = st.text_input("🔍 Search Patient (ID or Name)", placeholder="e.g., 770, Patient_770, or patient name").strip()
q_l = q.lower()
selected_patient_id = None

if q:
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
        st.info(f"Found {len(exact_hits)} exact matches. Please select one.")
        options = exact_hits['patient_id'].tolist()
        selected_patient_id = st.selectbox("Select matching patient", options, format_func=lambda pid: f"{id_to_name.get(pid, str(pid))} (ID: {pid})")
    else:
        partial_mask = (
            df_search['patient_id_str_l'].str.contains(q_l, na=False) |
            df_search['prefixed_id_l'].str.contains(q_l, na=False) |
            df_search['name_str_l'].str.contains(q_l, na=False)
        )
        partial_hits = df_search.loc[partial_mask]

        if len(partial_hits) >= 1:
            st.info(f"Found {len(partial_hits)} match(es). Select from the list:")
            options = partial_hits['patient_id'].tolist()
            selected_patient_id = st.selectbox("Select matching patient", options, format_func=lambda pid: f"{id_to_name.get(pid, str(pid))} (ID: {pid})")
        else:
            st.warning("⚠ No matching patient found. Showing full list below.")
            selected_patient_id = st.selectbox("Select patient", patient_ids, format_func=lambda pid: id_to_name.get(pid, str(pid)))
else:
    selected_patient_id = st.selectbox("Select patient", patient_ids, format_func=lambda pid: id_to_name.get(pid, str(pid)))

patient_row = patients_df.loc[patients_df['patient_id'] == selected_patient_id].iloc[0]


# -------------------------
# Patient Dashboard (Improved UI)
# -------------------------
st.markdown("## 🩺 Patient Dashboard")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Risk Score", f"{patient_row['risk_score']:.1f}%", patient_row['risk_level'])
with col2:
    st.metric("Readmission Prob", patient_row['readmit_flag'])
with col3:
    st.metric("Expected Saving", f"${patient_row['expected_saving']:,.0f}")

alert_color = "#e74c3c" if patient_row['readmit_flag']=="Will be readmitted" else "#f39c12" if patient_row['readmit_flag']=="May be readmitted" else "#27ae60"
st.markdown(
    f"""
    <div style="padding:12px; border-radius:10px; background-color:{alert_color}; color:white; font-weight:bold; text-align:center;">
        ⚠ Predictive Readmission Alert: {patient_row['readmit_flag']} ({patient_row['readmit_prob']*100:.1f}%)
    </div>
    """,
    unsafe_allow_html=True
)

# -------------------------
# Tabs
# -------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Risk Assessment",
    "🏥 Penalty Reduction",
    "📝 AI Summary",
    "🔬 SHAP Analysis",
    "📋 Post-Discharge Plan",
    "📊 Root-Cause Explorer"
])

# -------------------------
# Tab 1: Risk Assessment
# -------------------------
import plotly.graph_objects as go

with tab1:
    st.markdown("## 🧾 Patient Risk Assessment")

    risk_score = patient_row['risk_score']
    risk_level = patient_row['risk_level']
    recommendation = patient_row['recommendation']

    # 🎯 Gauge Chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Risk Score (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "red"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 60], 'color': "yellow"},
                {'range': [60, 100], 'color': "tomato"}
            ]
        }
    ))

    st.plotly_chart(fig, use_container_width=True)

    # 🎨 Color mapping for risk levels
    risk_colors = {
        "LOW": "#2ecc71",       # Green
        "MEDIUM": "#f39c12",    # Orange
        "HIGH": "#e74c3c"       # Red
    }
    risk_color = risk_colors.get(risk_level.upper(), "#34495e")

    # 🎨 Color mapping for recommendations
    rec_colors = {
        "Extended monitoring": "#2980b9",    # Blue
        "Immediate intervention": "#c0392b", # Dark Red
        "Routine check": "#27ae60"           # Green
    }
    rec_color = rec_colors.get(recommendation, "#34495e")

    # UI Block for Risk Level and Recommendation
    st.markdown(
        f"""
        <h3 style="color:#d35400;">⚠️ Risk Level</h3>   <!-- Changed to strong orange -->
        <span style="background:linear-gradient(135deg,{risk_color},#ffffff20);
                    color:white;padding:8px 18px;border-radius:25px;
                    font-size:20px;font-weight:bold;
                    box-shadow:0 2px 6px rgba(0,0,0,0.2);">
            {risk_level}
        </span>

        <h3 style="color:#16a085;margin-top:25px;">💡 Recommendation</h3> <!-- Changed to teal/green -->
        <span style="background:linear-gradient(135deg,#2980b9,#6dd5fa);
                    color:white;padding:8px 18px;border-radius:25px;
                    font-size:18px;font-weight:bold;
                    box-shadow:0 2px 6px rgba(0,0,0,0.2);">
            {recommendation}
        </span>
        """,
        unsafe_allow_html=True
    )

# -------------------------
# Tab 2: Simulation
# -------------------------
with tab2:
    st.markdown("## 🧪 Simulation Results")

    # Simulation parameters
    readmission_cost = 15000
    prevention_success_rate = 0.7
    high_risk_fraction = 0.2

    patient_saving_sim = readmission_cost * patient_row['risk_score']/100 * prevention_success_rate
    high_risk_patients_sim = total_patients * high_risk_fraction
    hospital_saving_sim = high_risk_patients_sim * prevention_success_rate * readmission_cost
    max_penalty_sim = 26_000_000_000 * 0.15
    hospital_saving_sim = min(hospital_saving_sim, max_penalty_sim)

    # UI cards instead of plain st.metric
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            f"""
            <div style="background:linear-gradient(135deg,#27ae60,#2ecc71);
                        padding:20px;border-radius:15px;color:white;
                        text-align:center;box-shadow:0 4px 10px rgba(0,0,0,0.15);">
                <h3 style="margin-bottom:5px;">👤 Individual Saving</h3>
                <p style="font-size:26px;font-weight:bold;margin:0;">
                    ${patient_saving_sim:,.0f}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div style="background:linear-gradient(135deg,#2980b9,#6dd5fa);
                        padding:20px;border-radius:15px;color:white;
                        text-align:center;box-shadow:0 4px 10px rgba(0,0,0,0.15);">
                <h3 style="margin-bottom:5px;">🏥 Hospital-wide Potential</h3>
                <p style="font-size:26px;font-weight:bold;margin:0;">
                    ${hospital_saving_sim:,.0f}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Footnote / explanation
    st.markdown(
        """
        <p style="font-size:14px;color:white;margin-top:15px;">
        💡 These estimates are based on <b>prevention success rate (70%)</b> 
        and <b>average readmission cost ($15,000)</b>.  
        Hospital-wide savings are capped at the CMS maximum penalty limit.
        </p>
        """,
        unsafe_allow_html=True
    )

# -------------------------
# Tab 3: AI Summary
# -------------------------
with tab3:
    st.markdown("## 🤖 AI-Generated Patient Summary")

    # Generate structured report
    report = generate_structured_report(patient_row)

    # Render each section as a colored card
    color_map = {
        "Clinical": "#e74c3c",     # red
        "Behavioral": "#f39c12",  # orange
        "Social/System": "#2980b9", # blue
        "Other": "#8e44ad"        # purple
    }

    icons = {
        "Clinical": "🩺",
        "Behavioral": "🧠",
        "Social/System": "🌍",
        "Other": "📌"
    }

    for section, content in report.items():
        items_html = ""
        if isinstance(content, dict):
            for k, v in content.items():
                items_html += f"<li><b>{k}:</b> {', '.join(v)}</li>"
        elif isinstance(content, list):
            for item in content:
                items_html += f"<li>{item}</li>"
        else:
            items_html += f"<li>{content}</li>"

        st.markdown(
            f"""
            <div style="background:{color_map.get(section,'#34495e')};padding:15px;
                        border-radius:12px;margin-bottom:15px;
                        color:white;box-shadow:0 4px 10px rgba(0,0,0,0.2);">
                <h4 style="margin:0;">{icons.get(section,'📄')} {section}</h4>
                <ul style="margin:8px 0 0 15px;">{items_html}</ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Generate PDF
    pdf_bytes = create_patient_pdf_bytes(patient_row, str(report))

    # One styled download button (no duplicate)
    custom_css = """
        <style>
        div.stDownloadButton > button {
            background: linear-gradient(90deg,#16a085,#27ae60) !important;
            color: white !important;
            font-weight: bold !important;
            border-radius: 25px !important;
            padding: 10px 24px !important;
            border: none !important;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

    st.download_button(
        "⬇ Download Patient Summary PDF",
        data=pdf_bytes,
        file_name=f"{patient_row.get('name','unknown')}_summary.pdf",
        mime="application/pdf"
    )

# -------------------------
# Tab 4: SHAP Analysis
# -------------------------
with tab4:
    if ml_model:
        import shap

        feature_cols = ml_model.get_booster().feature_names
        for col in feature_cols:
            if col not in patients_df.columns:
                patients_df[col] = 0

        X_patient = patients_df.loc[patients_df['patient_id'] == selected_patient_id, feature_cols]

        explainer = shap.TreeExplainer(ml_model)
        shap_values = explainer.shap_values(X_patient)

        # Convert to long-format DataFrame for root-cause tab
        shap_values_df = pd.DataFrame({
            'feature': feature_cols,
            'shap_value': shap_values[0],
            'patient_id': selected_patient_id
        })

        shap_df = (
            pd.DataFrame({
                'Feature': feature_cols,
                'Contribution': shap_values[0]
            })
            .sort_values(by='Contribution', key=lambda x: abs(x), ascending=False)
            .head(5)
        )

        # Add direction for coloring
        shap_df["Direction"] = shap_df["Contribution"].apply(lambda x: "↑ Increases Risk" if x > 0 else "↓ Decreases Risk")

        st.markdown("### 🔬 Top 5 Factors Driving Readmission Risk")

        import altair as alt
        chart = (
            alt.Chart(shap_df)
            .mark_bar()
            .encode(
                x=alt.X("Contribution:Q", title="Impact on Risk"),
                y=alt.Y("Feature:N", sort="-x"),
                color=alt.Color("Direction:N", scale=alt.Scale(domain=["↑ Increases Risk","↓ Decreases Risk"],
                                                            range=["#d9534f","#5cb85c"])),
                tooltip=["Feature", "Contribution"]
            )
        )

        text = chart.mark_text(
            align="left",
            baseline="middle",
            dx=3
        ).encode(
            text=alt.Text("Contribution:Q", format=".2f")
        )

        st.altair_chart(chart + text, use_container_width=True)
        st.caption("🔴 Positive values = higher risk contribution | 🟢 Negative values = protective effect")

        # --- Medical explanation mapping ---
        explanation_dict = {
            "BP-mean": "Average blood pressure. Abnormal values (too high or too low) increase cardiovascular risk.",
            "haemoglobin": "Hemoglobin levels indicate oxygen-carrying capacity. Low values suggest anemia, linked to fatigue and complications.",
            "charlson_index": "Charlson Comorbidity Index measures the overall burden of chronic illnesses. Higher scores = higher readmission risk.",
            "agefactor": "Patient’s age factor. Older age is associated with weaker immunity and more comorbidities.",
            "heart rate": "Resting heart rate. Abnormal rates (tachycardia or bradycardia) can indicate underlying cardiovascular or systemic issues."
        }

        # Auto-explanation for other features
        def auto_explain(feature: str) -> str:
            f = feature.lower()
            if "bp" in f or "blood" in f:
                return "Blood pressure related measurement, linked to cardiovascular risk."
            elif "hr" in f or "heart" in f or "pulse" in f:
                return "Heart rate indicator. Abnormal values may suggest cardiovascular stress."
            elif "age" in f:
                return "Age-related risk factor. Older age increases likelihood of complications."
            elif "charlson" in f or "comorbidity" in f:
                return "Index reflecting burden of chronic diseases."
            elif "hemo" in f or "rbc" in f or "hgb" in f:
                return "Blood/hemoglobin related measure. Low values may indicate anemia."
            elif "glucose" in f or "sugar" in f:
                return "Glucose level. High values suggest diabetes or metabolic issues."
            elif "bmi" in f or "weight" in f:
                return "Body weight indicator. Higher values associated with obesity-related risks."
            else:
                return "No medical explanation available."

        # --- Show medical context cards ---
        st.markdown("### 📘 Medical Context for Factors")
        for _, row in shap_df.iterrows():
            feature = row["Feature"]
            contribution = row["Contribution"]
            explanation = explanation_dict.get(feature, auto_explain(feature))
            color = "#d9534f" if contribution > 0 else "#5cb85c"
            direction = "↑ Risk" if contribution > 0 else "↓ Protective"

            st.markdown(
                f"""
                <div style="border-left:6px solid {color}; padding:10px; margin-bottom:12px;
                            border-radius:8px;">
                    <b style="color:{color};">{feature}</b>  
                    <br><span style="font-size:16px;color:grey;">{explanation}</span>  
                    <br><b>Contribution:</b> <span style="color:{color};">{contribution:.2f} ({direction})</span>
                </div>
                """,
                unsafe_allow_html=True
            )

    else:
        st.info("SHAP analysis available only with ML model.")

# -------------------------
# Tab 5: Post-Discharge Plan
# -------------------------
from huggingface_hub import InferenceClient

def generate_post_discharge_plan_safe(row):
    try:
        with st.spinner("Generating AI Post-Discharge Plan..."):
            patient_info = f"Name: {row.get('name','Unknown')}, Age: {row.get('agefactor',0)}, Risk Score: {row.get('risk_score',0):.2f}%"
            prompt = f"Generate a clinician-friendly post-discharge plan for the patient:\n{patient_info}"

            # Initialize client (requires HF API token in env)
            client = InferenceClient(model="mistralai/Mistral-7B-Instruct-v0.1")

            # Correct usage with `inputs=`
            response = client.text_generation(inputs=prompt, max_new_tokens=300)

            # response is a string in new API
            return response.strip() if isinstance(response, str) else str(response)
    except Exception as e:
        return f"❌ Could not generate plan: {e}"


        
import re
from fpdf import FPDF

def create_pdf_from_text(patient_row, text, title="Post-Discharge Plan"):
    # Replace non-latin1 characters (like emojis) with a placeholder
    safe_text = re.sub(r'[^\x00-\xFF]', '?', str(text))

    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, title, ln=True, align="C")
    pdf.ln(10)

    # Patient info
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 8, f"Patient Name: {patient_row.get('name','Unknown')}")
    pdf.multi_cell(0, 8, f"Age: {patient_row.get('agefactor',0)}")
    pdf.multi_cell(0, 8, f"Disease: {patient_row.get('disease','Unknown')}")
    pdf.ln(5)

    # Discharge plan text
    pdf.multi_cell(0, 8, safe_text)

    # Return PDF bytes
    return pdf.output(dest='S').encode('latin1', errors='replace')


with tab5:
    st.subheader("AI-Generated Post-Discharge Plan")
    discharge_plan = generate_post_discharge_plan_safe(patient_row)
    st.text_area("Plan", value=discharge_plan, height=400)
    pdf_bytes = create_pdf_from_text(patient_row, discharge_plan, title="Post-Discharge Plan")
    st.download_button("⬇ Download PDF", data=pdf_bytes, file_name=f"{patient_row.get('name','unknown')}_discharge_plan.pdf", mime="application/pdf")

# -------------------------
# Root-Cause Explorer Tab
# -------------------------
with tab6:
    st.subheader("Readmission Root-Cause Explorer")

    # shap_values_df should be prepared from your existing ML model
    # Example:
    # shap_values_df = pd.DataFrame({
    #     'patient_id': [...],
    #     'feature': [...],
    #     'shap_value': [...]
    # })

    # Map features to categories
    feature_categories = {
        'agefactor': 'Clinical',
        'WBC mean': 'Clinical',
        'heart rate': 'Clinical',
        'diabetes': 'Clinical',
        'hypertension': 'Clinical',
        'missed_followups': 'Behavioral',
        'med_adherence': 'Behavioral',
        'lives_alone': 'Social/System',
        'distance_to_hospital': 'Social/System',
        'insurance_status': 'Social/System'
    }

    def root_cause_explorer(patient_id, shap_values_df):
        patient_shap = shap_values_df[shap_values_df['patient_id']==patient_id].copy()
        if patient_shap.empty:
            st.warning("No SHAP values found for this patient.")
            return

        # Add category
        patient_shap['Category'] = patient_shap['feature'].map(feature_categories).fillna('Other')

        # Category summary
        category_summary = (
            patient_shap.groupby('Category')['shap_value']
            .apply(lambda x: np.sum(np.abs(x)))
            .reset_index()
            .rename(columns={'shap_value':'Total_Impact'})
        )

        # ==========================
        # 🔝 Top Drivers - Waterfall Chart
        # ==========================
        st.markdown("### 🔝 Top Drivers of Readmission Risk (Waterfall View)")

        import plotly.graph_objects as go

        # Select top 5 features by absolute SHAP value
        top_features = (
            patient_shap.sort_values(by="shap_value", key=lambda x: abs(x), ascending=False)
            .head(5)
        )

        # Build waterfall chart
        fig = go.Figure(
            go.Waterfall(
                name="Risk Impact",
                orientation="v",
                measure=["relative"] * len(top_features),
                x=top_features["feature"],
                text=[f"{val:.2f}" for val in top_features["shap_value"]],
                y=top_features["shap_value"],
                connector={"line": {"color": "rgba(63, 63, 63, 0.5)"}},
                textposition="outside",
                increasing={"marker": {"color": "salmon"}},  # factors that ↑ risk
                decreasing={"marker": {"color": "lightgreen"}},  # factors that ↓ risk
            )
        )

        fig.update_layout(
            title="Feature Contributions to Patient’s Readmission Risk",
            xaxis_title="Feature",
            yaxis_title="Impact on Risk Score",
            showlegend=False,
            waterfallgap=0.3,
        )

        st.plotly_chart(fig, use_container_width=True)

        # 📝 Doctor-friendly explanation
        st.info("""
        *How to read this chart:*
        - Each bar shows how much a feature pushes the patient’s readmission risk *up (red)* 
          or *down (green)*.
        - Longer bars = stronger influence.
        - The cumulative effect of these drivers explains why the patient’s overall risk is high or low.
        
        Example: If HbA1c and missed_followups have tall red bars, they are the strongest risk drivers.
        If med_adherence shows a green bar, it means good adherence is protecting against readmission risk.
        """)



        # Pie chart: category contributions
        st.markdown("### 🗂 Category-wise Risk Contribution")
        pie_chart = (
            alt.Chart(category_summary)
            .mark_arc(innerRadius=50)
            .encode(
                theta=alt.Theta('Total_Impact:Q'),
                color=alt.Color('Category:N', scale=alt.Scale(scheme='set1')),
                tooltip=['Category','Total_Impact']
            )
        )
        st.altair_chart(pie_chart, use_container_width=True)

        # 👉 Doctor-friendly explanation
        st.info("""
        *How to read this chart:*
        - Each slice shows how much a category (Clinical, Behavioral, Social/System) contributes to the patient’s readmission risk.  
        - Larger slices = stronger impact on overall risk.  
        - Example: if Behavioral is 40%, then missed follow-ups or poor medication adherence are the main drivers.  
        - This helps you focus interventions where they will matter most.
        """)

        # ⚠ Highlight biggest driver
        top_category = category_summary.sort_values("Total_Impact", ascending=False).iloc[0]
        st.warning(f"⚠ Main driver: *{top_category['Category']}* "
                f"({top_category['Total_Impact']:.2f} impact on risk)")

        # 📝 Doctor-friendly explanation for "Other"
        if top_category['Category'] == "Other":
            st.info(f"""
            *Note for Doctors:*
            The main driver for this patient falls under *'Other'*, meaning the model
            identified a risk factor that wasn’t explicitly categorized as Clinical, Behavioral,
            or Social/System.

            This usually includes *less common medical variables* or lab values that still
            influence readmission risk. While they are important, they may not fall into the
            standard categories.
            """)

        # Optional: simulate modifiable factors
        st.markdown("### 🔄 Simulate Modifiable Factors")
        modifiable_features = ['diabetes', 'hypertension', 'med_adherence', 'missed_followups']
        simulated_shap = top_features.copy()
        for feat in modifiable_features:
            if feat in top_features['feature'].values:
                current_val = st.slider(f"{feat} level", 0, 1, 1)
                simulated_shap.loc[simulated_shap['feature']==feat, 'shap_value'] *= current_val

        simulated_risk_change = np.sum(np.abs(simulated_shap['shap_value']))
        st.metric("Simulated Total Risk Contribution", f"{simulated_risk_change:.2f}")

        # 📝 Doctor-friendly explanation
        st.info(f"""
        *Interpretation:*
        This value ({simulated_risk_change:.2f}) represents the total contribution of
        modifiable factors (like diabetes control, hypertension, medication adherence, and follow-up visits)
        to the patient's readmission risk.

        - A *higher value* = these factors strongly drive risk.
        - If you adjust the sliders (e.g., improving medication adherence),
        the contribution decreases, showing *potential risk reduction*.
        """)

    # Call the explorer for the selected patient
    if ml_model and 'shap_values_df' in globals():
        root_cause_explorer(selected_patient_id, shap_values_df)
    else:
        st.info("Run the SHAP Analysis tab first to generate SHAP values for this patient.")





