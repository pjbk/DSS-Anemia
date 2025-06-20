import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from streamlit.components.v1 import html

# ----------------- Load Model & Scaler -----------------
model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")

# ----------------- Page Config -----------------
st.set_page_config(
    page_title="Anemia DSS",
     page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- Custom CSS -----------------
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        background-color: #e63946;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: bold;
    }
    .stButton>button:hover { background-color: #c1121f; }
    .css-1aumxhk {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .header { color: #780000; font-size: 2.5em; font-weight: bold; margin-bottom: 0.5em; }
    .subheader { color: #003049; font-size: 1.2em; margin-bottom: 1.5em; }
    .normal-range { font-size: 0.8em; color: #457b9d; font-style: italic; }
    </style>
""", unsafe_allow_html=True)

# ----------------- Sidebar -----------------
with st.sidebar:
    st.markdown("### 🔍 About Anemia DSS")
    st.markdown("""
    This app predicts **anemia risk** based on patients clinical hematological parameters using a pre-trained **Random Forest Classifier** model.

   _________________________________________________________________________________________
    """)

# ----------------- UI Layout -----------------
st.markdown('<div class="header">Anemia Risk Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Enter the hematological parameters below to assess anemia risk.</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Patient Information")
    age = st.number_input("Age (years)", 1, 120, 30)
    gender = st.selectbox("Gender", ["Female", "Male"])

with col2:
    st.markdown("### .  .  .")
    hb = st.number_input("Hemoglobin (g/dL)", 3.0, 20.0, 12.0, step=0.05)
    st.markdown('<div class="normal-range">Normal: 12–18 g/dL</div>', unsafe_allow_html=True)
    rbc = st.number_input("RBC (10⁶/μL)", 1.0, 8.0, 4.5, step=0.01)
    st.markdown('<div class="normal-range">Normal: 3.9–6.1 10⁶/μL</div>', unsafe_allow_html=True)

col3, col4 = st.columns(2)

with col3:
    pcv = st.number_input("PCV/Hematocrit (%)", 10.0, 60.0, 36.0, step=0.1)
    st.markdown('<div class="normal-range">Normal: 36–54%</div>', unsafe_allow_html=True)
    mcv = st.number_input("MCV (fL)", 50.0, 120.0, 85.0, step=0.1)
    st.markdown('<div class="normal-range">Normal: 80–100 fL</div>', unsafe_allow_html=True)

with col4:
    mch = st.number_input("MCH (pg)", 15.0, 40.0, 27.0, step=0.1)
    st.markdown('<div class="normal-range">Normal: 26–34 pg</div>', unsafe_allow_html=True)
    mchc = st.number_input("MCHC (g/dL)", 25.0, 38.0, 32.0, step=0.1)
    st.markdown('<div class="normal-range">Normal: 31–36 g/dL</div>', unsafe_allow_html=True)

# ----------------- Prediction -----------------
gender_num = 1 if gender == "Female" else 0
features = np.array([[gender_num, age, hb, rbc, pcv, mcv, mch, mchc]])
feature_names = ["Gender", "Age", "Hemoglobin", "RBC", "Hematocrit", "MCV", "MCH", "MCHC"]
input_scaled = scaler.transform(features)


# ----------------- Column 1: Prediction Result -----------------
if st.button("Predict Anemia Risk"):
    prediction = model.predict(input_scaled)[0]
    probability_vector = model.predict_proba(input_scaled)[0]  # Full [non-anemic, anemic]
    probability = probability_vector[1]  # Probability of being anemic

    st.markdown("---")
    st.subheader("Analysis Results")
    res_col1, res_col2 = st.columns(2)

    # ----- Column 1: Anemia Status and Risk Tag -----
    with res_col1:
        if prediction == 0:
            st.success("## Non-Anemic")
            # st.markdown("The blood parameters suggest **normal hemoglobin levels**.")
        else:
            st.error("### Anemic")
            # st.markdown("The blood parameters suggest **possible anemia**.")

        tag = "**High Risk of Anemia**" if prediction == 1 else "**Low Risk of Anemia**"
        if prediction == 1:
            st.error(f"{tag}: The blood parameters indicate a high likelihood of anemia. Further diagnostic assessment is recommended.")
        else:
            st.success(f"{tag}: Hematological findings do not support a diagnosis of anemia. No clinical concern identified based on current values.")

    # ----- Column 2: Confidence Scores -----
    with res_col2:
        st.markdown("### Confidence Score")
        classes = ["Non-Anemic", "Anemic"]
        colors = ['#28a745', '#e63946']

        for i, (cls, prob) in enumerate(zip(classes, probability_vector)):
            st.write(f"{cls}:")
            progress_html = f"""
            <div style="background-color: #e0e0e0; border-radius: 10px; width: 100%; height: 20px; margin-bottom: 5px;">
                <div style="background-color: {colors[i]}; 
                            width: {prob*100}%; height: 100%; border-radius: 10px; text-align: center; 
                            color: white; font-weight: bold;">{prob*100:.2f}%</div>
            </div>
            """
            st.markdown(progress_html, unsafe_allow_html=True)
  
    st.markdown("---")
    # ----------------- SHAP Explanation -----------------
    st.markdown("### XAI-Based Model Explanation (SHAP Force Plot)")

    try:
        feature_names = ['gender', 'age', 'hb', 'rbc', 'pcv', 'mcv', 'mch', 'mchc']

        # Adjust estimator name if needed
        rf_model = model.named_estimators_['cat']
        explainer = shap.TreeExplainer(rf_model)
        shap_values = explainer.shap_values(input_scaled)

        shap.initjs()
        force_plot = shap.force_plot(
            base_value=explainer.expected_value,
            shap_values=shap_values[0],
            features=input_scaled[0],
            feature_names=feature_names,
            matplotlib=False
        )
        html(
            f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>",
            height=130
        )
        st.markdown("---")
        # AI-generated Explanation
        st.markdown("### XAI-Generated Report")
        shap_dict = {
            name: (value, shap_val)
            for name, value, shap_val in zip(feature_names, input_scaled[0], shap_values[0])
        }

        sorted_features = sorted(shap_dict.items(), key=lambda x: abs(x[1][1]), reverse=True)

        st.markdown("Top 5 SHAP-Weighted Risk Factors")
        top_n = 5
        cols = st.columns(top_n)
        for i, (feature, (value, shap_val)) in enumerate(sorted_features[:top_n]):
            direction = "🠙 increased" if shap_val > 0 else "🠛 decreased"
            color = "#e74c3c" if shap_val > 0 else "#27ae60"
            with cols[i]:
                st.markdown(
                    f"<span style='color:{color}'>→ <b>{feature}</b><br>{value:.2f} ({direction} risk)</span>",
                    unsafe_allow_html=True
            )


    except Exception as e:
        st.error("⚠️ SHAP explanation failed.")
        st.exception(e)

# ----------------- Footer -----------------
st.markdown("""
---
<div style='text-align: center; font-size: 15px;'>
Developed by <b>Pankaj Bhowmik</b><br>
Lecturer, Department of Computer Science and Engineering <br>
Hajee Mohammad Danesh Science and Technology University<br>
© 2025 All Rights Reserved.
</div>
""", unsafe_allow_html=True)
