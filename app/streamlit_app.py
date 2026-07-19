# streamlit_app.py - Main Streamlit application for menopause stage prediction
# Author: Sara YOUSSE
# Date: 2026-07-18

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from utils import (
    load_model_and_explainer,
    predict,
    get_shap_values,
    generate_openai_explanation,
    chat_with_openai
)

# ------------------------------------------------------------------------------
# Page configuration and custom CSS for better UX/UI
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="🩸 Menopause Stage Predictor",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to improve visual appeal and readability
st.markdown("""
    <style>
        .main-title { font-size: 3rem; font-weight: bold; color: #c0392b; }
        .sub-title { font-size: 1.2rem; color: #34495e; }
        .result-box { background-color: #f0f4f8; padding: 20px; border-radius: 10px; }
        .metric-label { font-weight: bold; color: #2c3e50; }
        .disclaimer { font-size: 0.8rem; color: #7f8c8d; margin-top: 30px; }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# Title and introduction
# ------------------------------------------------------------------------------
st.markdown('<div class="main-title">🩸 Menopause Stage Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI-powered decision support – explainable and empathetic</div>', unsafe_allow_html=True)
st.markdown("---")

# ------------------------------------------------------------------------------
# Load the model and explainer (cached for performance)
# ------------------------------------------------------------------------------
@st.cache_resource
def load_assets():
    return load_model_and_explainer()

model, explainer, feature_names = load_assets()

# ------------------------------------------------------------------------------
# Sidebar: user input for health parameters
# ------------------------------------------------------------------------------
st.sidebar.header("🧬 Your health parameters")

# Section: General information
st.sidebar.subheader("👤 General information")
age = st.sidebar.slider(
    "Age (years)",
    18, 80, 45,
    help="Your current age in years"
)

depression = st.sidebar.slider(
    "Depression score (0-27)",
    0, 27, 5,
    help="PHQ-9 score: 0=none, 27=severe"
)

income = st.sidebar.number_input(
    "Income-to-poverty ratio (0-5)",
    0.0, 5.0, 2.5, 0.1,
    help="Socio-economic indicator"
)

education = st.sidebar.selectbox(
    "Education level",
    [1, 2, 3, 4, 5],
    index=2,
    format_func=lambda x: {1:"< High school", 2:"High school", 3:"Some college", 4:"Bachelor", 5:"Master or +"}[x]
)

# Section: Reproductive health
st.sidebar.subheader("💊 Reproductive health")
age_first_period = st.sidebar.number_input(
    "Age at first period",
    8, 20, 13,
    help="Age when you had your first menstruation"
)

hormone_use = st.sidebar.selectbox(
    "Hormone therapy?",
    ["No", "Yes", "Unknown"],
    help="Current use of hormone replacement or contraception"
)
hormone_map = {"No": 2, "Yes": 1, "Unknown": np.nan}
hormone = hormone_map[hormone_use]

# Section: Hormonal panel
st.sidebar.subheader("🧪 Hormonal panel")
fsh = st.sidebar.number_input(
    "FSH (U/L)",
    0.0, 200.0, 10.0, 0.1,
    help="Follicle-stimulating hormone"
)

estradiol = st.sidebar.number_input(
    "Estradiol (pg/mL)",
    0.0, 500.0, 50.0, 1.0
)

testosterone = st.sidebar.number_input(
    "Testosterone (ng/dL)",
    0.0, 200.0, 30.0, 0.1
)

shbg = st.sidebar.number_input(
    "SHBG (nmol/L)",
    0.0, 200.0, 50.0, 1.0,
    help="Sex hormone-binding globulin"
)

# Section: Sleep and lifestyle
st.sidebar.subheader("😴 Sleep & lifestyle")
bedtime = st.sidebar.slider("Bedtime (hours after midnight)", 0, 6, 0)
wake_time = st.sidebar.slider("Wake time (hours after midnight)", 5, 12, 8)
# Calculate sleep duration in minutes
sleep_minutes = (wake_time - bedtime) if wake_time > bedtime else (24 + wake_time - bedtime)
st.sidebar.caption(f"Estimated sleep duration: {sleep_minutes // 60}h{sleep_minutes % 60}min")

alcohol = st.sidebar.number_input("Alcohol (drinks/day)", 0.0, 10.0, 1.0, 0.5)

health = st.sidebar.selectbox(
    "Self-rated health",
    [1, 2, 3, 4, 5],
    index=2,
    format_func=lambda x: {1:"Excellent", 2:"Very good", 3:"Good", 4:"Fair", 5:"Poor"}[x]
)

# Section: OpenAI API key (secure input)
st.sidebar.markdown("---")
st.sidebar.subheader("🔑 OpenAI Integration")
openai_api_key = st.sidebar.text_input(
    "OpenAI API Key",
    type="password",
)
use_openai = st.sidebar.checkbox("Enable AI explanation with OpenAI", value=False)

# ------------------------------------------------------------------------------
# Prepare input DataFrame with all features in the correct order
# ------------------------------------------------------------------------------
input_data = pd.DataFrame([[
    age, income, education, age_first_period, hormone,
    np.nan,  # RHQ078 (reason for no periods) – left blank, will be imputed
    testosterone, fsh, estradiol, shbg,
    alcohol, health, depression, bedtime*60, wake_time*60
]], columns=feature_names)

# ------------------------------------------------------------------------------
# Predict and display results
# ------------------------------------------------------------------------------
pred, proba = predict(model, input_data)
class_map = {1: "Premenopause", 2: "Perimenopause", 3: "Postmenopause"}
pred_label = class_map[pred]

st.subheader("🔮 Prediction result")
col1, col2, col3 = st.columns(3)
col1.metric("Predicted stage", pred_label, delta=None)
col2.metric("Confidence", f"{proba[pred-1]:.1%}")
col3.metric("Data completeness", "OK")

# Show probabilities as a bar chart
st.subheader("📊 Probabilities per stage")
prob_df = pd.DataFrame({
    "Stage": ["Premenopause", "Perimenopause", "Postmenopause"],
    "Probability": proba
})
st.bar_chart(prob_df.set_index("Stage"))

# ------------------------------------------------------------------------------
# SHAP explanation (waterfall plot)
# ------------------------------------------------------------------------------
st.subheader("🧠 Explanation of the prediction (SHAP)")

# Get SHAP values for this instance
shap_values, input_shap = get_shap_values(explainer, model, input_data, feature_names)

# Detect structure and extract for the predicted class
if isinstance(shap_values, list):
    # TreeExplainer returns a list per class
    class_idx = pred - 1
    shap_sample = shap_values[class_idx][0]
    base_value = explainer.expected_value[class_idx]
elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
    # New Explainer returns (samples, features, classes)
    class_idx = pred - 1
    shap_sample = shap_values[0, :, class_idx]
    base_value = explainer.expected_value[class_idx]
else:
    # Fallback: single output (e.g., binary or unexpected)
    shap_sample = shap_values[0] if isinstance(shap_values, np.ndarray) else shap_values[0]
    base_value = explainer.expected_value if hasattr(explainer, 'expected_value') else 0

# Generate waterfall plot
fig, ax = plt.subplots(figsize=(10, 6))
shap.waterfall_plot(
    shap.Explanation(
        values=shap_sample,
        base_values=base_value,
        data=input_shap.iloc[0],
        feature_names=feature_names
    ),
    show=False
)
st.pyplot(fig)
plt.close(fig)

# ------------------------------------------------------------------------------
# Natural language explanation using OpenAI (if enabled)
# ------------------------------------------------------------------------------
if use_openai and openai_api_key:
    # Extract top 5 features with largest absolute SHAP contribution
    shap_abs = np.abs(shap_sample)
    top_indices = np.argsort(shap_abs)[-5:][::-1]
    top_features = [(feature_names[i], shap_sample[i]) for i in top_indices]

    with st.spinner("Generating personalized explanation..."):
        explanation = generate_openai_explanation(
            openai_api_key, age, depression, fsh, age_first_period,
            hormone_use, estradiol, testosterone, shbg,
            sleep_minutes, alcohol, health, pred_label,
            top_features
        )
    st.subheader("💬 Personalized AI explanation")
    st.write(explanation)
else:
    st.info("Enable the option above and enter your OpenAI API key to get a natural-language explanation.")

# ------------------------------------------------------------------------------
# Chatbot assistant (conversational interface)
# ------------------------------------------------------------------------------
st.markdown("---")
st.subheader("💬 Personal assistant (chatbot)")

# Initialize conversation history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input for new user message
if prompt := st.chat_input("Ask a question about your health or results..."):
    # Append user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if openai_api_key:
        # Build context with key patient data
        context = f"Age: {age}, Depression: {depression}/27, Predicted stage: {pred_label}, FSH: {fsh} U/L, Estradiol: {estradiol} pg/mL"
        reply = chat_with_openai(openai_api_key, st.session_state.messages, context)
        # Append assistant reply to history
        st.session_state.messages.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)
    else:
        st.warning("Please enter your OpenAI API key to use the chatbot.")

# ------------------------------------------------------------------------------
# Disclaimer
# ------------------------------------------------------------------------------
st.markdown("---")
st.caption("⚠️ Research prototype – not a substitute for medical advice. Data is anonymized and simulated.")