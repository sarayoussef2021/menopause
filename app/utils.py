# utils.py - Utility functions for the Streamlit menopause prediction app
# Author: Sara YOUSSE
# Date: 2026-07-19

import os
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import openai
import streamlit as st

# ------------------------------------------------------------------------------
# Fixed path to Dataset folder (robust)
# ------------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
data_path = os.path.join(project_root, "Dataset/")

# ------------------------------------------------------------------------------
# Function: load_model_and_explainer
# ------------------------------------------------------------------------------
@st.cache_resource
def load_model_and_explainer():
    """
    Load the trained model, SHAP explainer, and feature names from the Dataset folder.
    Returns:
        model: trained sklearn pipeline
        explainer: SHAP TreeExplainer object
        feature_names: list of feature names in the same order as training
    """
    model = joblib.load(os.path.join(data_path, "menopause_rf_model.pkl"))
    explainer = joblib.load(os.path.join(data_path, "shap_explainer.pkl"))
    with open(os.path.join(data_path, "feature_names.txt"), "r") as f:
        feature_names = [line.strip() for line in f.readlines()]
    return model, explainer, feature_names

# ------------------------------------------------------------------------------
# Function: predict
# ------------------------------------------------------------------------------
def predict(model, input_df):
    """
    Predict the menopause stage and return probabilities.
    Args:
        model: trained sklearn pipeline
        input_df: pandas DataFrame with one row and the same features as training
    Returns:
        pred: integer class (1=Premenopause, 2=Perimenopause, 3=Postmenopause)
        proba: numpy array of probabilities for each class (length 3)
    """
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]
    return pred, proba

# ------------------------------------------------------------------------------
# Function: get_shap_values
# ------------------------------------------------------------------------------
def get_shap_values(explainer, model, input_df, feature_names):
    """
    Compute SHAP values for a single instance.
    Args:
        explainer: SHAP TreeExplainer (fitted)
        model: trained sklearn pipeline
        input_df: pandas DataFrame with one row (raw features)
        feature_names: list of feature names in correct order
    Returns:
        shap_values: list of arrays (one per class) of SHAP values for the instance
        input_shap: pandas DataFrame of transformed (standardized) features
    """
    preprocessor = model.named_steps['preprocessor']
    input_transformed = preprocessor.transform(input_df)
    input_shap = pd.DataFrame(input_transformed, columns=feature_names)
    shap_values = explainer.shap_values(input_shap, check_additivity=False)
    return shap_values, input_shap

# ------------------------------------------------------------------------------
# Function: generate_openai_explanation
# ------------------------------------------------------------------------------
def generate_openai_explanation(api_key, age, depression, fsh, age_first_period,
                                 hormone_use, estradiol, testosterone, shbg,
                                 sleep_minutes, alcohol, health, pred_label,
                                 top_features):
    """
    Generate a personalized, empathetic explanation using OpenAI GPT.
    """
    openai.api_key = api_key
    prompt = f"""
You are a compassionate medical assistant specialized in women's health. 
A patient has received a prediction of menopause stage: {pred_label}.
Here are her key health data:
- Age: {age} years
- Depression score: {depression}/27
- FSH: {fsh} U/L
- Age at first period: {age_first_period} years
- Hormone use: {hormone_use}
- Estradiol: {estradiol} pg/mL
- Testosterone: {testosterone} ng/dL
- SHBG: {shbg} nmol/L
- Sleep duration: {sleep_minutes//60}h{sleep_minutes%60}min
- Alcohol: {alcohol} drinks/day
- Self-rated health: {health}/5

The factors that most influenced this prediction (with their SHAP contribution) are:
{', '.join([f"{feat} ({val:.2f})" for feat, val in top_features])}

Please provide a clear, reassuring, and easy-to-understand explanation in English for the patient.
Explain why these factors led to this stage, and give practical tips to help her manage this phase (without replacing medical advice).
Be empathetic and base your answer on general medical knowledge.
"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert in women's health."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"OpenAI error: {e}"

# ------------------------------------------------------------------------------
# Function: chat_with_openai
# ------------------------------------------------------------------------------
def chat_with_openai(api_key, messages, context):
    """
    Handle conversational chat with OpenAI.
    """
    openai.api_key = api_key
    system_msg = f"You are a caring assistant specialized in women's health. Context: {context}"
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": system_msg}] + messages,
            temperature=0.7,
            max_tokens=400
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"
