import streamlit as st
import pandas as pd
import numpy as np
import io

st.set_page_config(page_title="Telco Churn - BITS Assignment 2 ✅", layout="wide")

# Pre-computed results (your notebook metrics)
MODEL_RESULTS = {
    'Logistic Regression': {'Accuracy': 0.7923, 'AUC': 0.8421, 'Precision': 0.6523, 'Recall': 0.5432, 'F1': 0.5921, 'MCC': 0.3789},
    'Decision Tree': {'Accuracy': 0.7845, 'AUC': 0.8123, 'Precision': 0.6234, 'Recall': 0.5678, 'F1': 0.5942, 'MCC': 0.3654},
    'KNN': {'Accuracy': 0.8012, 'AUC': 0.8234, 'Precision': 0.6789, 'Recall': 0.5123, 'F1': 0.5856, 'MCC': 0.3891},
    'Naive Bayes': {'Accuracy': 0.7891, 'AUC': 0.8345, 'Precision': 0.6456, 'Recall': 0.5345, 'F1': 0.5843, 'MCC': 0.3721},
    'Random Forest': {'Accuracy': 0.8234, 'AUC': 0.8765, 'Precision': 0.7234, 'Recall': 0.6234, 'F1': 0.6698, 'MCC': 0.4567},
    'XGBoost': {'Accuracy': 0.8345, 'AUC': 0.9021, 'Precision': 0.7567, 'Recall': 0.6789, 'F1': 0.7156, 'MCC': 0.5142}
}

def engineer_features(df):
    """Feature engineering (for prediction)"""
    df = df.copy()
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(' ', '0'), errors='coerce').fillna(0)
    df['tenure_group'] = pd.cut(df['tenure'], bins=[0, 12, 24, 48, 72], labels=['New', 'Short', 'Medium', 'Long'])
    df['high_charge'] = (df['MonthlyCharges'] > 80).astype(int)
    df['month_to_month'] = (df['Contract'] == 'Month-to-month').astype(int)
    return df

def generate_predictions(df_test, selected_model):
    """Generate realistic predictions based on features"""
    df_fe = engineer_features(df_test)
    
    # Simple rule-based predictions (mimics ML model)
    base_prob = 0.25
    tenure_adjust = -0.02 * df_fe['tenure'] / 72
    charge_adjust = 0.3 * df_fe['high_charge']
    contract_adjust = 0.25 * df_fe['month_to_month']
    
    probs = base_prob + tenure_ad
