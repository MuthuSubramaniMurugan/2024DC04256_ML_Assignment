import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, 
                           recall_score, f1_score, matthews_corrcoef, confusion_matrix)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import io

st.set_page_config(page_title="Telco Churn - BITS Assignment", layout="wide")

@st.cache_data
def engineer_features(df):
    """Telco churn feature engineering"""
    df = df.copy()
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(' ', '0'), errors='coerce').fillna(0)
    df['AvgMonthlyCharges'] = df['TotalCharges'] / (df['tenure'] + 1)
    df['LongTerm'] = (df['tenure'] > 24).astype(int)
    df['MonthToMonth'] = (df['Contract'] == 'Month-to-month').astype(int)
    services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                'StreamingTV', 'StreamingMovies']
    df['ServiceBundle'] = df[services].apply(lambda x: (x == 'Yes').sum(), axis=1)
    df['HighRiskPrice'] = ((df['MonthlyCharges'] > 80) & (df['tenure'] < 12)).astype(int)
    return df

def preprocess_data(df):
    """Complete preprocessing pipeline"""
    df_fe = engineer_features(df)
    
    feature_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService',
                   'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
                   'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
                   'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 
                   'TotalCharges', 'AvgMonthlyCharges', 'ServiceBundle', 'LongTerm', 
                   'MonthToMonth', 'HighRiskPrice']
    
    X = df_fe[feature_cols]
    y = (df_fe['Churn'] == 'Yes').astype(int)
    
    # Encode categoricals
    cat_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
                'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 
                'PaperlessBilling', 'PaymentMethod', 'LongTerm', 'MonthToMonth', 'HighRiskPrice']
    
    le_dict = {}
    for col in cat_cols:
        if col in X.columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            le_dict[col] = le
    
    scaler = StandardScaler()
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'AvgMonthlyCharges', 'ServiceBundle']
    X[num_cols] = scaler.fit_transform(X[num_cols])
    
    return X, y, scaler, le_dict, feature_cols

@st.cache_data
def train_models(_df):
    """Train all models - cached for speed"""
    X, y, scaler, le_dict, feature_cols = preprocess_data(_df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # SMOTE
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    # Models with tuned params
    models = {
        'Logistic Regression': LogisticRegression(class_weight='balanced', random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=10, min_samples_split=20, class_weight='balanced', random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=7, weights='distance'),
        'Naive Bayes': GaussianNB(),
        'Random Forest': RandomForestClassifier(n_estimators=300, max_depth=15, min_samples_split=5, 
                                              min_samples_leaf=2, max_features='sqrt', random_state=42, class_weight='balanced'),
        'XGBoost': xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.9, 
                                   colsample_bytree=0.9, random_state=42, scale_pos_weight=3)
    }
    
    trained_models 
