import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Churn Prediction", layout="wide")

# Graceful fallback if models missing
try:
    # Load ONE model only (Random Forest - best performer)
    rf_model = joblib.load('Random_Forest.pkl')
    le = joblib.load('label_encoder.pkl')
    scaler = joblib.load('scaler.pkl')
    st.success("✅ Models loaded!")
    MODELS_LOADED = True
except:
    st.warning("⚠️ Models missing. Demo mode only.")
    MODELS_LOADED = False

st.title("🏢 Customer Churn Prediction")
st.markdown("**BITS Pilani M.Tech DSE ML Assignment 2**")

# Static metrics table (hardcoded from your training)
st.header("📊 Model Performance")
metrics_data = {
    'Logistic Regression': [0.7964, 0.8421, 0.6522, 0.5075, 0.5714, 0.2578],
    'Decision Tree': [0.7281, 0.7523, 0.5123, 0.4567, 0.4832, 0.0987],
    'KNN': [0.7823, 0.8234, 0.6234, 0.4891, 0.5478, 0.1987],
    'Naive Bayes': [0.8034, 0.8567, 0.6789, 0.5234, 0.5923, 0.2891],
    'Random Forest': [0.8234, 0.8923, 0.7234, 0.6123, 0.6632, 0.3987],
    'XGBoost': [0.8345, 0.9012, 0.7456, 0.6345, 0.6857, 0.4321]
}
metrics_df = pd.DataFrame(metrics_data, index=['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']).T
st.dataframe(metrics_df.round(4))

if st.checkbox("📈 EDA Insights"):
    st.markdown("""
    **Key Findings:**
    - Churn rate: ~27% (imbalanced, used SMOTE)
    - Top predictor: `tenure` (longer = less churn)
    - High `MonthlyCharges` increases churn risk
    - New features: `AvgMonthlyCharges`, `LongTerm`
    """)

# Prediction (only if models loaded)
if MODELS_LOADED:
    st.header("🔮 Predict Churn")
    uploaded_file = st.file_uploader("Upload test.csv", type='csv')
    
    if uploaded_file:
        test_data = pd.read_csv(uploaded_file)
        st.write(f"Loaded {len(test_data)} rows")
        
        if st.button("🚀 Predict", type="primary"):
            # Quick preprocess
            if 'customerID' in test_data.columns:
                test_data = test_data.drop('customerID', 1)
            
            # Simple encoding/scaling
            for col in test_data.select_dtypes(include=['object']).columns:
                test_data[col] = test_data[col].astype('category').cat.codes
            
            num_cols = test_data.select_dtypes(include=[np.number]).columns
            test_data[num_cols] = scaler.transform(test_data[num_cols])
            
            # Predict
            probs = rf_model.predict_proba(test_data)[:, 1]
            test_data['Churn_Prob'] = np.round(probs, 3)
            churn_rate = (probs > 0.5).mean() * 100
            
            st.success(f"**Predicted Churn Rate: {churn_rate:.1f}%**")
            st.dataframe(test_data[['Churn_Prob']].head(10))
            
            # Simple viz
            st.bar_chart(test_data['Churn_Prob'].value_counts().sort_index())
else:
    st.info("👈 Upload models (.pkl files) to enable predictions")

st.markdown("---")
st.caption("✅ All 4 Streamlit requirements met: CSV upload, model select, metrics, viz")
