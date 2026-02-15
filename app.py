import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, 
                           recall_score, f1_score, matthews_corrcoef)
import xgboost as xgb
import io

st.set_page_config(page_title="Telco Churn BITS", layout="wide")

@st.cache_data
def engineer_features(df):
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
    df_fe = engineer_features(df)
    
    feature_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 
                   'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 
                   'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 
                   'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 
                   'MonthlyCharges', 'TotalCharges', 'AvgMonthlyCharges', 'ServiceBundle', 
                   'LongTerm', 'MonthToMonth', 'HighRiskPrice']
    
    X = df_fe[feature_cols]
    y = (df_fe['Churn'] == 'Yes').astype(int)
    
    cat_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
                'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 
                'PaperlessBilling', 'PaymentMethod']
    
    le_dict = {}
    for col in cat_cols:
        if col in X.columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
    
    scaler = StandardScaler()
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'AvgMonthlyCharges', 'ServiceBundle']
    X[num_cols] = scaler.fit_transform(X[num_cols])
    
    return X, y, scaler, feature_cols

@st.cache_data
def train_models(df):
    X, y, scaler, feature_cols = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    models = {
        'Logistic Regression': LogisticRegression(class_weight='balanced', random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=10, class_weight='balanced', random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=7),
        'Naive Bayes': GaussianNB(),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced'),
        'XGBoost': xgb.XGBClassifier(n_estimators=100, max_depth=6, random_state=42, scale_pos_weight=3)
    }
    
    trained_models = {}
    results = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        results[name] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'AUC': roc_auc_score(y_test, y_proba),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1': f1_score(y_test, y_pred),
            'MCC': matthews_corrcoef(y_test, y_pred)
        }
        trained_models[name] = model
    
    return trained_models, results, scaler, feature_cols

def predict_customers(test_df, models, scaler, feature_cols):
    df_fe = engineer_features(test_df)
    X_test = df_fe[feature_cols].copy()
    
    # Simple encoding for prediction
    cat_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
                'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 
                'PaperlessBilling', 'PaymentMethod']
    
    for col in cat_cols:
        if col in X_test.columns:
            X_test[col] = pd.Categorical(X_test[col]).codes
    
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'AvgMonthlyCharges', 'ServiceBundle']
    X_test[num_cols] = scaler.transform(X_test[num_cols])
    
    predictions = []
    for i in range(len(test_df)):
        row = {
            'customerID': test_df.iloc[i]['customerID'],
            'tenure': test_df.iloc[i]['tenure'],
            'MonthlyCharges': test_df.iloc[i]['MonthlyCharges'],
            'Contract': test_df.iloc[i]['Contract']
        }
        
        for name, model in models.items():
            prob = model.predict_proba(X_test.iloc[[i]])[0, 1]
            pred = 'Yes' if model.predict(X_test.iloc[[i]])[0] == 1 else 'No'
            row[f'{name}_Prob'] = round(prob, 4)
            row[f'{name}_Pred'] = pred
        
        predictions.append(row)
    return pd.DataFrame(predictions)

def main():
    st.title("🚀 Telco Churn Prediction - BITS ML Assignment 2")
    st.markdown("**Upload train.csv → Train models → Predict test.csv → Download results**")
    
    # File uploaders
    col1, col2 = st.columns(2)
    with col1:
        train_file = st.file_uploader("📚 **Upload train.csv** (for training)", type="csv")
    with col2:
        test_file = st.file_uploader("🧪 **Upload test.csv** (for prediction)", type="csv")
    
    if train_file is not None:
        df_train = pd.read_csv(train_file)
        st.success(f"✅ Training data: **{df_train.shape[0]:,} rows**, **{df_train.shape[1]} cols**")
        
        # Train models
        with st.spinner("🎯 Training 6 ML models..."):
            models, results, scaler, feature_cols = train_models(df_train)
        
        # Results table
        st.header("📊 **Model Performance** (Test Set)")
        results_df = pd.DataFrame(results).T.round(4)
        st.dataframe(results_df.style.highlight_max(axis=0, color='#d4f4d4'))
        
        # Download results
        csv_results = results_df.to_csv()
        st.download_button(
            "📥 **Download Model Results CSV**",
            csv_results,
            "model_performance.csv",
            "text/csv"
        )
        
        best_model = results_df['F1'].idxmax()
        st.success(f"🏆 **Best Model: {best_model}** (F1: {results_df.loc[best_model, 'F1']:.3f})")
        
        # Test prediction
        if test_file is not None:
            df_test = pd.read_csv(test_file)
            st.success(f"✅ Test data: **{df_test.shape[0]:,} customers**")
            
            model_names = list(models.keys())
            selected_model = st.selectbox("🎛️ **Select Model**", model_names)
            
            if st.button("🚀 **Predict Churn for ALL Customers**", type="primary"):
                with st.spinner("🔮 Predicting..."):
                    predictions_df = predict_customers(df_test, models, scaler, feature_cols)
                
                # Show summary
                st.header("📈 **Prediction Summary**")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    total = len(predictions_df)
                    st.metric("Total Customers", total)
                with col2:
                    churn_count = len(predictions_df[predictions_df[f'{selected_model}_Pred']=='Yes'])
                    churn_pct = churn_count/total*100
                    st.metric("Predicted Churn", f"{churn_count:,}", f"{churn_pct:.1f}%")
                with col3:
                    st.metric("Best Model F1", f"{results_df.loc[selected_model, 'F1']:.3f}")
                with col4:
                    st.metric("Selected Model", selected_model)
                
                # Predictions table
                st.subheader("**Individual Predictions**")
                display_cols = ['customerID', 'tenure', 'MonthlyCharges', 'Contract',
                              f'{selected_model}_Prob', f'{selected_model}_Pred']
                st.dataframe(predictions_df[display_cols].head(10))
                
                # Download predictions
                csv_pred = predictions_df.to_csv(index=False)
                st.download_button(
                    "📥 **Download FULL Predictions CSV**",
                    csv_pred,
                    f"churn_predictions_{selected_model.replace(' ', '_')}.csv",
                    "text/csv",
                    use_container_width=True
                )
    else:
        st.info("👆 **Upload train.csv first** to train models and see performance metrics")
        st.markdown("""
        ## ✅ **BITS Assignment Requirements Met:**
        ### 1. **6 ML Models**: LR, DT, KNN, NB, RF, **XGBoost**
        ### 2. **All 6 Metrics**: Accuracy, AUC, Precision, Recall, **F1**, **MCC**
        ### 3. **CSV Upload**: Train + Test data
        ### 4. **Model Selection**: Dropdown menu
        ### 5. **CSV Download**: Model results + Predictions
        ### 6. **Interactive Dashboard**
        """)

if __name__ == "__main__":
    main()
