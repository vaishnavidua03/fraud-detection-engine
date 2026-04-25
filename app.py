import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Fraud Detection Engine", layout="wide")
st.title("🔍 Financial Fraud Risk Engine")
st.markdown("Upload transaction data to detect fraud using ML")

SAMPLE_DATA_URL = "https://raw.githubusercontent.com/vaishnavidua03/fraud-detection-engine/main/data/sample/sample_transactions.csv"

@st.cache_resource
def load_or_train_model():
    if os.path.exists('models/fraud_pipeline.joblib'):
        pipeline = joblib.load('models/fraud_pipeline.joblib')
        with open('models/threshold.json') as f:
            threshold = json.load(f)['threshold']
        return pipeline, threshold

    st.info("Training model for first time... this takes 2-3 minutes")
    
    # Download sample data from GitHub
    df = pd.read_csv(SAMPLE_DATA_URL)
    
    X = df.drop('is_fraud', axis=1)
    y = df['is_fraud']
    
    categorical_features = ['type']
    numerical_features = ['amount', 'oldbalanceOrg', 'newbalanceOrig',
                          'oldbalanceDest', 'newbalanceDest']
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=50,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        ))
    ])
    
    pipeline.fit(X, y)
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(pipeline, 'models/fraud_pipeline.joblib')
    
    threshold = 0.1
    with open('models/threshold.json', 'w') as f:
        json.dump({'threshold': threshold, 'cost': 0}, f)
    
    return pipeline, threshold

pipeline, threshold = load_or_train_model()

st.sidebar.header("Settings")
custom_threshold = st.sidebar.slider(
    "Fraud Decision Threshold",
    min_value=0.1, max_value=0.9,
    value=float(threshold), step=0.05
)
st.sidebar.markdown(f"**Current threshold:** {custom_threshold}")
st.sidebar.markdown("Lower threshold = catch more fraud but more false alarms")

uploaded_file = st.file_uploader("Upload CSV of transactions", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if 'is_fraud' in df.columns:
        df = df.drop('is_fraud', axis=1)

    st.subheader("Raw Data Preview")
    st.dataframe(df.head())

    with st.spinner("Scoring transactions..."):
        fraud_prob = pipeline.predict_proba(df)[:, 1]
        fraud_flag = (fraud_prob >= custom_threshold).astype(int)

    df['fraud_probability'] = fraud_prob
    df['fraud_flag'] = fraud_flag

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transactions", len(df))
    col2.metric("Flagged as Fraud", int(fraud_flag.sum()))
    col3.metric("Fraud Rate", f"{fraud_flag.mean()*100:.2f}%")
    col4.metric("Avg Fraud Probability", f"{fraud_prob.mean():.4f}")

    st.subheader("🚨 High Risk Transactions")
    high_risk = df[df['fraud_flag'] == 1].sort_values(
        'fraud_probability', ascending=False
    )
    st.dataframe(high_risk.head(20))

    st.subheader("Fraud Probability Distribution")
    fig, ax = plt.subplots()
    ax.hist(fraud_prob, bins=50, color='steelblue', edgecolor='white')
    ax.axvline(custom_threshold, color='red', linestyle='--',
               label=f'Threshold: {custom_threshold}')
    ax.set_xlabel('Fraud Probability')
    ax.set_ylabel('Count')
    ax.legend()
    st.pyplot(fig)

    st.subheader("Download Results")
    csv = df.to_csv(index=False)
    st.download_button(
        "Download Scored Transactions",
        csv,
        "scored_transactions.csv",
        "text/csv"
    )

    st.subheader("Model Performance")
    col1, col2 = st.columns(2)
    try:
        col1.image('reports/figures/roc_curve.png', caption='ROC Curve')
        col2.image('reports/figures/pr_curve.png', caption='Precision-Recall Curve')
        st.image('reports/figures/shap_summary.png', caption='SHAP Feature Importance')
        st.image('reports/figures/confusion_matrix.png', caption='Confusion Matrix')
    except:
        st.info("Model performance charts available when running locally")

else:
    st.info("👆 Upload a CSV file to get started")
    st.markdown("Expected columns: `type`, `amount`, `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`")
    