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

st.set_page_config(page_title="Fraud Detection Engine", layout="wide")
st.title("🔍 Financial Fraud Risk Engine")
st.markdown("Upload transaction data to detect fraud using ML")

def generate_sample_data():
    """Generate synthetic fraud data for training on cloud"""
    np.random.seed(42)
    n = 10000
    
    types = np.random.choice(
        ['CASH_OUT', 'PAYMENT', 'CASH_IN', 'TRANSFER', 'DEBIT'],
        n, p=[0.35, 0.34, 0.22, 0.08, 0.01]
    )
    amount = np.random.exponential(scale=50000, size=n)
    oldbalanceOrg = np.random.exponential(scale=100000, size=n)
    newbalanceOrig = np.maximum(0, oldbalanceOrg - amount)
    oldbalanceDest = np.random.exponential(scale=100000, size=n)
    newbalanceDest = oldbalanceDest + amount
    
    # Fraud logic — large transfers with zero new balance
    is_fraud = (
        (amount > 200000) &
        (newbalanceOrig == 0) &
        (np.isin(types, ['CASH_OUT', 'TRANSFER']))
    ).astype(int)
    
    df = pd.DataFrame({
        'type': types,
        'amount': amount,
        'oldbalanceOrg': oldbalanceOrg,
        'newbalanceOrig': newbalanceOrig,
        'oldbalanceDest': oldbalanceDest,
        'newbalanceDest': newbalanceDest,
        'is_fraud': is_fraud
    })
    return df

@st.cache_resource
def load_or_train_model():
    if os.path.exists('models/fraud_pipeline.joblib'):
        pipeline = joblib.load('models/fraud_pipeline.joblib')
        with open('models/threshold.json') as f:
            threshold = json.load(f)['threshold']
        return pipeline, threshold

    with st.spinner("Training model for first time... (1-2 minutes)"):
        df = generate_sample_data()
        
        X = df.drop('is_fraud', axis=1)
        y = df['is_fraud']
        
        preprocessor = ColumnTransformer(transformers=[
            ('num', StandardScaler(),
             ['amount', 'oldbalanceOrg', 'newbalanceOrig',
              'oldbalanceDest', 'newbalanceDest']),
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['type'])
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
st.sidebar.markdown("Lower = catch more fraud but more false alarms")

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
        st.info("Run locally to see full model performance charts")

else:
    st.info("👆 Upload a CSV file to get started")
    st.markdown("Expected columns: `type`, `amount`, `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`")