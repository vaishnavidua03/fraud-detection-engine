# 🔍 Financial Fraud Risk Engine

![Python](https://img.shields.io/badge/Python-3.14-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-Optimized-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red)
![AUC](https://img.shields.io/badge/AUC-0.9997-brightgreen)

## 📌 Business Problem

Financial fraud costs billions annually. In India alone, UPI fraud resulted in 
losses of ₹1,087 crore in 2024 (NPCI Data). Traditional rule-based systems 
miss complex fraud patterns. This project builds an ML-powered fraud detection 
system that flags suspicious transactions in real-time with explainable predictions.

## 🏗️ Architecture

    Raw Transactions (6.3M PaySim)
            ↓
    Data Preparation (data_prep.py)
      - Missing value handling
      - UPI context mapping
      - Stratified train/test split
            ↓
    Feature Engineering (features.py)
      - StandardScaler (numerical)
      - OneHotEncoder (categorical)
            ↓
    Bayesian Hyperparameter Tuning (Optuna - 10 trials)
            ↓
    XGBoost + SMOTE Pipeline
      - scale_pos_weight=100
      - SMOTE sampling_strategy=0.1
            ↓
    Cost-Sensitive Threshold Optimization
      - FN cost = 10x FP cost
      - Best threshold = 0.10
            ↓
    SHAP Explainability
            ↓
    Streamlit Dashboard (Deployed)

    ## 🔬 Technical Stack

| Component | Technology |
|---|---|
| ML Model | XGBoost + SMOTE |
| Hyperparameter Tuning | Bayesian Optimization (Optuna) |
| Explainability | SHAP |
| Data Pipeline | Scikit-learn Pipeline |
| Dashboard | Streamlit |
| Data | PaySim (6.3M transactions) |

## 📊 Model Performance

| Metric | Value |
|---|---|
| AUC Score | **0.9997** |
| Dataset Size | 6.3M transactions |
| Fraud Rate | 0.129% |
| Threshold Strategy | Cost-sensitive (FN = 10x FP) |

## 💡 Key Business Insights

- Model flags **0.17%** of transactions as fraudulent
- Cost-sensitive threshold minimizes business loss from missed fraud
- TRANSFER and CASH_OUT transaction types show highest fraud probability
- Zero-balance transactions after transfer = highest fraud signal

## 🚀 Live Demo

👉 [fraud-detection-engine-katxwou9uigkxdtqvdd8qc.streamlit.app](https://fraud-detection-engine-katxwou9uigkxdtqvdd8qc.streamlit.app)

## 📁 Project Structure

raud-detection-engine/
├── data/
│   ├── raw/          # PaySim dataset (local only)
│   └── processed/    # Train/test splits (local only)
├── models/           # Saved model + threshold + best params
├── reports/
│   ├── figures/      # ROC, PR, SHAP, Confusion Matrix
│   └── metrics/      # AUC, classification report
├── src/
│   ├── data_prep.py          # Data cleaning + UPI mapping
│   ├── features.py           # ML pipeline
│   ├── train_model.py        # XGBoost + Optuna tuning
│   ├── evaluate.py           # Cost-sensitive evaluation
│   ├── explain.py            # SHAP explainability
│   └── score_new_transactions.py  # Batch scoring
├── app.py            # Streamlit dashboard
└── requirements.txt

## ⚙️ How to Run Locally

```bash
# Clone repo
git clone https://github.com/vaishnavidua03/fraud-detection-engine.git
cd fraud-detection-engine

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download PaySim dataset from Kaggle
# Place CSV in data/raw/

# Run pipeline
python src/data_prep.py
python src/train_model.py
python src/evaluate.py
python src/explain.py

# Launch dashboard
python -m streamlit run app.py
```

## 🔍 UPI Context Mapping

| PaySim Type | UPI Equivalent |
|---|---|
| TRANSFER | UPI P2P Transfer |
| CASH_OUT | UPI Merchant Payment |
| PAYMENT | UPI Bill Payment |
| CASH_IN | UPI Wallet Topup |
| DEBIT | UPI Auto Debit |

## 📋 Regulatory Context

This project follows financial data best practices aligned with:
- **PCI DSS** — Payment Card Industry Data Security Standard
- **RBI Fraud Reporting Guidelines** — India's central bank fraud framework
- **PSD2** — Payment Services Directive (explainability requirements)

## 👩‍💻 Author

**Vaishnavi Dua** | B.Tech AI & ML 
