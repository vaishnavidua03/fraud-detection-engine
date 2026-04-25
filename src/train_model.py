import pandas as pd
import joblib
import json
import os
from sklearn.metrics import classification_report, roc_auc_score
from features import build_pipeline

def train():
    print("Loading training data...")
    train_df = pd.read_csv('data/processed/transactions_train.csv')
    test_df = pd.read_csv('data/processed/transactions_test.csv')
    
    X_train = train_df.drop('is_fraud', axis=1)
    y_train = train_df['is_fraud']
    X_test = test_df.drop('is_fraud', axis=1)
    y_test = test_df['is_fraud']
    
    print("Building pipeline...")
    pipeline = build_pipeline()
    
    print("Training model... (this may take a few minutes)")
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    report = classification_report(y_test, y_pred, output_dict=True)
    auc = roc_auc_score(y_test, y_prob)
    
    print(f"\nAUC Score: {auc:.4f}")
    print(classification_report(y_test, y_pred))
    
    # Save model
    os.makedirs('models', exist_ok=True)
    joblib.dump(pipeline, 'models/fraud_pipeline.joblib')
    print("Model saved to models/fraud_pipeline.joblib")
    
    # Save metrics
    metrics = {'auc': auc, 'classification_report': report}
    os.makedirs('reports/metrics', exist_ok=True)
    with open('reports/metrics/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print("Metrics saved!")

if __name__ == "__main__":
    import sys
    sys.path.append('src')
    train()