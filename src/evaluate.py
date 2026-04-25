import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
from sklearn.metrics import (precision_recall_curve, roc_curve, 
                              auc, confusion_matrix, ConfusionMatrixDisplay)
import os, sys
sys.path.append('src')

def evaluate():
    # Load model and test data
    pipeline = joblib.load('models/fraud_pipeline.joblib')
    test_df = pd.read_csv('data/processed/transactions_test.csv')
    X_test = test_df.drop('is_fraud', axis=1)
    y_test = test_df['is_fraud']
    
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    os.makedirs('reports/figures', exist_ok=True)
    
    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig('reports/figures/roc_curve.png')
    plt.close()
    print("ROC curve saved")
    
    # 2. Precision-Recall Curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.savefig('reports/figures/pr_curve.png')
    plt.close()
    print("PR curve saved")
    
    # 3. COST-SENSITIVE THRESHOLD OPTIMIZATION
    print("\nOptimizing threshold based on business cost...")
    best_threshold = 0.5
    best_cost = float('inf')
    threshold_results = []
    
    for threshold in np.arange(0.05, 0.95, 0.05):
        y_pred = (y_prob >= threshold).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Business cost: missing fraud = 10x worse than false alarm
        cost = (10 * fn) + (1 * fp)
        threshold_results.append({
            'threshold': round(float(threshold), 2),
            'cost': int(cost),
            'fn': int(fn),
            'fp': int(fp),
            'tp': int(tp)
        })
        
        if cost < best_cost:
            best_cost = cost
            best_threshold = threshold
    
    print(f"Best threshold: {best_threshold:.2f} | Business cost: {best_cost}")
    
    # Save best threshold — fixed int64 serialization issue
    with open('models/threshold.json', 'w') as f:
        json.dump({
            'threshold': float(best_threshold), 
            'cost': int(best_cost)
        }, f, indent=2)
    
    # Save all threshold results
    with open('reports/metrics/threshold_search.json', 'w') as f:
        json.dump(threshold_results, f, indent=2)
    
    # 4. Confusion Matrix at best threshold
    y_pred_best = (y_prob >= best_threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred_best)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f'Confusion Matrix (threshold={best_threshold:.2f})')
    plt.savefig('reports/figures/confusion_matrix.png')
    plt.close()
    print("Confusion matrix saved")
    
    print("\nEvaluation complete!")

if __name__ == "__main__":
    evaluate()