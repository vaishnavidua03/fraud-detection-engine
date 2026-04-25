import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np
import os, sys
sys.path.append('src')

def explain():
    print("Loading model and data...")
    pipeline = joblib.load('models/fraud_pipeline.joblib')
    test_df = pd.read_csv('data/processed/transactions_test.csv')
    
    # Use small sample for SHAP
    sample = test_df.sample(500, random_state=42)
    X_sample = sample.drop('is_fraud', axis=1)
    
    # Get preprocessor and classifier separately
    preprocessor = pipeline.named_steps['preprocessor']
    classifier = pipeline.named_steps['classifier']
    
    # Transform data first
    X_transformed = preprocessor.transform(X_sample)
    
    # Convert to dense array if sparse
    if hasattr(X_transformed, 'toarray'):
        X_transformed = X_transformed.toarray()
    
    print("Calculating SHAP values... (takes a few minutes)")
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_transformed)
    
    os.makedirs('reports/figures', exist_ok=True)
    
    # Handle both single and multi-output SHAP values
    if isinstance(shap_values, list):
        sv = shap_values[1]  # fraud class
    else:
        sv = shap_values
    
    # Get feature names after transformation
    try:
        cat_features = preprocessor.named_transformers_['cat']\
            .get_feature_names_out(['type'])
        num_features = ['amount', 'oldbalanceOrg', 'newbalanceOrig',
                       'oldbalanceDest', 'newbalanceDest']
        feature_names = list(num_features) + list(cat_features)
    except:
        feature_names = [f'feature_{i}' for i in range(X_transformed.shape[1])]
    
    # Global SHAP summary plot
    plt.figure()
    shap.summary_plot(
        sv,
        X_transformed,
        feature_names=feature_names,
        show=False
    )
    plt.tight_layout()
    plt.savefig('reports/figures/shap_summary.png', bbox_inches='tight')
    plt.close()
    print("SHAP summary plot saved!")

if __name__ == "__main__":
    explain()