import pandas as pd
from sklearn.model_selection import train_test_split
import os

def load_and_prepare_data():
    # Load raw data
    print("Loading raw data...")
    df = pd.read_csv('data/raw/PS_20174392719_1491204439457_log.csv')
    
    # Check shape
    print(f"Dataset shape: {df.shape}")
    print(f"Fraud cases: {df['isFraud'].sum()}")
    print(f"Fraud %: {df['isFraud'].mean()*100:.3f}%")
    
    # Keep only relevant columns
    df = df[['amount', 'type', 'oldbalanceOrg', 'newbalanceOrig',
             'oldbalanceDest', 'newbalanceDest', 'isFraud']]
    
    # Rename target column
    df = df.rename(columns={'isFraud': 'is_fraud'})
    
    # Drop missing values
    df = df.dropna()
    
    # Sample for faster development (use full data for final version)
    # df = df.sample(100000, random_state=42)  # uncomment to use subset
    
    # Stratified split — ensures fraud cases appear in both train and test
    X = df.drop('is_fraud', axis=1)
    y = df['is_fraud']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Save processed data
    train_df = X_train.copy()
    train_df['is_fraud'] = y_train.values
    test_df = X_test.copy()
    test_df['is_fraud'] = y_test.values
    
    train_df.to_csv('data/processed/transactions_train.csv', index=False)
    test_df.to_csv('data/processed/transactions_test.csv', index=False)
    
    print("Data preparation complete!")
    print(f"Train size: {len(train_df)} | Test size: {len(test_df)}")
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    load_and_prepare_data()