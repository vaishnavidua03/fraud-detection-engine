from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

def build_pipeline():
    # Define which columns are categorical vs numerical
    categorical_features = ['type']
    numerical_features = ['amount', 'oldbalanceOrg', 'newbalanceOrig',
                          'oldbalanceDest', 'newbalanceDest']
    
    # Preprocessing for numerical: scale to same range
    numerical_transformer = StandardScaler()
    
    # Preprocessing for categorical: convert text to numbers
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    
    # Combine both preprocessors
    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    
    # Full pipeline: preprocess → model
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced',  # handles imbalanced data
            n_jobs=-1  # uses all CPU cores
        ))
    ])
    
    return pipeline
