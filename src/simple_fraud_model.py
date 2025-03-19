import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import pickle
import os
from datetime import datetime

# Create directories for saving results
os.makedirs("models", exist_ok=True)
os.makedirs("data/test_results", exist_ok=True)

def load_data(file_path, nrows=None, selected_features=None):
    """Load and preprocess data with option to limit rows and columns"""
    print(f"Loading data from {file_path}...")
    
    # If selected features provided, use them to reduce memory usage
    if selected_features:
        df = pd.read_csv(file_path, nrows=nrows, usecols=selected_features)
    else:
        df = pd.read_csv(file_path, nrows=nrows)
        
    print(f"Data loaded with shape: {df.shape}")
    return df

def preprocess_features(df, categories=None, mode='fit'):
    """Extract and preprocess essential features"""
    print(f"Preprocessing features... (mode: {mode})")
    
    # Extract features from datetime
    df['trans_date'] = pd.to_datetime(df['trans_date_trans_time'])
    df['hour'] = df['trans_date'].dt.hour
    df['day_of_week'] = df['trans_date'].dt.dayofweek
    
    # Calculate distance between card holder and merchant
    df['distance'] = np.sqrt((df['lat'] - df['merch_lat'])**2 + (df['long'] - df['merch_long'])**2)
    
    # Process categorical features
    if 'category' in df.columns:
        if mode == 'fit':
            # In fit mode, we determine the top categories
            top_categories = df['category'].value_counts().head(10).index.tolist()
            # Create category dummies
            for cat in top_categories:
                df[f'cat_{cat}'] = (df['category'] == cat).astype(int)
            return_cats = top_categories
        else:
            # In transform mode, we use the categories provided
            if categories is None:
                raise ValueError("Categories must be provided in transform mode")
            # Create the same category dummies as in fit
            for cat in categories:
                df[f'cat_{cat}'] = (df['category'] == cat).astype(int)
            return_cats = categories
    else:
        return_cats = []
    
    # Select numerical features
    numerical_features = ['amt', 'hour', 'day_of_week', 'distance']
    
    # Combine all feature columns
    feature_cols = numerical_features.copy()
    category_cols = [f'cat_{cat}' for cat in return_cats]
    feature_cols.extend(category_cols)
    
    # Scale numerical features
    scaler = StandardScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])
    
    # Get X and y
    X = df[feature_cols]
    y = df['is_fraud'] if 'is_fraud' in df.columns else None
    
    return X, y, feature_cols, return_cats

def train_model(X_train, y_train):
    """Train a simple Random Forest model"""
    print("Training model...")
    
    # Create model with modest parameters
    model = RandomForestClassifier(
        n_estimators=50,  # Reduced from 100
        max_depth=8,      # Reduced from 10
        min_samples_split=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    # Train model
    model.fit(X_train, y_train)
    print("Model training complete.")
    
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model and save results"""
    print("Evaluating model...")
    
    # Predict on test data
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    print("\nClassification Report:")
    print(report)
    
    print("\nConfusion Matrix:")
    print(cm)
    
    print(f"\nROC AUC: {roc_auc:.4f}")
    
    # Save evaluation results
    with open("data/test_results/simple_model_evaluation.txt", "w") as f:
        f.write(f"Evaluation Results - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write(f"\nROC AUC: {roc_auc:.4f}\n")
    
    return report, roc_auc

def main():
    print("Starting simplified fraud detection model training...")
    
    # Select only necessary features to reduce memory usage
    selected_features = [
        'trans_date_trans_time', 'category', 'amt', 
        'lat', 'long', 'merch_lat', 'merch_long', 'is_fraud'
    ]
    
    # Load data with selected features
    train_df = load_data("data/fraudTrain_sample.csv", selected_features=selected_features)
    test_df = load_data("data/fraudTest_sample.csv", selected_features=selected_features)
    
    # Process training features - this determines our feature set
    X_train, y_train, feature_cols, categories = preprocess_features(train_df, mode='fit')
    
    # Process test features - using the categories from training
    X_test, y_test, _, _ = preprocess_features(test_df, categories=categories, mode='transform')
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Save model
    print("Saving model...")
    with open("models/simple_fraud_model.pkl", "wb") as f:
        pickle.dump({
            'model': model,
            'features': feature_cols,
            'categories': categories
        }, f)
    
    # Evaluate
    report, roc_auc = evaluate_model(model, X_test, y_test)
    
    print("\nSimple fraud detection model training completed.")
    print(f"Model ROC AUC: {roc_auc:.4f}")
    print("Model and evaluation results saved.")

if __name__ == "__main__":
    main() 