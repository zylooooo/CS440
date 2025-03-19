import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import GridSearchCV
import os
import pickle
from datetime import datetime

# Create directories for saving results if they don't exist
os.makedirs("models", exist_ok=True)
os.makedirs("data/plots", exist_ok=True)
os.makedirs("data/test_results", exist_ok=True)

# Function to load and preprocess the data
def load_data(file_path):
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Data loaded with shape: {df.shape}")
    return df

# Data exploration function
def explore_data(df, save_prefix=""):
    print("Exploring data...")
    
    # Basic information
    print("\nData types and missing values:")
    print(df.info())
    
    print("\nData statistics:")
    print(df.describe())
    
    # Checking class distribution
    fraud_count = df['is_fraud'].value_counts()
    print("\nClass distribution:")
    print(fraud_count)
    
    # Plot and save class distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x='is_fraud', data=df)
    plt.title('Class Distribution (Fraud vs Non-Fraud)')
    plt.xlabel('Fraud (1) vs Non-Fraud (0)')
    plt.ylabel('Count')
    plt.savefig(f"data/plots/{save_prefix}class_distribution.png")
    
    # Plot transaction amount distribution
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(df[df['is_fraud']==0]['amt'], kde=True)
    plt.title('Amount Distribution - Non-Fraud')
    plt.xlabel('Transaction Amount')
    
    plt.subplot(1, 2, 2)
    sns.histplot(df[df['is_fraud']==1]['amt'], kde=True)
    plt.title('Amount Distribution - Fraud')
    plt.xlabel('Transaction Amount')
    plt.tight_layout()
    plt.savefig(f"data/plots/{save_prefix}amount_distribution.png")
    
    # Category distribution for fraud transactions
    plt.figure(figsize=(12, 6))
    fraud_by_category = df[df['is_fraud']==1]['category'].value_counts()
    sns.barplot(x=fraud_by_category.index, y=fraud_by_category.values)
    plt.title('Fraud Transactions by Category')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f"data/plots/{save_prefix}fraud_by_category.png")

    return fraud_count

# Feature engineering and preprocessing
def preprocess_data(df):
    print("Preprocessing data...")
    
    # Extract datetime features
    df['trans_date'] = pd.to_datetime(df['trans_date_trans_time'])
    df['hour'] = df['trans_date'].dt.hour
    df['day_of_week'] = df['trans_date'].dt.dayofweek
    df['month'] = df['trans_date'].dt.month
    
    # Calculate distance between card holder and merchant
    df['distance'] = np.sqrt((df['lat'] - df['merch_lat'])**2 + (df['long'] - df['merch_long'])**2)
    
    # Select features for the model
    categorical_features = ['category', 'gender', 'state']
    numerical_features = ['amt', 'hour', 'day_of_week', 'month', 'distance', 'city_pop']
    
    # Get X (features) and y (target)
    X = df[categorical_features + numerical_features].copy()
    y = df['is_fraud']
    
    return X, y, categorical_features, numerical_features

# Build and train model
def build_model(X_train, y_train, categorical_features, numerical_features):
    print("Building model...")
    
    # Define preprocessing for categorical and numerical features
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Create the pipeline with preprocessor and classifier
    # Use a simpler model with predefined parameters instead of grid search
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    print("Training model...")
    model.fit(X_train, y_train)
    
    return model

# Evaluate model
def evaluate_model(model, X_test, y_test):
    print("Evaluating model...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Generate classification report
    report = classification_report(y_test, y_pred)
    print("\nClassification Report:")
    print(report)
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig("data/plots/confusion_matrix.png")
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig("data/plots/roc_curve.png")
    
    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.savefig("data/plots/precision_recall_curve.png")
    
    # Save evaluation results
    with open("data/test_results/evaluation_report.txt", "w") as f:
        f.write(f"Evaluation Results - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write(f"\nROC AUC: {roc_auc:.4f}\n")
    
    return report, roc_auc

# Main function
def main():
    print("Starting fraud detection modeling process...")
    
    # Load data
    # Check if sampled data exists, otherwise use full datasets
    train_file = "data/fraudTrain_sample.csv" if os.path.exists("data/fraudTrain_sample.csv") else "data/fraudTrain.csv"
    test_file = "data/fraudTest_sample.csv" if os.path.exists("data/fraudTest_sample.csv") else "data/fraudTest.csv"
    
    train_df = load_data(train_file)
    test_df = load_data(test_file)
    
    # Explore data
    explore_data(train_df, save_prefix="train_")
    explore_data(test_df, save_prefix="test_")
    
    # Preprocess data
    X_train, y_train, categorical_features, numerical_features = preprocess_data(train_df)
    X_test, y_test, _, _ = preprocess_data(test_df)
    
    # Build and train model
    model = build_model(X_train, y_train, categorical_features, numerical_features)
    
    # Save model
    print("Saving model...")
    model_filename = "models/fraud_detection_model.pkl"
    with open(model_filename, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model saved to {model_filename}")
    
    # Evaluate model
    report, roc_auc = evaluate_model(model, X_test, y_test)
    
    print("\nFraud detection model training complete.")
    print(f"Model ROC AUC: {roc_auc:.4f}")
    print("Model and evaluation results saved.")

    try:
        import shap
        print("Creating SHAP explainer for model interpretability...")
        # Create a small representative sample of data for the explainer
        sample_size = min(1000, len(X_train))
        X_sample = X_train.sample(sample_size, random_state=42)
        
        # Create explainer
        explainer = shap.TreeExplainer(model.named_steps['classifier'])
        
        # Save the explainer
        with open("models/shap_explainer.pkl", "wb") as f:
            pickle.dump(explainer, f)
        print("SHAP explainer saved successfully.")
    except ImportError:
        print("SHAP library not available. Skipping explainer creation.")
    except Exception as e:
        print(f"Error creating SHAP explainer: {str(e)}")

if __name__ == "__main__":
    main() 