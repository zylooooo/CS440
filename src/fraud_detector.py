import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime

class FraudDetector:
    """
    A class to detect credit card fraud using a pre-trained model.
    """
    
    def __init__(self, model_path="models/fraud_detection_model.pkl"):
        """
        Initialize the fraud detector with a pre-trained model.
        
        Parameters:
        -----------
        model_path : str
            Path to the pickled model file
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        print(f"Loading model from {model_path}...")
        with open(model_path, 'rb') as file:
            self.model = pickle.load(file)
        print("Model loaded successfully.")
        
        # Define the features used by the model
        self.categorical_features = ['category', 'gender', 'state']
        self.numerical_features = ['amt', 'hour', 'day_of_week', 'month', 'distance', 'city_pop']
        
    def preprocess_transaction(self, transaction):
        """
        Preprocess a single transaction or a DataFrame of transactions.
        
        Parameters:
        -----------
        transaction : dict or pd.DataFrame
            Transaction data with required fields
            
        Returns:
        --------
        pd.DataFrame
            Preprocessed features ready for prediction
        """
        if isinstance(transaction, dict):
            df = pd.DataFrame([transaction])
        else:
            df = transaction.copy()
        
        # Extract datetime features
        if 'trans_date_trans_time' in df.columns:
            df['trans_date'] = pd.to_datetime(df['trans_date_trans_time'])
            df['hour'] = df['trans_date'].dt.hour
            df['day_of_week'] = df['trans_date'].dt.dayofweek
            df['month'] = df['trans_date'].dt.month
        elif all(x in df.columns for x in ['hour', 'day_of_week', 'month']):
            pass  # Features already exist
        else:
            # Use current time if not provided
            now = datetime.now()
            df['hour'] = now.hour
            df['day_of_week'] = now.weekday()
            df['month'] = now.month
        
        # Calculate distance if coordinates are available
        if all(x in df.columns for x in ['lat', 'long', 'merch_lat', 'merch_long']):
            df['distance'] = np.sqrt((df['lat'] - df['merch_lat'])**2 + (df['long'] - df['merch_long'])**2)
        elif 'distance' not in df.columns:
            df['distance'] = 0  # Default value if not provided
        
        # Ensure all required features are present
        for feature in self.categorical_features + self.numerical_features:
            if feature not in df.columns:
                raise ValueError(f"Missing required feature: {feature}")
        
        # Select only the features needed for prediction
        X = df[self.categorical_features + self.numerical_features]
        
        return X
    
    def predict(self, transaction, threshold=0.5):
        """
        Predict if a transaction is fraudulent.
        
        Parameters:
        -----------
        transaction : dict or pd.DataFrame
            Transaction data with required fields
        threshold : float, default=0.5
            Probability threshold for classifying as fraud
            
        Returns:
        --------
        dict or pd.DataFrame
            Contains 'fraud_prediction' (bool) and 'fraud_probability' (float)
        """
        try:
            X = self.preprocess_transaction(transaction)
            
            # Get fraud probability
            fraud_proba = self.model.predict_proba(X)[:, 1]
            
            # Make binary prediction based on threshold
            fraud_prediction = fraud_proba >= threshold
            
            # Prepare results
            if isinstance(transaction, dict):
                return {
                    'fraud_prediction': bool(fraud_prediction[0]),
                    'fraud_probability': float(fraud_proba[0]),
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
            else:
                result_df = transaction.copy()
                result_df['fraud_prediction'] = fraud_prediction
                result_df['fraud_probability'] = fraud_proba
                result_df['prediction_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                return result_df
            
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            if isinstance(transaction, dict):
                return {
                    'error': str(e),
                    'fraud_prediction': None,
                    'fraud_probability': None
                }
            else:
                raise e

def main():
    """
    Demo function to show how to use the FraudDetector class.
    """
    # Create an instance of the fraud detector
    detector = FraudDetector()
    
    # Example transaction (modify fields as needed)
    example_transaction = {
        'category': 'shopping',
        'gender': 'M',
        'state': 'CA',
        'amt': 120.50,
        'hour': 14,
        'day_of_week': 2,
        'month': 6,
        'distance': 5.2,
        'city_pop': 500000
    }
    
    # Make prediction
    result = detector.predict(example_transaction)
    
    # Display result
    print("\nFraud Detection Result:")
    print(f"Fraud Prediction: {'Yes' if result['fraud_prediction'] else 'No'}")
    print(f"Fraud Probability: {result['fraud_probability']:.4f}")
    print(f"Timestamp: {result['timestamp']}")

if __name__ == "__main__":
    main() 