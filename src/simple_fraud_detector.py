import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime

class SimpleFraudDetector:
    """
    A simplified class to detect credit card fraud using the pre-trained model.
    """
    
    def __init__(self, model_path="models/simple_fraud_model.pkl"):
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
            model_data = pickle.load(file)
            self.model = model_data['model']
            self.features = model_data['features']
            self.categories = model_data.get('categories', [])
        print("Model loaded successfully.")
        print(f"Model uses {len(self.features)} features")
        print(f"Categories: {self.categories}")
        
    def preprocess_transaction(self, transaction):
        """
        Preprocess a transaction for prediction.
        
        Parameters:
        -----------
        transaction : dict
            Transaction data with required fields
            
        Returns:
        --------
        pd.DataFrame
            Preprocessed features ready for prediction
        """
        # Convert to DataFrame if it's a dict
        if isinstance(transaction, dict):
            df = pd.DataFrame([transaction])
        else:
            df = transaction.copy()
        
        # Datetime features
        if 'trans_date_trans_time' in df.columns:
            df['trans_date'] = pd.to_datetime(df['trans_date_trans_time'])
            df['hour'] = df['trans_date'].dt.hour
            df['day_of_week'] = df['trans_date'].dt.dayofweek
        else:
            now = datetime.now()
            df['hour'] = now.hour
            df['day_of_week'] = now.weekday()
        
        # Calculate distance
        if all(col in df.columns for col in ['lat', 'long', 'merch_lat', 'merch_long']):
            df['distance'] = np.sqrt((df['lat'] - df['merch_lat'])**2 + (df['long'] - df['merch_long'])**2)
        else:
            df['distance'] = 0  # Default value
        
        # Create category features
        if 'category' in df.columns and self.categories:
            # Create dummy variables for each category in our model
            for cat in self.categories:
                col_name = f'cat_{cat}'
                df[col_name] = (df['category'] == cat).astype(int)
        
        # Make sure all required features exist
        for feature in self.features:
            if feature not in df.columns:
                df[feature] = 0  # Add default value if feature doesn't exist
        
        # Return only the features used by the model
        return df[self.features]
    
    def predict(self, transaction, threshold=0.5):
        """
        Predict if a transaction is fraudulent.
        
        Parameters:
        -----------
        transaction : dict
            Transaction data to evaluate
        threshold : float, default=0.5
            Probability threshold for classifying as fraud
            
        Returns:
        --------
        dict
            Result with fraud prediction and probability
        """
        try:
            X = self.preprocess_transaction(transaction)
            
            # Get probability of fraud
            fraud_proba = self.model.predict_proba(X)[:, 1]
            
            # Make binary prediction based on threshold
            fraud_prediction = fraud_proba >= threshold
            
            # Return result
            result = {
                'fraud_prediction': bool(fraud_prediction[0]),
                'fraud_probability': float(fraud_proba[0]),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            return result
            
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return {
                'error': str(e),
                'fraud_prediction': None,
                'fraud_probability': None
            }

def main():
    """Demo function showing how to use the detector"""
    
    # Example transaction
    example_transaction = {
        'trans_date_trans_time': '2023-01-01 12:30:45',
        'category': 'shopping',
        'amt': 120.50,
        'lat': 33.9765,
        'long': -118.3937,
        'merch_lat': 34.0522,
        'merch_long': -118.2437
    }
    
    # Create detector
    try:
        detector = SimpleFraudDetector()
        
        # Make prediction
        result = detector.predict(example_transaction)
        
        # Display result
        print("\nFraud Detection Result:")
        if 'error' in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Fraud Prediction: {'Yes' if result['fraud_prediction'] else 'No'}")
            print(f"Fraud Probability: {result['fraud_probability']:.4f}")
            print(f"Timestamp: {result['timestamp']}")
            
    except FileNotFoundError:
        print("\nModel file not found. You need to train the model first by running:")
        print("python src/simple_fraud_model.py")

if __name__ == "__main__":
    main() 