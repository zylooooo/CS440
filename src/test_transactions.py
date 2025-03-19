import pandas as pd
import numpy as np
from datetime import datetime
import sys
from simple_fraud_detector import SimpleFraudDetector

def test_transactions():
    """Test multiple transactions with our fraud detection model"""
    print("Credit Card Fraud Detection - Test Transactions")
    print("=" * 50)
    
    # Create test transactions
    transactions = [
        {
            "id": 1,
            "trans_date_trans_time": "2023-01-01 14:30:00",
            "category": "shopping_pos",
            "amt": 50.25,
            "lat": 40.7128,
            "long": -74.0060,
            "merch_lat": 40.7135,
            "merch_long": -74.0055
        },
        {
            "id": 2,
            "trans_date_trans_time": "2023-01-01 02:15:00",  # Late night
            "category": "entertainment",
            "amt": 1299.99,  # High amount
            "lat": 34.0522,
            "long": -118.2437,
            "merch_lat": 37.7749,  # Far from cardholder
            "merch_long": -122.4194
        },
        {
            "id": 3,
            "trans_date_trans_time": "2023-01-02 12:30:00",
            "category": "grocery_pos",
            "amt": 85.40,
            "lat": 51.5074,
            "long": -0.1278,
            "merch_lat": 51.5080,
            "merch_long": -0.1290
        },
        {
            "id": 4,
            "trans_date_trans_time": "2023-01-02 15:45:00",
            "category": "shopping_net",
            "amt": 3500.00,  # Very high amount
            "lat": 48.8566,
            "long": 2.3522,
            "merch_lat": 52.5200,  # Different country
            "merch_long": 13.4050
        },
        {
            "id": 5,
            "trans_date_trans_time": "2023-01-03 09:20:00",
            "category": "health_fitness",
            "amt": 120.00,
            "lat": 35.6762,
            "long": 139.6503,
            "merch_lat": 35.6895,
            "merch_long": 139.6917
        }
    ]
    
    try:
        # Initialize detector
        detector = SimpleFraudDetector()
        
        # Process each transaction
        results = []
        
        for trans in transactions:
            # Calculate distance for display (approximate km)
            distance_km = np.sqrt(
                (trans["lat"] - trans["merch_lat"])**2 + 
                (trans["long"] - trans["merch_long"])**2
            ) * 111  # Rough conversion to km
            
            # Get prediction
            prediction = detector.predict(trans)
            
            # Add to results
            result = {
                "id": trans["id"],
                "category": trans["category"],
                "amount": trans["amt"],
                "time": trans["trans_date_trans_time"],
                "distance_km": distance_km,
                "fraud_prediction": prediction["fraud_prediction"],
                "fraud_probability": prediction["fraud_probability"]
            }
            results.append(result)
        
        # Display results
        print("\nTransaction Analysis Results:")
        print("-" * 50)
        
        for r in results:
            print(f"\nTransaction #{r['id']}:")
            print(f"  Category: {r['category']}")
            print(f"  Amount: ${r['amount']:.2f}")
            print(f"  Time: {r['time']}")
            print(f"  Distance: {r['distance_km']:.2f} km")
            print(f"  Fraud Prediction: {'⚠️ YES' if r['fraud_prediction'] else 'NO'}")
            print(f"  Fraud Probability: {r['fraud_probability']:.4f}")
        
        # Summary
        print("\nSummary:")
        print("-" * 50)
        total = len(results)
        fraud = sum(1 for r in results if r['fraud_prediction'])
        legitimate = total - fraud
        
        print(f"Total transactions: {total}")
        print(f"Flagged as fraud: {fraud}")
        print(f"Legitimate: {legitimate}")
        
        # Save to CSV
        df = pd.DataFrame(results)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"data/test_results/fraud_predictions_{timestamp}.csv"
        df.to_csv(filename, index=False)
        print(f"\nResults saved to {filename}")
        
    except FileNotFoundError:
        print("\nError: Model file not found!")
        print("Please train the model first by running: python src/simple_fraud_model.py")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    test_transactions() 