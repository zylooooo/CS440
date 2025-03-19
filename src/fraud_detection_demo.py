import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, time, timedelta
import shap
import sys
import random

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.fraud_detector import FraudDetector

class FraudDetectionDashboard:
    def __init__(self):
        st.set_page_config(
            page_title="Credit Card Fraud Detection Demo",
            page_icon="💳",
            layout="wide"
        )
        
        # Initialize all session state variables first
        if 'show_stop_confirmation' not in st.session_state:
            st.session_state.show_stop_confirmation = False
        
        if 'random_transaction' not in st.session_state:
            st.session_state.random_transaction = {}
        
        # Now use the session state variables safely
        if st.sidebar.button("⚠️ Stop Application", help="Click to stop the app"):
            st.session_state.show_stop_confirmation = True
        
        # Display confirmation dialog if needed
        if st.session_state.show_stop_confirmation:
            st.sidebar.warning("Are you sure you want to stop the application?")
            col1, col2 = st.sidebar.columns(2)
            if col1.button("Yes, Stop App"):
                st.sidebar.success("Stopping the application...")
                st.stop()
            if col2.button("No, Continue"):
                st.session_state.show_stop_confirmation = False
        
        # Load the trained model
        try:
            self.detector = FraudDetector(model_path="models/fraud_detection_model.pkl")
            st.sidebar.success("Model loaded successfully!")
        except FileNotFoundError:
            st.sidebar.error("Model not found! Please train the model first.")
            st.stop()
            
        # Load SHAP explainer for the model (optional)
        try:
            self.explainer = self.load_explainer()
        except:
            self.explainer = None
            st.sidebar.warning("SHAP explainer not available. Explanations will be limited.")
    
    def load_explainer(self):
        # This is optional but adds better explanations
        if os.path.exists("models/shap_explainer.pkl"):
            with open("models/shap_explainer.pkl", "rb") as f:
                return pickle.load(f)
        else:
            # Create a simple sample for the explainer
            # This could be improved by saving a properly trained explainer
            sample_X = pd.DataFrame({
                'amt': [100.0],
                'hour': [12],
                'day_of_week': [3],
                'month': [6],
                'distance': [5.0],
                'city_pop': [50000],
                'category': ['shopping'],
                'gender': ['M'],
                'state': ['CA']
            })
            
            # Process the sample
            X_processed = self.detector.preprocess_transaction(sample_X)
            
            # Create a simple explainer
            return shap.TreeExplainer(self.detector.model.named_steps['classifier'])
    
    def run(self):
        """Main function to run the Streamlit app"""
        st.title("💳 Credit Card Fraud Detection Demo")
        st.markdown("""
        This interactive dashboard allows you to simulate credit card transactions 
        and see if they would be detected as fraudulent by our machine learning model.
        """)
        
        # Create tabs for different sections
        tab1, tab2, tab3 = st.tabs(["Transaction Simulator", "Fraud Insights", "About the Model"])
        
        with tab1:
            self.transaction_simulator()
            
        with tab2:
            self.fraud_insights()
            
        with tab3:
            self.model_info()
    
    def transaction_simulator(self):
        """Create the transaction simulation interface with improved location selection"""
        st.header("Transaction Simulator")
        st.markdown("Enter transaction details below to check if it would be flagged as fraud.")
        
        # Add a random generation button at the top
        if st.button("🎲 Generate Random Transaction", help="Fill the form with random but realistic transaction data"):
            random_data = self.generate_random_transaction()
            st.session_state.random_transaction = random_data
        
        # Create two columns for input fields
        col1, col2 = st.columns(2)
        
        with col1:
            # Use random data if available
            default_amount = st.session_state.get('random_transaction', {}).get('amount', 120.50)
            amount = st.number_input("Transaction Amount ($)", 
                                    min_value=1.0, 
                                    max_value=10000.0, 
                                    value=float(default_amount))
            
            default_category = st.session_state.get('random_transaction', {}).get('category', 'shopping_pos')
            category = st.selectbox("Category", [
                'grocery_pos', 'gas_transport', 'home', 'shopping_pos', 
                'shopping_net', 'kids_pets', 'entertainment', 
                'personal_care', 'food_dining', 'health_fitness', 'misc_pos'
            ], index=[
                'grocery_pos', 'gas_transport', 'home', 'shopping_pos', 
                'shopping_net', 'kids_pets', 'entertainment', 
                'personal_care', 'food_dining', 'health_fitness', 'misc_pos'
            ].index(default_category))
            
            default_gender = st.session_state.get('random_transaction', {}).get('gender', 'M')
            gender = st.selectbox("Cardholder Gender", 
                                 ["M", "F"], 
                                 index=0 if default_gender == "M" else 1)
            
            default_state = st.session_state.get('random_transaction', {}).get('state', 'CA')
            state = st.selectbox("State", [
                "CA", "NY", "TX", "FL", "IL", "PA", "OH", "GA", "NC", "MI", 
                "NJ", "VA", "WA", "AZ", "MA", "TN", "IN", "MO", "MD", "WI"
            ], index=[
                "CA", "NY", "TX", "FL", "IL", "PA", "OH", "GA", "NC", "MI", 
                "NJ", "VA", "WA", "AZ", "MA", "TN", "IN", "MO", "MD", "WI"
            ].index(default_state) if default_state in [
                "CA", "NY", "TX", "FL", "IL", "PA", "OH", "GA", "NC", "MI", 
                "NJ", "VA", "WA", "AZ", "MA", "TN", "IN", "MO", "MD", "WI"
            ] else 0)
            
        with col2:
            # Handles time selection
            st.subheader("Transaction Time")
            default_time = st.session_state.get('random_transaction', {}).get('trans_time', datetime.now().time())
            trans_time = st.time_input("Time", default_time)
            
            default_date = st.session_state.get('random_transaction', {}).get('trans_date', datetime.now().date())
            trans_date = st.date_input("Date", default_date)
            
            # Improved Location UI
            st.subheader("Locations")
            
            # Predefined city locations
            cities = {
                "New York, NY": [40.7128, -74.0060],
                "Los Angeles, CA": [34.0522, -118.2437],
                "Chicago, IL": [41.8781, -87.6298],
                "Houston, TX": [29.7604, -95.3698],
                "Phoenix, AZ": [33.4484, -112.0740],
                "Philadelphia, PA": [39.9526, -75.1652],
                "San Antonio, TX": [29.4241, -98.4936],
                "San Diego, CA": [32.7157, -117.1611],
                "Dallas, TX": [32.7767, -96.7970],
                "San Francisco, CA": [37.7749, -122.4194],
                "Austin, TX": [30.2672, -97.7431],
                "Seattle, WA": [47.6062, -122.3321],
                "Denver, CO": [39.7392, -104.9903],
                "Miami, FL": [25.7617, -80.1918],
                "Custom Location": [0, 0]
            }
            
            # Default values from random generator if available
            default_ch_lat = st.session_state.get('random_transaction', {}).get('user_lat', 34.0522)
            default_ch_long = st.session_state.get('random_transaction', {}).get('user_long', -118.2437)
            
            # Find closest city to default coordinates
            default_city = "Los Angeles, CA"  # fallback
            if 'random_transaction' in st.session_state:
                min_dist = float('inf')
                for city, coords in cities.items():
                    if city != "Custom Location":
                        dist = ((coords[0] - default_ch_lat)**2 + (coords[1] - default_ch_long)**2)**0.5
                        if dist < min_dist:
                            min_dist = dist
                            default_city = city
            
            # Cardholder location selection
            ch_location = st.selectbox(
                "Cardholder Location", 
                list(cities.keys()),
                index=list(cities.keys()).index(default_city)
            )
            
            # Handle custom location entry
            if ch_location == "Custom Location":
                user_lat = st.number_input("Cardholder Latitude", 
                                          min_value=-90.0, 
                                          max_value=90.0, 
                                          value=default_ch_lat)
                user_long = st.number_input("Cardholder Longitude", 
                                           min_value=-180.0, 
                                           max_value=180.0, 
                                           value=default_ch_long)
            else:
                user_lat, user_long = cities[ch_location]
                st.text(f"Coordinates: ({user_lat:.4f}, {user_long:.4f})")
            
            # Similar approach for merchant location
            default_m_lat = st.session_state.get('random_transaction', {}).get('merch_lat', 34.0549)
            default_m_long = st.session_state.get('random_transaction', {}).get('merch_long', -118.2426)
            
            # Find closest city to merchant coordinates
            default_merch_city = "Los Angeles, CA"  # fallback
            if 'random_transaction' in st.session_state:
                min_dist = float('inf')
                for city, coords in cities.items():
                    if city != "Custom Location":
                        dist = ((coords[0] - default_m_lat)**2 + (coords[1] - default_m_long)**2)**0.5
                        if dist < min_dist:
                            min_dist = dist
                            default_merch_city = city
            
            # Add "Same as Cardholder" option
            merch_options = ["Same as Cardholder"] + list(cities.keys())
            merch_location = st.selectbox(
                "Merchant Location", 
                merch_options,
                index=0 if ch_location == default_merch_city else 
                      (merch_options.index(default_merch_city) if default_merch_city in merch_options else 0)
            )
            
            # Handle merchant location logic
            if merch_location == "Same as Cardholder":
                merch_lat, merch_long = user_lat, user_long
                st.text(f"Coordinates: ({merch_lat:.4f}, {merch_long:.4f})")
            elif merch_location == "Custom Location":
                merch_lat = st.number_input("Merchant Latitude", 
                                           min_value=-90.0, 
                                           max_value=90.0, 
                                           value=default_m_lat)
                merch_long = st.number_input("Merchant Longitude", 
                                            min_value=-180.0, 
                                            max_value=180.0, 
                                            value=default_m_long)
            else:
                merch_lat, merch_long = cities[merch_location]
                st.text(f"Coordinates: ({merch_lat:.4f}, {merch_long:.4f})")
            
            # Calculate distance
            distance = np.sqrt((user_lat - merch_lat)**2 + (user_long - merch_long)**2) * 111  # Approx km
            st.info(f"Distance: {distance:.2f} km")
            
        # Combine date and time
        trans_datetime = datetime.combine(trans_date, trans_time)
        trans_datetime_str = trans_datetime.strftime("%Y-%m-%d %H:%M:%S")
        
        # Create transaction dict
        transaction = {
            'trans_date_trans_time': trans_datetime_str,
            'category': category,
            'amt': amount,
            'gender': gender,
            'state': state,
            'lat': user_lat,
            'long': user_long,
            'merch_lat': merch_lat,
            'merch_long': merch_long,
            'city_pop': 1000000,  # Default value, could be made adjustable
            'hour': trans_datetime.hour,
            'day_of_week': trans_datetime.weekday(),
            'month': trans_datetime.month,
            'distance': distance
        }
        
        # Prediction button
        if st.button("Analyze Transaction", type="primary"):
            self.analyze_transaction(transaction)
    
    def analyze_transaction(self, transaction):
        """Analyze a transaction and display results"""
        st.subheader("Analysis Results")
        
        with st.spinner("Analyzing transaction..."):
            # Get prediction
            result = self.detector.predict(transaction)
            
            # Display result with appropriate styling
            if result['fraud_prediction']:
                st.error("⚠️ This transaction is flagged as potentially fraudulent!")
                fraud_status = "FRAUD ALERT"
            else:
                st.success("✅ This transaction appears legitimate")
                fraud_status = "LEGITIMATE"
            
            # Create metrics row
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Transaction Status", fraud_status)
            col2.metric("Fraud Probability", f"{result['fraud_probability']:.2%}")
            col3.metric("Amount", f"${transaction['amt']:.2f}")
            col4.metric("Distance", f"{transaction['distance']:.2f} km")
            
            # Store the transaction in session state for later exploration
            if 'transactions' not in st.session_state:
                st.session_state.transactions = []
                
            transaction_copy = transaction.copy()
            transaction_copy.update({
                'fraud_prediction': result['fraud_prediction'],
                'fraud_probability': result['fraud_probability'],
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            st.session_state.transactions.append(transaction_copy)
            
            # Provide explanation
            st.subheader("Explanation")
            self.explain_prediction(transaction, result)
            
    def explain_prediction(self, transaction, result):
        """Provide an explanation for the prediction"""
        # Create two columns
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Basic explanation
            st.markdown("### Key Factors")
            
            # Factors that might indicate fraud
            factors = []
            
            # Amount-based factors
            if transaction['amt'] > 1000:
                factors.append({"factor": "High transaction amount", 
                                "desc": f"Amount of ${transaction['amt']:.2f} is higher than typical", 
                                "impact": "high"})
            
            # Time-based factors
            hour = transaction['hour']
            if hour < 6 or hour > 22:
                factors.append({"factor": "Unusual transaction time", 
                                "desc": f"Transaction time {hour}:00 is outside normal hours", 
                                "impact": "medium"})
            
            # Distance-based factors
            if transaction['distance'] > 100:
                factors.append({"factor": "Large distance", 
                                "desc": f"Distance of {transaction['distance']:.2f} km is unusually large", 
                                "impact": "high"})
            
            # Category-based factors
            high_risk_categories = ['shopping_net', 'entertainment']
            if transaction['category'] in high_risk_categories:
                factors.append({"factor": "High-risk category", 
                                "desc": f"Category '{transaction['category']}' has higher fraud rates", 
                                "impact": "medium"})
            
            # Display factors
            if len(factors) > 0:
                for f in factors:
                    if f["impact"] == "high":
                        st.warning(f"**{f['factor']}**: {f['desc']}")
                    else:
                        st.info(f"**{f['factor']}**: {f['desc']}")
            else:
                st.success("No major risk factors detected in this transaction")
        
        with col2:
            # Risk gauge
            st.markdown("### Fraud Risk Score")
            fig, ax = plt.figure(figsize=(4, 3)), plt.axes()
            
            # Create gauge chart
            prob = result['fraud_probability']
            colors = ['green', 'yellow', 'orange', 'red']
            thresholds = [0, 0.3, 0.6, 0.9, 1.0]
            
            for i in range(len(colors)):
                ax.axvspan(thresholds[i], thresholds[i+1], alpha=0.3, color=colors[i])
            
            ax.arrow(0, 0, prob, 0, head_width=0.05, head_length=0.05, fc='black', ec='black')
            ax.set_xlim(0, 1)
            ax.set_ylim(-0.1, 0.1)
            ax.set_yticks([])
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
            ax.set_title('Fraud Risk')
            
            st.pyplot(fig)
            
            # Add a more detailed explanation based on the probability
            if prob < 0.3:
                st.write("This transaction has characteristics typical of legitimate transactions.")
            elif prob < 0.6:
                st.write("This transaction has some unusual characteristics but isn't strongly indicative of fraud.")
            elif prob < 0.9:
                st.write("This transaction has multiple characteristics commonly associated with fraudulent activity.")
            else:
                st.write("This transaction strongly matches patterns seen in known fraudulent transactions.")

    def fraud_insights(self):
        """Display insights from past transactions"""
        st.header("Fraud Insights Dashboard")
        
        if 'transactions' not in st.session_state or len(st.session_state.transactions) == 0:
            st.info("No transactions analyzed yet. Use the Transaction Simulator to analyze transactions.")
            return
        
        # Convert session transactions to DataFrame
        df = pd.DataFrame(st.session_state.transactions)
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Transactions", len(df))
        col2.metric("Flagged as Fraud", len(df[df['fraud_prediction']]))
        col3.metric("Average Risk Score", f"{df['fraud_probability'].mean():.2%}")
        
        # Create tabs for different insights
        tab1, tab2 = st.tabs(["Transaction History", "Risk Analysis"])
        
        with tab1:
            st.subheader("Recent Transactions")
            # Format the dataframe for display
            display_df = df[['timestamp', 'category', 'amt', 'distance', 'fraud_probability', 'fraud_prediction']].copy()
            display_df.columns = ['Timestamp', 'Category', 'Amount', 'Distance (km)', 'Risk Score', 'Flagged as Fraud']
            display_df['Risk Score'] = display_df['Risk Score'].apply(lambda x: f"{x:.2%}")
            display_df['Amount'] = display_df['Amount'].apply(lambda x: f"${x:.2f}")
            
            # Highlight fraud predictions
            def highlight_fraud(val):
                color = 'red' if val == True else 'green'
                return f'background-color: {color}; color: white'
            
            styled_df = display_df.style.applymap(
                highlight_fraud, subset=['Flagged as Fraud']
            )
            
            st.dataframe(styled_df, use_container_width=True)
            
        with tab2:
            st.subheader("Risk Factors Analysis")
            
            # Create side-by-side visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Amount vs Fraud Probability
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.scatterplot(x='amt', y='fraud_probability', 
                                hue='fraud_prediction', data=df, ax=ax)
                ax.set_xlabel('Transaction Amount')
                ax.set_ylabel('Fraud Probability')
                ax.set_title('Amount vs. Fraud Risk')
                st.pyplot(fig)
            
            with col2:
                # Distance vs Fraud Probability
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.scatterplot(x='distance', y='fraud_probability', 
                                hue='fraud_prediction', data=df, ax=ax)
                ax.set_xlabel('Distance (km)')
                ax.set_ylabel('Fraud Probability')
                ax.set_title('Distance vs. Fraud Risk')
                st.pyplot(fig)
            
            # Category analysis
            st.subheader("Category Risk Analysis")
            
            # Category vs average fraud probability
            cat_risk = df.groupby('category')['fraud_probability'].mean().sort_values(ascending=False).reset_index()
            cat_risk.columns = ['Category', 'Average Risk Score']
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Category', y='Average Risk Score', data=cat_risk, ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.set_title('Average Fraud Risk by Category')
            ax.set_ylim(0, 1)
            st.pyplot(fig)
    
    def model_info(self):
        """Display information about the model"""
        st.header("About the Fraud Detection Model")
        
        st.markdown("""
        ### Model Overview
        
        This fraud detection system uses a Random Forest classifier to identify potentially fraudulent credit card transactions. 
        The model was trained on a dataset of over 1.8 million transactions, with approximately 0.4% labeled as fraudulent.
        
        ### Key Features Used
        
        The model analyzes the following aspects of each transaction:
        
        - **Transaction amount**: Unusually high amounts may indicate fraud
        - **Transaction timing**: The hour, day of week, and month can help identify suspicious patterns
        - **Distance**: The distance between the cardholder and merchant locations
        - **Category**: Some merchant categories have higher fraud rates
        - **Cardholder profile**: Basic demographic information that helps establish normal patterns
        
        ### Performance Metrics
        
        The model achieves the following performance on test data:
        
        - **ROC AUC**: 0.9832 (excellent discriminative ability)
        - **Overall accuracy**: 98%
        - **Recall for fraud detection**: 88% (percentage of actual fraud caught)
        - **Precision for fraud detection**: 13% (percentage of fraud alerts that are actual fraud)
        
        ### Interpretation of Results
        
        When using this simulator:
        
        - A **low fraud probability** (<30%) suggests the transaction is likely legitimate
        - A **medium fraud probability** (30-60%) indicates some unusual characteristics
        - A **high fraud probability** (>60%) suggests strong fraud indicators
        
        In a real-world system, high-probability transactions would be flagged for review or might trigger additional verification steps.
        """)
        
        # Add a footnote
        st.markdown("---")
        st.caption("Note: This is a demonstration system. A production fraud detection system would incorporate additional factors and safeguards.")

    def generate_random_transaction(self):
        """Generate realistic random transaction data"""
        # Common city locations with [name, state, lat, long]
        locations = [
            ["New York", "NY", 40.7128, -74.0060],
            ["Los Angeles", "CA", 34.0522, -118.2437],
            ["Chicago", "IL", 41.8781, -87.6298],
            ["Houston", "TX", 29.7604, -95.3698],
            ["Phoenix", "AZ", 33.4484, -112.0740],
            ["Philadelphia", "PA", 39.9526, -75.1652],
            ["San Antonio", "TX", 29.4241, -98.4936],
            ["San Diego", "CA", 32.7157, -117.1611],
            ["Miami", "FL", 25.7617, -80.1918],
            ["Seattle", "WA", 47.6062, -122.3321]
        ]
        
        # Transaction categories with weighted probabilities
        categories = [
            'grocery_pos', 'gas_transport', 'home', 'shopping_pos', 
            'shopping_net', 'kids_pets', 'entertainment', 
            'personal_care', 'food_dining', 'health_fitness', 'misc_pos'
        ]
        
        # Category-based amount ranges [min, max, typical]
        amount_ranges = {
            'grocery_pos': [10, 300, 75],
            'gas_transport': [20, 120, 45],
            'home': [50, 2000, 200],
            'shopping_pos': [10, 500, 90],
            'shopping_net': [15, 3000, 150],
            'kids_pets': [10, 500, 80],
            'entertainment': [20, 1000, 150],
            'personal_care': [15, 300, 65],
            'food_dining': [15, 200, 60],
            'health_fitness': [20, 1000, 100],
            'misc_pos': [5, 500, 50]
        }
        
        # Randomly select location (cardholder location)
        location = random.choice(locations)
        state = location[1]
        ch_lat = location[2]
        ch_long = location[3]
        
        # Determine if transaction is local or remote (80% local)
        is_local = random.random() < 0.8
        
        if is_local:
            # Local transaction - merchant is near cardholder
            # Add small random offset to coordinates (within ~5km)
            m_lat = ch_lat + random.uniform(-0.05, 0.05)
            m_long = ch_long + random.uniform(-0.05, 0.05)
        else:
            # Remote transaction - merchant is in a different location
            remote_location = random.choice([loc for loc in locations if loc != location])
            m_lat = remote_location[2]
            m_long = remote_location[3]
        
        # Randomly select category
        category = random.choice(categories)
        
        # Generate amount based on category
        range_info = amount_ranges.get(category, [10, 500, 50])
        
        # 70% chance of typical amount, 30% chance of outlier
        if random.random() < 0.7:
            # Typical amount with some variation
            amount = random.uniform(
                range_info[2] * 0.7,
                range_info[2] * 1.5
            )
        else:
            # Outlier amount
            amount = random.uniform(range_info[0], range_info[1])
        
        # Random gender
        gender = random.choice(["M", "F"])
        
        # Random time - weighted toward business hours
        hour = random.choices(
            range(24),
            weights=[1, 1, 1, 1, 1, 2, 5, 10, 15, 20, 20, 25, 
                    30, 25, 20, 15, 15, 20, 15, 10, 5, 3, 2, 1]
        )[0]
        minute = random.randint(0, 59)
        transaction_time = time(hour, minute)
        
        # Random date (within last 30 days)
        days_ago = random.randint(0, 30)
        transaction_date = datetime.now().date() - timedelta(days=days_ago)
        
        # Return all the random data
        return {
            'amount': round(amount, 2),
            'category': category,
            'gender': gender,
            'state': state,
            'trans_time': transaction_time,
            'trans_date': transaction_date,
            'user_lat': ch_lat,
            'user_long': ch_long,
            'merch_lat': m_lat,
            'merch_long': m_long
        }

if __name__ == "__main__":
    dashboard = FraudDetectionDashboard()
    dashboard.run()
