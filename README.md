# Credit Card Fraud Detection

This project builds a machine learning model to detect fraudulent credit card transactions and provides an interactive dashboard for analysis.

## Quick Start Guide

### Step 1: Set Up Environment

```shell
# Clone the repository (if not done already)
git clone https://github.com/yourusername/credit-card-fraud-detection.git
cd credit-card-fraud-detection

# Create virtual environment (recommended)
python -m venv cs440
# Activate virtual environment
# On Windows:
cs440\Scripts\activate
# On Mac/Linux:
source cs440/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Prepare Data (Optional)

If the datasets are too large for your machine, create smaller samples:

```shell
python src/dataset_sampler.py
```

### Step 3: Train the Model

You have two options:

```shell
# Option 1: Full model (more features, higher accuracy)
python src/fraud_detection_model.py

# Option 2: Simplified model (faster, uses less memory)
python src/simple_fraud_model.py
```

### Step 4: Test the Model

Verify model performance with test transactions:

```shell
python src/test_transactions.py
```

### Step 5: Run the Interactive Dashboard

```shell
streamlit run src/fraud_detection_demo.py
```

This launches the interactive dashboard in your web browser.

## Dataset

The project uses two datasets:

- `data/fraudTrain.csv`: Training dataset with labeled credit card transactions
- `data/fraudTest.csv`: Testing dataset for evaluating the model

Each dataset contains:

- Transaction details (date, time, amount, merchant, category)
- Card holder information (name, gender, location, job)
- Merchant information (location)
- Target variable: `is_fraud` (1 for fraudulent transactions, 0 for legitimate ones)

## Interactive Dashboard Features

The dashboard has three main sections:

1. **Transaction Simulator**: Enter transaction details to check for fraud
   - Enter custom transaction details
   - Generate random realistic transactions with a single click
   - Select locations easily from a city dropdown
   - Analyze potential fraud risk factors

2. **Fraud Insights**: View statistics and patterns from analyzed transactions
   - Track fraud detection metrics
   - Visualize transaction risk by amount, distance, and category
   - Review transaction history with risk highlighting

3. **Model Information**: Learn about the model and how it works
   - Model performance metrics
   - Key features used in fraud detection
   - Interpretation guidelines for fraud probabilities

## Project Structure

```text
.
├── data/
│   ├── fraudTrain.csv            # Training dataset
│   ├── fraudTest.csv             # Testing dataset
│   ├── plots/                    # Visualizations 
│   └── test_results/             # Evaluation results
├── models/                       # Saved models directory
├── src/
│   ├── fraud_detection_model.py  # Main model training script
│   ├── simple_fraud_model.py     # Simplified model training (faster)
│   ├── dataset_sampler.py        # Dataset sampling utility
│   ├── fraud_detector.py         # Fraud prediction class
│   ├── simple_fraud_detector.py  # Simplified detector
│   ├── test_transactions.py      # Transaction testing utility
│   └── fraud_detection_demo.py   # Interactive dashboard application
└── requirements.txt              # Python dependencies
```

## Requirements

The project requires the following Python packages:

```text
numpy>=1.20.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
shap>=0.40.0
streamlit>=1.12.0
```

## Model Details

The fraud detection model:

- Uses both categorical features (merchant category, gender, state) and numerical features (amount, time, distance)
- Implements feature engineering to extract transaction patterns
- Handles class imbalance through balanced sampling
- Achieves ROC AUC of 0.98+ on test data
- 88% recall for fraudulent transactions

## Troubleshooting

- **Memory issues**: Use `simple_fraud_model.py` instead of `fraud_detection_model.py`
- **Slow performance**: Reduce sample sizes in `dataset_sampler.py`
- **Model not found**: Ensure you've trained the model before running the dashboard
- **Dashboard not stopping**: Use the Stop Application button in the sidebar
- **Keyboard interrupt not working**: Use the Stop button in the dashboard instead
