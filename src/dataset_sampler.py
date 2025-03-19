import pandas as pd
import numpy as np
from sklearn.utils import resample

def sample_dataset(input_file, output_file, sample_size=None, balanced=True):
    """
    Create a smaller sample of the dataset with optional balancing of classes.
    
    Parameters:
    -----------
    input_file : str
        Path to the input CSV file
    output_file : str
        Path to save the sampled CSV file
    sample_size : int, optional
        Size of the sample to create (if None, uses 10% of dataset)
    balanced : bool, default=True
        Whether to balance the fraud and non-fraud classes
    """
    print(f"Loading dataset from {input_file}...")
    df = pd.read_csv(input_file)
    print(f"Original dataset shape: {df.shape}")
    
    # Default sample size is 10% of the dataset
    if sample_size is None:
        sample_size = int(len(df) * 0.1)
    
    if balanced:
        # Separate fraud and non-fraud transactions
        fraud_df = df[df['is_fraud'] == 1]
        non_fraud_df = df[df['is_fraud'] == 0]
        
        print(f"Original class distribution: Fraud={len(fraud_df)}, Non-fraud={len(non_fraud_df)}")
        
        # Determine how many of each to sample
        fraud_sample_size = min(sample_size // 2, len(fraud_df))
        non_fraud_sample_size = sample_size - fraud_sample_size
        
        # Sample from each class
        fraud_sample = fraud_df.sample(fraud_sample_size, random_state=42)
        non_fraud_sample = non_fraud_df.sample(non_fraud_sample_size, random_state=42)
        
        # Combine samples
        sampled_df = pd.concat([fraud_sample, non_fraud_sample])
        sampled_df = sampled_df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
    else:
        # Random sampling without balancing
        sampled_df = df.sample(sample_size, random_state=42)
    
    print(f"Sampled dataset shape: {sampled_df.shape}")
    print(f"Sampled class distribution: {sampled_df['is_fraud'].value_counts()}")
    
    # Save the sampled dataset
    sampled_df.to_csv(output_file, index=False)
    print(f"Sampled dataset saved to {output_file}")
    
    return sampled_df.shape

if __name__ == "__main__":
    # Sample the training dataset
    sample_dataset(
        input_file="data/fraudTrain.csv",
        output_file="data/fraudTrain_sample.csv",
        sample_size=100000,  # Adjust as needed
        balanced=True
    )
    
    # Sample the testing dataset
    sample_dataset(
        input_file="data/fraudTest.csv",
        output_file="data/fraudTest_sample.csv",
        sample_size=50000,  # Adjust as needed
        balanced=True
    ) 