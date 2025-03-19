import pandas as pd
import numpy as np

data = pd.read_csv('data/fraudTrain.csv')

print("Training data features:")
print(data.columns.tolist())

data_test = pd.read_csv('data/fraudTest.csv')

print("Testing data features:")
print(data_test.columns.tolist())


