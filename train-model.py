# train_model.py

import pandas as pd
from autogluon.tabular import TabularPredictor
import os

# Load your training dataset
data_path = "data/01_raw/raw_data.csv"  # ← adjust path if needed
df = pd.read_csv(data_path)

# Define label column
label = "y"  # ← replace with your actual target column name

# Directory to save the model
output_dir = "data/06_models/bank_model"
os.makedirs(output_dir, exist_ok=True)

# Train the model
predictor = TabularPredictor(label=label).fit(df)

# Save the model
predictor.save(output_dir)

print(f"✅ Model trained and saved to {output_dir}")
