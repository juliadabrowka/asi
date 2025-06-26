#!/usr/bin/env python3
"""
Quick training script for fast model development.
This script uses the fastest settings for quick iterations.
"""

import os
import sys
import pandas as pd
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from autogluon.tabular import TabularPredictor
import pickle

def quick_train():
    """Train a model with fastest settings for development."""
    
    # Load data (assuming it exists)
    data_path = "data/03_primary/train_df.csv"
    if not os.path.exists(data_path):
        print(f"❌ Training data not found at {data_path}")
        print("Please run the data processing pipeline first:")
        print("kedro run --pipeline data_processing")
        return
    
    print("🚀 Starting quick training with fastest settings...")
    
    # Load data
    train_df = pd.read_csv(data_path)
    print(f"📊 Loaded {len(train_df)} training samples")
    
    # Quick training settings
    model_path = "data/06_models/quick_model/"
    
    # Clean up existing model
    if os.path.exists(model_path):
        import shutil
        shutil.rmtree(model_path)
    
    # Initialize predictor with fastest settings
    predictor = TabularPredictor(
        label='deposit',
        path=model_path,
        eval_metric="accuracy",
        verbosity=1,
    )
    
    # Train with fastest preset
    print("⏱️ Training with 'optimize_for_deployment' preset (30 seconds max)...")
    predictor.fit(
        train_data=train_df,
        time_limit=30,  # 30 seconds max
        presets='optimize_for_deployment',
        num_cpus=2,  # Use fewer CPUs to avoid system slowdown
    )
    
    print("✅ Quick training completed!")
    print(f"📁 Model saved to: {model_path}")
    
    # Save as pickle for Streamlit app
    pickle_path = "data/06_models/trained_model_predictor.pkl"
    with open(pickle_path, 'wb') as f:
        pickle.dump(predictor, f)
    print(f"💾 Model also saved as pickle: {pickle_path}")
    
    # Quick evaluation
    if 'deposit' in train_df.columns:
        predictions = predictor.predict(train_df)
        accuracy = (predictions == train_df['deposit']).mean()
        print(f"📈 Quick accuracy on training data: {accuracy:.3f}")
    
    return predictor

if __name__ == "__main__":
    quick_train() 