import os
import shutil
from autogluon.tabular import TabularPredictor  # For tabular data
from sklearn.metrics import accuracy_score
import time


def train_model(train_df, val_df, parameters):
    """
    Trains an AutoGluon model for tabular data classification.
    """
    model_output_path = 'data/06_models/autogluon_model/'
    model_options = parameters["model_options"]

    # Clean up any existing model files
    if os.path.exists(model_output_path):
        shutil.rmtree(model_output_path)

    print(f"Starting model training with {model_options['presets']} configuration:")
    print(f"   - Time limit: {model_options['time_limit']} seconds ({model_options['time_limit'] / 60:.1f} minutes)")
    print(f"   - Preset: {model_options['presets']} (balanced quality/speed)")
    print(f"   - Training on {'GPU' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'CPU'}")

    # Initialize the AutoGluon predictor for tabular data
    predictor = TabularPredictor(
        label=model_options["label_column"],
        path=model_output_path,
        eval_metric="accuracy",
        verbosity=1,  # Reduced from 2 to 1 for less verbose output
    )

    # Train the model with optimized settings
    predictor.fit(
        train_data=train_df,
        tuning_data=val_df,
        time_limit=model_options["time_limit"],
        presets=model_options["presets"],
        num_cpus=4,  # Limit CPU usage to avoid system slowdown
        num_gpus=1 if os.environ.get('CUDA_VISIBLE_DEVICES') else 0,  # Use GPU if available
    )

    print(f"Model training completed successfully!")
    print(f"Model saved to: {model_output_path}")

    # Save the trained model as pickle for the app
    import pickle
    pickle_path = 'data/06_models/trained_model_predictor.pkl'
    os.makedirs(os.path.dirname(pickle_path), exist_ok=True)
    with open(pickle_path, 'wb') as f:
        pickle.dump(predictor, f)
    print(f"Model also saved as pickle: {pickle_path}")

    return predictor


def evaluate_model(test_df, trained_predictor=None):
    """
    Evaluates the trained model on the test set.
    """
    if trained_predictor is not None:
        # Use the predictor passed from training
        predictor = trained_predictor
        print("Using trained predictor from previous node")
    else:
        # Try to load from disk
        model_path = 'data/06_models/autogluon_model'
        if not os.path.exists(model_path):
            print(f"❌ Model not found at {model_path}")
            print("Please run the training pipeline first")
            return 0.0

        print(f"Loading model from {model_path}...")
        try:
            predictor = TabularPredictor.load(model_path)
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return 0.0

    # Start the evaluation
    print(f"Starting evaluation on the test set...")
    start_time = time.time()
    
    # Make sure the test_df has the correct label column
    if 'deposit' not in test_df.columns and 'label' in test_df.columns:
        label_col = 'label'
    else:
        label_col = 'deposit'
    
    predictions = predictor.predict(test_df)
    elapsed_time = time.time() - start_time

    # Calculate accuracy
    accuracy = accuracy_score(test_df[label_col], predictions)
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Evaluation completed in {elapsed_time:.2f} seconds.")

    return accuracy