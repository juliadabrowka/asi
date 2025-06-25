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
    print(f"   - Time limit: {model_options['time_limit']} seconds ({model_options['time_limit'] / 3600:.1f} hours)")
    print(f"   - Preset: {model_options['presets']} (maximum quality)")
    print(f"   - This will use the best possible model architecture and training strategy")

    # Initialize the AutoGluon predictor for tabular data
    predictor = TabularPredictor(
        label=parameters["label_column"],
        path=model_output_path,
        eval_metric="accuracy",
        verbosity=2,
    )

    # Train the model with maximum quality settings
    predictor.fit(
        train_data=train_df,
        tuning_data=val_df,
        time_limit=model_options["time_limit"],
        presets=model_options["presets"]
    )

    print(f"Model training completed successfully!")
    print(f"Model saved to: {model_output_path}")

    # Save the trained model
    predictor.save(model_output_path)

    return None


def evaluate_model(test_df):
    """
    Evaluates the trained model on the test set.
    """
    model_path = 'data/06_models/autogluon_model'

    print(f"Loading model from {model_path}...")
    predictor = TabularPredictor.load(model_path)

    # Start the evaluation
    print(f"Starting evaluation on the test set...")
    start_time = time.time()
    predictions = predictor.predict(test_df)
    elapsed_time = time.time() - start_time

    # Calculate accuracy
    accuracy = accuracy_score(test_df['label'], predictions)
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Evaluation completed in {elapsed_time:.2f} seconds.")

    return accuracy