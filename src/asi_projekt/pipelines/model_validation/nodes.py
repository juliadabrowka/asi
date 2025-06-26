import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from autogluon.tabular import TabularPredictor


def evaluate_model_metrics(test_df, model_options):
    model_path = "data/06_models/autogluon_model"
    label_column = model_options["label_column"]

    try:
        predictor = TabularPredictor.load(model_path)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Please run the training pipeline first")
        return {
            "predictions": [],
            "true_labels": test_df[label_column].values,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "inference_time": 0.0,
            "avg_inference_time": 0.0,
            "class_labels": ["no", "yes"],
        }

    start_time = time.time()
    predictions = predictor.predict(test_df)
    elapsed = time.time() - start_time

    return {
        "predictions": predictions,
        "true_labels": test_df[label_column].values,
        "accuracy": accuracy_score(test_df[label_column], predictions),
        "precision": precision_score(test_df[label_column], predictions, average="weighted"),
        "recall": recall_score(test_df[label_column], predictions, average="weighted"),
        "f1_score": f1_score(test_df[label_column], predictions, average="weighted"),
        "inference_time": elapsed,
        "avg_inference_time": elapsed / len(test_df),
        "class_labels": ["no", "yes"],
    }


def save_confusion_matrix(metrics, output_path: str):
    cm = confusion_matrix(metrics["true_labels"], metrics["predictions"])

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    return output_path


def save_classification_report(metrics, output_path: str):
    true_names = [metrics["class_labels"][i] for i in metrics["true_labels"]]
    pred_names = [metrics["class_labels"][i] for i in metrics["predictions"]]
    report = classification_report(true_names, pred_names, target_names=metrics["class_labels"], digits=3)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("CLASSIFICATION REPORT\n")
        f.write("=" * 40 + "\n")
        f.write(report)

    return output_path


def analyze_model_size(model_path="data/06_models/autogluon_model"):
    total_size = sum(
        os.path.getsize(os.path.join(dirpath, f))
        for dirpath, _, filenames in os.walk(model_path)
        for f in filenames
    )
    return {
        "size_mb": total_size / 1024**2,
        "size_gb": total_size / 1024**3
    }


def final_assessment(metrics, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("FINAL MODEL ASSESSMENT\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"F1 Score: {metrics['f1_score']:.4f}\n")
        f.write(f"Avg Inference Time: {metrics['avg_inference_time']:.4f} sec\n")
    return output_path
