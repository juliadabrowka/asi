import pandas as pd
import pickle
import os


def generate_label_map_tabular(data_path: str, label_column: str, output_path: str):
    """
    Generate label map from categorical labels in a tabular dataset.

    Args:
        data_path (str): Path to CSV file with tabular data.
        label_column (str): Name of the target column with labels.
        output_path (str): Where to save label_map.pkl
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"CSV not found at {data_path}")

    df = pd.read_csv(data_path)

    if label_column not in df.columns:
        raise ValueError(f"'{label_column}' column not found in dataset.")

    unique_labels = sorted(df[label_column].unique())
    label_map = {i: label for i, label in enumerate(unique_labels)}

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(label_map, f)

    print(f"✅ Saved label_map to: {output_path}")
    print("Label map:")
    for k, v in label_map.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    generate_label_map_tabular(
        data_path="data/01_raw/raw_data.csv",
        label_column="deposit",
        output_path="data/06_models/label_map.pkl"
    )
