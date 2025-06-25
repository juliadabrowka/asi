import pandas as pd
from sklearn.model_selection import train_test_split


def preprocess_data(data: pd.DataFrame, parameters: dict):
    """
    Preprocess the raw tabular data by encoding labels and splitting
    into train/validation/test sets.
    """
    label_column = parameters["label_column"]

    if label_column not in data.columns:
        raise ValueError(f"Label column '{label_column}' not in dataset.")

    print(f"✅ Loaded data: {data.shape}")
    print(f"🎯 Label column: {label_column}")
    print(f"🔍 Unique values: {data[label_column].unique()}")

    # Encode label column
    data[label_column], uniques = pd.factorize(data[label_column])
    print(f"🧠 Encoded labels: {dict(enumerate(uniques))}")

    # First split: train+val vs test
    train_val_df, test_df = train_test_split(
        data,
        test_size=parameters["test_size"],
        random_state=parameters["random_state"],
        stratify=data[label_column],
    )

    # Second split: train vs val
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=parameters["test_size"],
        random_state=parameters["random_state"],
        stratify=train_val_df[label_column],
    )

    return train_df, val_df, test_df
