from kedro.pipeline import Pipeline, node, pipeline
from .nodes import train_model, evaluate_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=train_model,
            inputs=["train_df", "val_df", "parameters"],
            outputs="trained_predictor",
            name="train_model_node"
        ),
        node(
            func=evaluate_model,
            inputs=["test_df", "trained_predictor"],
            outputs="test_accuracy",
            name="evaluate_model_node"
        ),
    ])
