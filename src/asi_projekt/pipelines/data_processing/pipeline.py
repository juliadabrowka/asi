from kedro.pipeline import Pipeline, node, pipeline
from .nodes import preprocess_data

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=preprocess_data,
            inputs=["raw_data", "params:model_options"],
            outputs=["train_df", "val_df", "test_df"],
            name="preprocess_tabular_data",
        ),
    ])
