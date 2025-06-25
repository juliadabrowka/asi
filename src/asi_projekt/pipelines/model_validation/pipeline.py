from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    evaluate_model_metrics,
    save_confusion_matrix,
    save_classification_report,
    analyze_model_size,
    final_assessment,
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=evaluate_model_metrics,
            inputs=["test_df", "params:model_options"],
            outputs="evaluation_metrics",
            name="evaluate_model_metrics_node",
        ),
        node(
            func=save_confusion_matrix,
            inputs=["evaluation_metrics", "params:confusion_matrix_path"],
            outputs="confusion_matrix_path",
            name="confusion_matrix_node",
        ),
        node(
            func=save_classification_report,
            inputs=["evaluation_metrics", "params:classification_report_path"],
            outputs="classification_report_path",
            name="classification_report_node",
        ),
        node(
            func=analyze_model_size,
            inputs=[],
            outputs="model_size_info",
            name="analyze_model_size_node",
        ),
        node(
            func=final_assessment,
            inputs=["evaluation_metrics", "params:final_assessment_path"],
            outputs="assessment_path",
            name="final_assessment_node",
        ),
    ])
