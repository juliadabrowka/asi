from __future__ import annotations
from .pipelines import data_processing as dp
from .pipelines import model_training as mt
from .pipelines import model_validation as mv
import os


def register_pipelines():
    model_dir = "C:/Users/julia/Desktop/UNI/asi-bank-project/data/06_models/autogluon_model"
    os.makedirs(model_dir, exist_ok=True)
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    data_processing_pipeline = dp.create_pipeline()
    model_training_pipeline = mt.create_pipeline()
    model_validation_pipeline = mv.create_pipeline()

    return {
        "__default__": data_processing_pipeline + model_training_pipeline + model_validation_pipeline,
        "dp": data_processing_pipeline,
        "train": model_training_pipeline,
        "mv": model_validation_pipeline,
        "model_validation": model_validation_pipeline,
        "full": data_processing_pipeline + model_training_pipeline + model_validation_pipeline,
    }