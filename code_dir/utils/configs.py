from pydantic import BaseModel

class TrainingConfig(BaseModel):
    target: str = "target"
    prediction_length: int = 10
    eval_metric: str = "MAPE"
    artifact_path: str = "chronos_finetuned_model"
    models_path: str = "bolt_base"
    epochs: int = 5
    learning_rate: float = 1e-5
    batch_size: int = 32
    time_limit: int = 3600
    context_length: int = 2048
    num_samples: int = 20