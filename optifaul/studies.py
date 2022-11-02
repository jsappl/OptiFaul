"""Hyperparameter studies and model sensitivity analysis."""

from typing import TYPE_CHECKING, List

from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters

if TYPE_CHECKING:
    from optuna import Study
    from pytorch_forecasting.metrics import MultiHorizonMetric
    from torch.utils.data import DataLoader


def hparams_study(
        train_dataloaders: "DataLoader", val_dataloaders: "DataLoader", model_path: str, max_epochs: int, n_trials: int,
        hidden_size_range: List[int], hidden_continuous_size_range: List[int], attention_head_size_range: List[int],
        dropout_range: List[int], learning_rate_range: List[int], use_learning_rate_finder: bool, log_dir: str,
        reduce_on_plateau_patience: int, output_size: int, loss: "MultiHorizonMetric") -> "Study":
    """Optimize Temporal Fusion Transformer hyperparameters with this wrapper function."""
    return optimize_hyperparameters(
        train_dataloaders=train_dataloaders,
        val_dataloaders=val_dataloaders,
        model_path=model_path,
        max_epochs=max_epochs,
        n_trials=n_trials,
        hidden_size_range=tuple(hidden_size_range),
        hidden_continuous_size_range=tuple(hidden_continuous_size_range),
        attention_head_size_range=tuple(attention_head_size_range),
        dropout_range=tuple(dropout_range),
        use_learning_rate_finder=use_learning_rate_finder,
        learning_rate_range=tuple(learning_rate_range),
        trainer_kwargs={"log_every_n_steps": 1},
        log_dir=log_dir,
        reduce_on_plateau_patience=16,
        output_size=5,
        loss=loss,
    )
