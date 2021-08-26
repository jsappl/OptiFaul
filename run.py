"""Train and test temporal fusion transformers with anaerobic digester data."""

import matplotlib

from src import benchmark, preprocessing, test, train

# Required for plotting on server with no display attached.
matplotlib.rcParams.update({"figure.max_open_warning": 0})
matplotlib.use("Agg")


def main(config: dict) -> None:
    """TFT model training and testing pipeline."""
    preprocessing.main(config)
    best_model_path = train.main(config)
    test.main(config, best_model_path)
    benchmark.main(config)


if __name__ == "__main__":
    cfg = {
        "root_dir": "./assets/data/raw/",
        "n_quantiles": 4,
        "train_dir": "./assets/data/preprocessed/",
        "split": 0.5,
        "max_encoder_length": 14,  # steps in history
        "max_prediction_length": 7,  # forecast steps
        "batch_size": 512,
        "log_dir": "./assets/runs/",
        "max_epochs": 2048,
        "hidden_size": 8,  # can range from 8 to 512
        "lstm_layers": 2,
        "dropout": 0.0,
        "attention_head_size": 4,
        "hidden_continuous_size": 8,
        "lr": 0.001,
        "weight_decay": 0.05,
    }

    main(cfg)
