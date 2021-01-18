"""Train and test temporal fusion transformers with anaerobic digester data."""

import matplotlib

from src import benchmark, preprocessing, train, test

# Required for plotting on server with no display attached.
matplotlib.rcParams.update({"figure.max_open_warning": 0})
matplotlib.use("Agg")


def main(config: dict) -> None:
    """TFT model training and testing pipeline."""
    preprocessing.main(config)
    train.main(config)
    test.main(config)
    benchmark.main(config)


if __name__ == "__main__":
    cfg = {
        "root_dir": "./assets/data/raw/",
        "n_quantiles": 4,
        "train_dir": "./assets/data/preprocessed/",
        "split": 0.5,
        "max_encoder_length": 7,  # steps in history
        "max_prediction_length": 7,  # forecast steps
        "batch_size": 128,
        "log_dir": "./assets/runs/",
        "max_epochs": 2048,
        "hidden_size": 90,
        "lstm_layers": 1,
        "dropout": 0.20899839637548223,
        "attention_head_size": 3,
        "hidden_continuous_size": 25,
        "lr": 0.01149255962845903,
        "weight_decay": 0.0,
    }
    main(cfg)
