"""Train and test temporal fusion transformers with anaerobic digester data."""

from pathlib import Path
from typing import TYPE_CHECKING

import hydra
import pytorch_lightning
import torch
from hydra.utils import instantiate

from optifaul.test import compute_metrics, goodness_of_fit
from optifaul.utils import namestr_from

if TYPE_CHECKING:
    from omegaconf import DictConfig


@hydra.main(config_path="config", config_name="config", version_base="1.2")
def main(cfg: "DictConfig") -> None:
    pytorch_lightning.seed_everything(cfg.seed, workers=True)

    if cfg.run_etl:
        etl = instantiate(cfg.etl)
        etl.execute()

    train_data, val_data, test_data = instantiate(cfg.dataset)

    train_loader = train_data.to_dataloader(train=True, batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    val_loader = val_data.to_dataloader(train=False, batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    test_loader = test_data.to_dataloader(train=False, batch_size=cfg.batch_size, num_workers=cfg.num_workers)

    loss = instantiate(cfg.loss)

    model = instantiate(
        cfg.model,
        dataset=train_data,
        loss=loss,
    )

    logger = instantiate(cfg.logging, version=namestr_from(model))

    callbacks = []
    for callback in cfg.callbacks:
        callbacks.append(instantiate(eval(f"cfg.callbacks.{callback}")))

    trainer = instantiate(
        cfg.trainer,
        logger=logger,
        callbacks=callbacks,
    )
    trainer.fit(model, train_loader, val_loader)

    model.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    save_dir = Path("./data/csv/")
    compute_metrics(model, test_loader, save_dir)
    goodness_of_fit(model, test_loader, save_dir)


if __name__ == "__main__":
    main()
