"""Train and test temporal fusion transformers with anaerobic digester data."""

from pathlib import Path
from typing import TYPE_CHECKING

import hydra
import pytorch_lightning as pl
from hydra.utils import instantiate

if TYPE_CHECKING:
    from omegaconf import DictConfig


@hydra.main(config_path="config", config_name="config", version_base="1.2")
def main(cfg: "DictConfig") -> None:
    pl.seed_everything(cfg.seed, workers=True)
    save_dir = Path(hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"])

    if cfg.run_etl:
        etl = instantiate(cfg.etl)
        etl.execute()

    train_data, val_data, test_data = instantiate(cfg.dataset)

    train_loader = train_data.to_dataloader(train=True, batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    val_loader = val_data.to_dataloader(train=False, batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    test_loader = test_data.to_dataloader(train=False, batch_size=cfg.batch_size, num_workers=cfg.num_workers)

    loss = instantiate(cfg.loss)

    logger = instantiate(cfg.logging)

if __name__ == "__main__":
    main()
