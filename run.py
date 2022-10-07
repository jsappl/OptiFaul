"""Train and test temporal fusion transformers with anaerobic digester data."""

from typing import TYPE_CHECKING

import hydra
from hydra.utils import instantiate

if TYPE_CHECKING:
    from omegaconf import DictConfig


@hydra.main(config_path="config", config_name="config", version_base="1.2")
def main(cfg: "DictConfig") -> None:
    if cfg.run_etl:
        etl = instantiate(cfg.etl)
        etl.execute()


if __name__ == "__main__":
    main()
