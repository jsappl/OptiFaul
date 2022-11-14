"""Train and test temporal fusion transformers with anaerobic digester data."""

import readline
from pathlib import Path
from typing import TYPE_CHECKING

import hydra
import pytorch_forecasting
import pytorch_lightning
from hydra.utils import instantiate

import optifaul.test as test
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

    if "optuna" in cfg.keys():
        instantiate(
            cfg.optuna,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            loss=loss,
        )
        return None

    if cfg.mode == "train":
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

        fig = predictions_plot(model, test_loader)
        trainer.logger.experiment.add_figure("test/predictions", fig)

    if cfg.mode == "test":
        save_dir = Path("./data/csv/")
        readline.set_completer_delims(" \t\n=")
        readline.parse_and_bind("tab: complete")
        checkpoint_path = input("Choose model checkpoint: ")

        model = eval(cfg.model._target_.rsplit(".", 1)[0]).load_from_checkpoint(checkpoint_path)

        test.partial_dependencies(model, test_loader, save_dir)
        test.compute_metrics(model, test_loader, save_dir)
        test.goodness_of_fit(model, test_loader, save_dir)
        test.variable_ranking(model, test_loader, save_dir)
        test.single_day_forecast(model, test_loader, save_dir)
        test.quantiles_single(model, test_loader, save_dir)


if __name__ == "__main__":
    main()
