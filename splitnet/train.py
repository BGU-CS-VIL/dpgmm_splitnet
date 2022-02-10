import logging

import hydra
import pytorch_lightning as pl

from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    EarlyStopping,
)

from src.utils.utils import print_config

log = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="set_transformer_2d")
def main(config: DictConfig):

    pl.seed_everything(config.general.seed)

    print_config(config, resolve=True)

    # ------------
    # Data:
    # ------------
    # Init Lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)

    # ------------
    # Model:
    # ------------
    # Init Lightning model
    log.info(f"Instantiating model <{config.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(
        config.model,
        cfg=config,
        _recursive_=False
    )

    # ------------------------
    # Loggers:
    # ------------------------

    exp_dir = config.general.exp_dir

    # ------------------------
    # Callbacks:
    # ------------------------

    pl_callbacks = []

    pl_callbacks.append(LearningRateMonitor(logging_interval="epoch"))

    checkpoint_callback = ModelCheckpoint(
        dirpath=exp_dir,
        save_last=True,
        filename="splitnet-{epoch}"
        + f"-dim:{config.datamodule.cfg.data_props.train.dimension}",
    )

    pl_callbacks.append(checkpoint_callback)

    if config.general.early_stopping:
        pl_callbacks.append(
            EarlyStopping(
                monitor="valid/nmi",
                min_delta=0.001,
                patience=5,
                verbose=True,
                mode="max",
            )
        )

    # ------------------------
    # Training:
    # ------------------------

    log.info("Loading pl.Trainer...")

    trainer = pl.Trainer(
        **config.training,
        callbacks=pl_callbacks,
    )

    log.info("Starting training!")
    trainer.fit(model, datamodule=datamodule)

    if config.general.include_tests:
        trainer.test(datamodule=datamodule, ckpt_path=None)

    # Print path to best checkpoint
    log.info(f"Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
