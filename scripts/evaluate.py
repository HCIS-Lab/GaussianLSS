from pathlib import Path
import logging

import torch
import pytorch_lightning as pl
import hydra

from GaussianLSS.common import setup_config, setup_experiment, load_backbone

log = logging.getLogger(__name__)

CONFIG_PATH = Path.cwd() / 'config'
CONFIG_NAME = 'config.yaml'


@hydra.main(version_base="1.3", config_path=str(CONFIG_PATH), config_name=CONFIG_NAME)
def main(cfg):
    # Setup
    torch.set_float32_matmul_precision('high')
    setup_config(cfg)

    if cfg.experiment.get("seed"):
        pl.seed_everything(cfg.experiment.seed, workers=True)

    # Paths
    ckpt_path = Path(cfg.ckpt)
    assert ckpt_path.exists(), f"[Error] Checkpoint not found at {ckpt_path}"

    # Load model and data
    model_module, data_module, _ = setup_experiment(cfg)
    model_module.backbone = load_backbone(str(ckpt_path))

    # Trainer
    trainer = pl.Trainer(
        logger=False,
        **cfg.trainer
    )

    # Evaluate
    trainer.validate(model_module, datamodule=data_module, ckpt_path=ckpt_path)


if __name__ == '__main__':
    main()
