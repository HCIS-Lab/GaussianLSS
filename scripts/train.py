from pathlib import Path

import logging
import pytorch_lightning as pl
import hydra

from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, ModelSummary

from GaussianLSS.common import setup_config, setup_experiment, load_backbone
from GaussianLSS.callbacks.visualization_callback import VisualizationCallback
import torch
from omegaconf import OmegaConf

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
log = logging.getLogger(__name__)

CONFIG_PATH = Path.cwd() / 'config'
CONFIG_NAME = 'config.yaml'


def maybe_resume_training(experiment):
    save_dir = Path(experiment.save_dir).resolve() 
    checkpoints = list(save_dir.glob(f'**/{experiment.uuid}/checkpoints/*.ckpt'))

    log.info(f'Searching {save_dir}.')

    if not checkpoints:
        return None

    log.info(f'Found {checkpoints[-1]}.')
    
    return checkpoints[-1]


@hydra.main(version_base="1.3", config_path=str(CONFIG_PATH), config_name=CONFIG_NAME)
def main(cfg):
    torch.set_float32_matmul_precision('high')
    setup_config(cfg)
    if cfg.experiment.get("seed"):
        pl.seed_everything(cfg.experiment.seed, workers=True)
    Path(cfg.experiment.save_dir).mkdir(exist_ok=True, parents=False)

    # Create and load model/data
    model_module, data_module, viz_fn = setup_experiment(cfg)

    # Optionally load model
    ckpt_path = maybe_resume_training(cfg.experiment)

    if ckpt_path is not None:
        model_module.backbone = load_backbone(ckpt_path)
    # Loggers and callbacks
    logger = pl.loggers.WandbLogger(project=cfg.experiment.project,
                                    save_dir=cfg.experiment.save_dir,
                                    id=cfg.experiment.uuid,
                                    # resume="never",
                                    )
    callbacks = [
        ModelSummary(max_depth=2),
        LearningRateMonitor(logging_interval='epoch'),
        ModelCheckpoint(
            filename='last',
            monitor ='val/metrics/mIoU',
            mode='max',
        ),

        VisualizationCallback(viz_fn, cfg.experiment.log_image_interval)
    ]

    # Traind
    trainer = pl.Trainer(logger=logger,
                         callbacks=callbacks,
                         strategy=DDPStrategy(),
                         **cfg.trainer)
    if trainer.global_rank == 0:
        cfg = OmegaConf.to_container(cfg, resolve=True)
        logger.experiment.config.update(cfg)
        
    trainer.fit(model_module, datamodule=data_module, ckpt_path=ckpt_path)


if __name__ == '__main__':
    main()
