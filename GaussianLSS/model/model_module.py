import torch
import pytorch_lightning as pl
import torch.distributed as dist
import json

class ModelModule(pl.LightningModule):
    def __init__(self, backbone, loss_func, metrics, optimizer_args, scheduler_args=None, cfg=None, val_only=False):
        super().__init__()

        self.save_hyperparameters(
            cfg,
            ignore=['loss_func', 'metrics', 'scheduler_args','val_only'],
            logger=False,
        )

        self.backbone = backbone
        self.loss_func = loss_func
        self.metrics = metrics
        self.val_only = val_only
        self.optimizer_args = optimizer_args
        self.scheduler_args = scheduler_args

    def forward(self, batch):
        return self.backbone(batch)

    def shared_step(self, batch, prefix='', on_step=False, return_output=True):
        pred = self(batch)
        loss, loss_details, weights = self.loss_func(pred, batch)
        if self.metrics is not None:
            if prefix == 'train':
                if not self.val_only:
                    self.metrics.update(pred, batch)
           
            else:
                self.metrics.update(pred, batch)
                
        if self.trainer is not None:
            self.log(f'{prefix}/loss', loss.detach(), on_step=on_step, on_epoch=True, logger=True)
            self.log_dict({f'{prefix}/loss/{k}': v.detach() for k, v in loss_details.items()},
                          on_step=on_step, on_epoch=True,logger=True)
            if self.training and weights:
                self.log_dict({f'{prefix}/weights/{k}': v.detach() for k, v in weights.items()},
                        on_step=on_step, on_epoch=True,logger=True)

            if 'num_gaussians' in pred:
                self.log(f'{prefix}/num_gaussians', pred['num_gaussians'], on_step=on_step, on_epoch=True, logger=True)
            
        # Used for visualizations
        if return_output:
            return {'loss': loss, 'batch': batch, 'pred': pred}

        return {'loss': loss}

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, 'train', on_step = True,
                                return_output = batch_idx % self.hparams.experiment.log_image_interval == 0)

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, 'val', on_step = False,
                                return_output = batch_idx % self.hparams.experiment.log_image_interval == 0)

    def on_validation_start(self) -> None:
        if not self.val_only:
            self._log_epoch_metrics('train')
        # self._enable_dataloader_shuffle(self.trainer.val_dataloaders)

    def on_validation_epoch_end(self): # validation_epoch_end
        self._log_epoch_metrics('val')

    def _log_epoch_metrics(self, prefix: str):
        """
        lightning is a little odd - it goes

        on_train_start
        ... does all the training steps ...
        on_validation_start
        ... does all the validation steps ...
        on_validation_epoch_end
        on_train_epoch_end
        """ 
        metrics = self.metrics.compute()
        ious = list()
        for key, value in metrics.items():
            if isinstance(value, dict):
                for subkey, val in value.items():
                    # print(f'{prefix}/metrics/{key}{subkey}: {val}')
                    self.log(f'{prefix}/metrics/{key}{subkey}', val, on_epoch=True, logger=True)
            else:
                if 'IoU' in key:
                    ious.append(value)
                self.log(f'{prefix}/metrics/{key}', value, on_epoch=True, logger=True)

        self.log(f'{prefix}/metrics/mIoU', torch.stack(ious).mean(), on_epoch=True, logger=True)
        self.metrics.reset()


    def _enable_dataloader_shuffle(self, dataloaders):
        """
        HACK for https://github.com/PyTorchLightning/pytorch-lightning/issues/11054
        """
        for v in dataloaders:
            v.sampler.shuffle = True
            v.sampler.set_epoch(self.current_epoch)

    def configure_optimizers(self, disable_scheduler=False):
        parameters = [x for x in self.backbone.parameters() if x.requires_grad]
        weighting_param = [x for x in self.loss_func.parameters() if x.requires_grad]

        optimizer = torch.optim.AdamW(parameters+weighting_param, **self.optimizer_args)

        if disable_scheduler or self.scheduler_args is None:
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda lr: 1)
        else:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, **self.scheduler_args)

        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]
