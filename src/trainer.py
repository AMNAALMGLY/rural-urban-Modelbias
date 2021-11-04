import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import torch.nn as nn
from utils.utils import Metric


class ResTrain(pl.LightningModule):
    def __init__(self, model, lr, weight_decay, loss_type, num_outputs, metric, **kwargs):

        '''Initializes the Trainer.
        Args

        - model:
        - lr: int learning_rate
        - weight_decay: int
        - loss_type: str, one of ['classification', 'regression']
        - num_outputs:output class  one of [None ,num_classes]
        - metric:List[str] ['r2','R2' ,'mse', 'rank'] TODO
        -

        '''
        super().__init__()

        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_type = loss_type
        if num_outputs:
            fc = nn.Linear(model.fc.in_features, num_outputs)
            model.fc = fc

        else:  # fearture extraction
            model.fc = nn.Sequential()

        self.metric = Metric().get_metric(metric)  # TODO if it is a list

        self.setup_criterion()

    def forward(self, x):

        raise NotImplementedError

    def _shared_step(self, batch, metric_fn):

        x = torch.tensor(batch['images'], device=self.model.conv1.weight.device)
        target = torch.tensor(batch['labels'], device=self.model.conv1.weight.device)
        x = x.reshape(-1, x.shape[-1], x.shape[-3], x.shape[-2])  # [batch_size ,in_channels, H ,W]

        outputs = self.model(x)
        outputs = outputs.squeeze(dim=-1)
        loss = self.criterion(outputs, target)

        if self.loss_type == 'classification':
            preds = nn.functional.softmax(outputs, dim=1)
        else:
            preds = outputs
        metric_fn.update(preds, target)

        print(loss)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, self.metric)

        # log the outputs!
        self.log('train_loss', loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)

        return {'loss': loss}

    def training_epoch_end(self, training_step_outputs):
        self.log(f'train_metric_epoch', self.metric.compute(), prog_bar=True, )

        # reset after each epoch
        self.metric.reset()

    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch, self.metric)

        # log the outputs!
        self.log('val_loss', loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)

        return {'loss': loss}

    def validation_epoch_end(self, validation_step_outputs):
        self.log(f'val_metric_epoch', self.metric.compute(), prog_bar=True, )

        # reset after each epoch
        self.metric.reset()

    def test_step(self, batch, batch_idx):
        loss = self._shared_step(batch, self.metric)

        # log the outputs!
        self.log('test_loss', loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)

        return {'loss': loss}

    def test_epoch_end(self, test_step_outputs):
        self.log(f'test_metric_epoch', self.metric.compute(), prog_bar=True, )

        # reset after each epoch
        self.metric.reset()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return {
            'optimizer': opt,
            'lr_scheduler': {
                'scheduler': ReduceLROnPlateau(opt,
                                               patience=10,
                                               min_lr=1e-8,
                                               factor=0.5,
                                               verbose=True),
                'monitor': 'train_loss',
            }
        }

    def setup_criterion(self):

        if self.loss_type == 'classification':
            self.criterion = nn.CrossEntropyLoss()

        elif self.loss_type == 'regression':
            self.criterion = nn.MSELoss()
