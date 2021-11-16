from torch.optim.lr_scheduler import ExponentialLR
import torch
import torch.nn as nn
from utils.utils import Metric
from configs import args


class Trainer:
    def __init__(self, model, lr, weight_decay, loss_type, num_outputs, metric,save_checkpoint, save_dir, **kwargs):

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

        if num_outputs is not None:

            fc = nn.Linear(model.fc.in_features, num_outputs)
            model.fc = fc

        else:  # fearture extraction   #TODO fix this!
            # model.fc = nn.Sequential()
            raise ValueError('please specify a value for your number of outputs for the loss function to evaluate '
                             'against')

        self.metric = Metric().get_metric(metric)  # TODO if it is a list
        #self.metric=Accuracy(num_classes=10)
        self.setup_criterion()

    '''
    def forward(self, x):



        output = self.model(x)

        return output
    '''
    def _shared_step(self, batch, metric_fn):
        #x,target=batch
        x = torch.tensor(batch['images'])
        x=x.type_as(self.model.conv1.weight)
        target = torch.tensor(batch['labels'])
        target=target.type_as(self.model.conv1.weight)
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
    def fit(self, trainloader, valid_loader,max_epochs,gpus, overfit_batches):
        self.model.to('cuda')
        best_loss = float('inf')
        for epoch in range(args.max_epochs):
            train_step = 0
            epoch_loss = 0
            train_steps = len(trainloader)
            valid_steps = len(batcher_valid)
            print('-----------------------Training--------------------------------')
            model.train()
            for record in batcher_train:
                x = torch.tensor(record['images'], device='cuda')
                x = x.reshape(-1, x.shape[-1], x.shape[-3], x.shape[-2])
                target = torch.tensor(record['labels'], device='cuda')
                output = model(x).squeeze(-1)

                train_loss = criterion(output, target)
                optimizer.zero_grad()

                train_loss.backward()

                optimizer.step()
                epoch_loss += train_loss.item()
                # print statistics
                print(f'Epoch {epoch} training Step {train_step}/{train_steps} train_loss {train_loss.item()}')
                if train_step % 10 == 0:
                    writer.add_scalar("Loss/train", train_loss, train_step)
                    # wandb.log({"train_loss": train_loss})
                train_step += 1

            avgloss = epoch_loss / train_steps
            print(f'End of Epoch training average Loss is {avgloss}')


    def load_from_checkpoint(path, model):
        print(f'loading the model from saved checkpoint at {path}')
        model.load_state_dict(torch.load(path))
        model.eval()
        return model

    def training_step(self, batch, batch_idx):
        opt=self.configure_optimizers()['optimizer']
        scheduler=self.configure_optimizers()['lr_scheduler']['scheduler']
        train_loss = self._shared_step(batch, self.metric)
        opt.zero_grad()

        train_loss.backward()

        opt.step()


        # log the outputs!
        self.log('train_loss', loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)

        return {'train_loss': loss}

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
    '''
    def validation_epoch_end(self, validation_step_outputs):
        self.log(f'val_metric_epoch', self.metric.compute(), prog_bar=True, )

        # reset after each epoch
        self.metric.reset()
    '''
    def test_step(self, batch, batch_idx):
        loss = self._shared_step(batch, self.metric)

        # log the outputs!
        self.log('test_loss', loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)

        return {'loss': loss}
    '''
    def test_epoch_end(self, test_step_outputs):
        self.log(f'test_metric_epoch', self.metric.compute(), prog_bar=True, )

        # reset after each epoch
        self.metric.reset()
    '''
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return {
            'optimizer': opt,
            'lr_scheduler': {
                'scheduler': ExponentialLR(opt,
                                           gamma=args.lr_decay,
                                           verbose=True),
                'monitor': 'val_loss',
            }
        }

    def setup_criterion(self):

        if self.loss_type == 'classification':
            print('i shoudnot be here')
            self.criterion = nn.CrossEntropyLoss()

        elif self.loss_type == 'regression':
            print('hiiii')
            self.criterion = nn.MSELoss()

# lr=self.learning_rate * (self.lr_decay ** self.epoch)
