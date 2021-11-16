import os

from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR
import torch
import torch.nn as nn
from utils.utils import Metric
from configs import args

writer = SummaryWriter()

class Trainer:
    def __init__(self, model, lr, weight_decay, loss_type, num_outputs, metric, save_dir, **kwargs):

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
        self.save_dir=save_dir
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
    def fit(self, trainloader, validloader,max_epochs,gpus, overfit_batches=None):
        self.model.to(gpus)

        scheduler = self.configure_optimizers()['lr_scheduler']['scheduler']
        best_loss = float('inf')
        start=time.time()
        for epoch in range(max_epochs):
            train_step = 0
            epoch_loss = 0
            train_steps = len(trainloader)
            valid_steps = len(validloader)
            print('-----------------------Training--------------------------------')
            self.model.train()
            for record in trainloader:
                train_loss=self.training_step(self, record,)
                epoch_loss += train_loss.item()
                # print statistics
                print(f'Epoch {epoch} training Step {train_step}/{train_steps} train_loss {train_loss.item()}')
                if train_step % 10 == 0:
                    writer.add_scalar("Loss/train", train_loss, train_step)
                    # wandb.log({"train_loss": train_loss})
                train_step += 1

            #Metric calulation and average loss

            avgloss = epoch_loss / train_steps
            print(f'End of Epoch training average Loss is {avgloss} ')

            with torch.no_grad():
                valid_step = 0
                valid_epoch_loss = 0
                print('--------------------------Validation-------------------- ')
                self.model.eval()
                for record in validloader:

                    valid_loss = self.validation_step(record)
                    valid_epoch_loss += valid_loss.item()
                    valid_step += 1
                    print(
                        f'Epoch {epoch} validation Step {valid_step}/{valid_steps} validation_loss {valid_loss.item()}')
                    if valid_step % 10 == 0:
                        writer.add_scalar("Loss/valid", valid_loss, valid_step)
                        # wandb.log({"valid_loss": valid_loss})
                avg_valid_loss=valid_epoch_loss / valid_steps
                metric_valid=self.metric.compute()
            #Saving best model after a threshold of epochs:
            if avg_valid_loss< best_loss:
                best_loss = avg_valid_loss
                if epoch >100: #TODO changes this to config
                    save_path = os.path.join(self.save_dir, f'Epoch {epoch} loss {best_loss}.ckpt')
                    torch.save(self.model.state_dict(), save_path)
                    print(f'best average validation loss  is at Epoch {epoch} and is {best_loss} , and {self.metric} is {metric_valid}')
                    print(f'Path to best model found during training: \n{save_path}')
            self.metric.reset()
            scheduler.step()
        print("Time Elapsed : {:.4f}s".format(time.time() - start))
        return best_loss, metric_valid, save_path
        #TODO implement overfit batches
        #TODO savelast
        #TODO resume from checkpoint

    def load_from_checkpoint(path, model):
        print(f'loading the model from saved checkpoint at {path}')
        model.load_state_dict(torch.load(path))
        model.eval()
        return model

    def training_step(self, batch,):
        opt=self.configure_optimizers()['optimizer']

        train_loss = self._shared_step(batch, self.metric)
        opt.zero_grad()

        train_loss.backward()

        opt.step()

        # log the outputs!

        return train_loss


    def validation_step(self, batch,):
        loss = self._shared_step(batch, self.metric)

        return  loss

    def test_step(self, batch, ):
        loss = self._shared_step(batch, self.metric)

        return loss

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
            self.criterion = nn.CrossEntropyLoss()

        elif self.loss_type == 'regression':
            self.criterion = nn.MSELoss()

