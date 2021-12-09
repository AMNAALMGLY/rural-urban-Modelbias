import os
import shutil
import time
from collections import defaultdict

import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR
import torch
import torch.nn as nn
from tqdm import tqdm

from utils.utils import Metric
from configs import args
import wandb
writer = SummaryWriter()
patience = args.patience


class Trainer:
    """A trainer class for model traiing
     ...

    Attributes
    ----------
       - model
       - lr
       - weight_decay
       -loss_type
       -num_outputs
       -metric
       -optimizor
       -scheduler
       -criterion

    Methods:
    --------
        -init :initalization
        -shared_step: shared training step between training, validation and testing
        -training_step: shared step implemented for training data
        -validation_step: shared step implemented for validation data
        -fit : do the training loop over the dataloaders
        -setup criterion: sets loss function
        -configure optimizor:setup optim and scheduler
    """

    def __init__(self, model, lr, weight_decay, loss_type, num_outputs, metric, save_dir, **kwargs):

        '''Initializes the Trainer.
        Args

        - model:PreAct Model class
        - lr: int learning_rate
        - weight_decay: int
        - loss_type: str, one of ['classification', 'regression']
        - num_outputs:output class  one of [None ,num_classes]
        - metric:List[str]  one of ['r2','R2' ,'mse', 'rank']


        '''
        super().__init__()

        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_type = loss_type
        self.save_dir = save_dir
        self.num_outputs=num_outputs
        if num_outputs is not None:

            fc = nn.Linear(model.fc.in_features, num_outputs)
            # initialization
            torch.nn.init.trunc_normal_(fc.weight.data, std=0.01)
            torch.nn.init.constant_(fc.bias.data, 0)

            model.fc = fc


        else:

            raise ValueError('please specify a value for your number of outputs for the loss function to evaluate '
                             'against')

        self.metric = Metric().get_metric(metric)  # TODO if it is a list

        self.scheduler = self.configure_optimizers()['lr_scheduler']['scheduler']
        self.opt = self.configure_optimizers()['optimizer']

        self.setup_criterion()

    def _shared_step(self, batch, metric_fn):
        if args.include_buildings:
            x = torch.tensor(batch[0]['images'], )
            b = torch.tensor(batch[1]['buildings'], )
            x = torch.cat((x, b), dim=-1)
            target = torch.tensor(batch[0]['labels'], )


        else:
            x = torch.tensor(batch['images'])
            target = torch.tensor(batch['labels'], )

        x = x.type_as(self.model.conv1.weight)


        target = target.type_as(self.model.conv1.weight)
        x = x.reshape(-1, x.shape[-1], x.shape[-3], x.shape[-2])  # [batch_size ,in_channels, H ,W]
        if self.loss_type == 'classification':
            #target=target.reshape(-1,1)

            target =torch.nn.functional.one_hot(target.long())
            print(target.shape)

        outputs = self.model(x)
        outputs = outputs.squeeze(dim=-1)

        loss = self.criterion(outputs, target.float())

        if self.loss_type == 'classification' and self.num_outputs >2:

            preds = nn.functional.softmax(outputs, dim=1)
        elif self.loss_type=='classification' and self.num_outputs==2:
            preds=nn.functional.sigmoid(outputs,dim=1)
        else:
            preds = torch.tensor(outputs, device='cuda')

        metric_fn.to('cuda')
        metric_fn.update(preds, target)

        return loss

    def training_step(self, batch, ):

        train_loss = self._shared_step(batch, self.metric)
        self.opt.zero_grad()

        train_loss.backward()

        self.opt.step()

        # log the outputs!

        return train_loss

    def validation_step(self, batch, ):
        loss = self._shared_step(batch, self.metric)

        return loss

    def test_step(self, batch, ):
        loss = self._shared_step(batch, self.metric)

        return loss

    def fit(self, trainloader, validloader, max_epochs, gpus, early_stopping=True, save_every=10, overfit_batches=None):

        self.model.to(gpus)
        # log the gradients
        wandb.watch(self.model, log='all')
        train_steps = len(trainloader)

        valid_steps = len(validloader)  # the number of batches
        best_loss = float('inf')
        count2 = 0  # count loss improving times
        r2_dict = defaultdict(lambda x: '')
        resume_path = None
        val_list = defaultdict(lambda x: '')
        start = time.time()

        for epoch in range(max_epochs):
            epoch_start = time.time()

            with tqdm(trainloader, unit="batch") as tepoch:
                train_step = 0
                epoch_loss = 0

                print('-----------------------Training--------------------------------')
                self.model.train()
                self.opt.zero_grad()
                for record in tepoch:
                    tepoch.set_description(f"Epoch {epoch}")

                    train_loss = self._shared_step(record, self.metric)
                    train_loss.backward()
                    # Implementing gradient accumlation
                    if (train_step+1) % args.accumlation_steps == 0:
                        self.opt.step()
                        self.opt.zero_grad()

                    epoch_loss += train_loss.item()
                    train_step += 1
                    # print statistics
                    print(f'Epoch {epoch} training Step {train_step}/{train_steps} train_loss {train_loss.item():.2f}')
                    if (train_step+1) % 20 == 0:
                        running_train = epoch_loss / (train_step)
                        wandb.log({"train_loss": running_train, 'epoch': epoch})


                    tepoch.set_postfix(loss=train_loss.item())
                    time.sleep(0.1)

            # Metric calulation and average loss
            r2 = self.metric.compute()
            wandb.log({'r2_train': r2, 'epoch': epoch})
            avgloss = epoch_loss / train_steps
            wandb.log({"Epoch_train_loss": avgloss, 'epoch': epoch})
            print(f'End of Epoch training average Loss is {avgloss:.2f} and R2 is {r2:.2f}')
            self.metric.reset()
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
                        f'Epoch {epoch} validation Step {valid_step}/{valid_steps} validation_loss {valid_loss.item():.2f}')
                    if (valid_step+1) % 20 == 0:
                        running_loss = valid_epoch_loss / (valid_step)
                        wandb.log({"valid_loss": running_loss, 'epoch': epoch})

                avg_valid_loss = valid_epoch_loss / valid_steps  # maybe valid steps is wrong

                r2_valid = self.metric.compute()
                print(f'Validation R2 is {r2_valid:.2f} and loss {avg_valid_loss}')
                wandb.log({'r2_valid': r2_valid, 'epoch': epoch})
                wandb.log({"Epoch_valid_loss": avg_valid_loss, 'epoch': epoch})

                # early stopping with r2:
                '''
             
                if r2_valid - best_valid>=0.05:
                    #best_loss = avg_valid_loss
                    wait=0
                    best_valid=r2_valid
                    if epoch > 100:
                        save_path = os.path.join(self.save_dir, f'best at metric Epoch{epoch}.ckpt')
                        torch.save(self.model.state_dict(), save_path)
                        print(f'best model at metric at Epoch {epoch}  metric {r2_valid} ,')
                        print(f'Path to best model found during training: \n{save_path}')
                else:
                    wait+=1
                    if wait >= patience and early_stopping:
                        print('.................Early Stopping .....................')
                        break
                        
    
                '''
                # early stopping with loss
                if best_loss - avg_valid_loss >= 0:
                    print('in loss improving loop by ')
                    print(best_loss - avg_valid_loss)
                    # loss is improving
                    counter = 0
                    count2 += 1
                    best_loss = avg_valid_loss
                    # start saving after a threshold of epochs and a patience of improvement
                    if epoch >= 100 and count2 >= 1:
                        print('in best path saving')
                        save_path = os.path.join(self.save_dir, f'best_Epoch{epoch}.ckpt')
                        torch.save(self.model.state_dict(), save_path)
                        # save r2 values and loss values
                        r2_dict[r2_valid] = save_path
                        val_list[avg_valid_loss] = save_path
                        print(f'best model  in loss at Epoch {epoch} loss {avg_valid_loss} ')
                        print(f'Path to best model at loss found during training: \n{save_path}')
                elif best_loss - avg_valid_loss < 0:
                    # loss is degrading
                    print('in loss degrading loop by :')
                    print(best_loss - avg_valid_loss)
                    counter += 1  # degrading tracker
                    count2 = 0  # improving tracker
                    if counter >= patience and early_stopping:
                        print(f'.................Early stopping can be in this Epoch{epoch}.....................')
                        #break

            # Saving the model for later use every 10 epochs:
            if epoch % save_every == 0:
                resume_dir = os.path.join(self.save_dir, 'resume_points')
                os.makedirs(resume_dir, exist_ok=True)
                resume_path = os.path.join(resume_dir, f'Epoch{epoch}.ckpt')
                torch.save(self.model.state_dict(), resume_path)
                print(f'Saving model to {resume_path}')

            self.metric.reset()
            self.scheduler.step()

            print("Time Elapsed for one epochs : {:.4f}m".format((time.time() - epoch_start) / 60))

        # choose the best model between the saved models in regard to r2 value or minimum loss
        if len(val_list.keys()) > 0:
            best_path = val_list[min(val_list.keys())]
            print(f'loss of best model saved is {min(val_list.keys())}')

            shutil.move(best_path,
                        os.path.join(self.save_dir, 'best.ckpt'))

        elif len(r2_dict.keys()) > 0:
            best_path = r2_dict[max(r2_dict.keys())]


            shutil.move(best_path,
                        os.path.join(self.save_dir, 'best.ckpt'))


        else:
            # best path is the last path which is saved at resume_points dir
            best_path = resume_path
            print(f'loss of best model saved from resume_point is {avg_valid_loss}')
            shutil.move(os.path.join(self.save_dir, best_path.split('/')[-2], best_path.split('/')[-1]),
                        os.path.join(self.save_dir, 'best.ckpt'))

            # better_path=best_path

        print("Time Elapsed for all epochs : {:.4f}m".format((time.time() - start) / 60))
        best_path = os.path.join(self.save_dir, 'best.ckpt')
        return best_loss, best_path,
        # TODO implement overfit batches
        # TODO savelast

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return {
            'optimizer': opt,
            'lr_scheduler': {
                 'scheduler': ExponentialLR(opt,
                              gamma=args.lr_decay),
                #'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)
                # 'scheduler':torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min')

            }
        }

    def setup_criterion(self):

        if self.loss_type == 'classification' and self.num_outputs >2:
            self.criterion = nn.CrossEntropyLoss()
        elif self.loss_type=='classification' and self.num_outputs==2:
            self.criterion=nn.BCELoss()

        elif self.loss_type == 'regression':
            self.criterion = nn.MSELoss()
        print(self.criterion)

    def weight_ex(self,x,class_model):
        '''
        use binary classification for finiding weights for data
        :return: data weighted by exp(h(x))
        '''
        hx=class_model(x)
        x=torch.exp(hx) * x
        return x