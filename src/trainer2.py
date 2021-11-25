import os
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
import  wandb
writer = SummaryWriter()
patience=6

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
            #initialization
            torch.nn.init.normal_(fc.weight)
            fc.bias.data.fill_(0)

            model.fc = fc


        else:

            raise ValueError('please specify a value for your number of outputs for the loss function to evaluate '
                             'against')

        self.metric = Metric().get_metric(metric)  # TODO if it is a list

        self.scheduler = self.configure_optimizers()['lr_scheduler']['scheduler']
        self.opt=self.configure_optimizers()['optimizer']

        #self.metric=Accuracy(num_classes=10)
        self.setup_criterion()

    '''
    def forward(self, x):



        output = self.model(x)

        return output
    '''
    def _shared_step(self, batch, metric_fn):
        print(batch)
        x = torch.tensor(batch['images'],)

        x=x.type_as(self.model.conv1.weight)
        target = torch.tensor(batch['labels'],)
        target=target.type_as(self.model.conv1.weight)
        x = x.reshape(-1, x.shape[-1], x.shape[-3], x.shape[-2])  # [batch_size ,in_channels, H ,W]

        outputs = self.model(x)
        outputs = outputs.squeeze(dim=-1)


        loss = self.criterion(outputs, target)

        if self.loss_type == 'classification':

            preds = nn.functional.softmax(outputs, dim=1,device='cuda')
        else:
            preds = torch.tensor(outputs,device='cuda')

        metric_fn.to('cuda')
        metric_fn.update(preds, target)

        return loss
    def fit(self, trainloader, validloader,max_epochs,gpus,early_stopping=True,save_every=10 ,overfit_batches=None):


        self.model.to(gpus)
        # log the gradients
        wandb.watch(self.model, log='all')
        train_steps = len(trainloader)
        valid_steps = len(validloader)
        best_loss = float('inf')
        count2=0
        r2_dict=defaultdict(lambda x:'')
        start=time.time()

        for epoch in range(max_epochs):
            with tqdm(trainloader, unit="batch") as tepoch:
                train_step = 0
                epoch_loss = 0

                print('-----------------------Training--------------------------------')
                self.model.train()
                for record in tepoch:
                    tepoch.set_description(f"Epoch {epoch}")
                    train_loss=self.training_step(record,)
                    epoch_loss += train_loss.item()
                    train_step += 1
                    # print statistics
                    print(f'Epoch {epoch} training Step {train_step}/{train_steps} train_loss {train_loss.item():.2f}')
                    if train_step % 20 == 0:
                        running_train=epoch_loss/(train_step)
                        writer.add_scalar("Loss/train", running_train, train_step)
                        wandb.log({"train_loss": running_train})

                    tepoch.set_postfix(loss=train_loss.item())
                    time.sleep(0.1)

            #Metric calulation and average loss
            r2=self.metric.compute()
            wandb.log({'r2_train':r2})
            avgloss = epoch_loss / train_steps
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
                        f'Epoch {epoch} validation Step {valid_step}/{valid_steps} vatrain_steplidation_loss {valid_loss.item():.2f}')
                    if valid_step % 20 == 0:
                        running_loss=valid_epoch_loss/(valid_step)
                        writer.add_scalar("Loss/valid", running_loss, valid_step)
                        wandb.log({"valid_loss": running_loss})
                avg_valid_loss=valid_epoch_loss / valid_steps
                r2_valid=self.metric.compute()
                print(f'Validation R2 is {r2_valid:.2f}')
                wandb.log({'r2_valid': r2_valid})

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
                if best_loss - avg_valid_loss >= 0.1:
                    # loss is improving
                    counter = 0
                    count2 += 1
                    best_loss = avg_valid_loss
                    # start saving after a threshold of epochs and a patience of improvement
                    if epoch >= 70 and count2 >= patience:
                        save_path = os.path.join(self.save_dir, f'best  at loss Epoch{epoch}.ckpt')
                        torch.save(self.model.state_dict(), save_path)
                        # save r2 values
                        r2_dict[r2_valid] = save_path
                        print(f'best model  in loss at Epoch {epoch} loss {avg_valid_loss} ')
                        print(f'Path to best model at loss found during training: \n{save_path}')
                elif best_loss - avg_valid_loss < 0.1:
                    # loss is degrading
                    counter += 1  # degrading tracker
                    count2 = 0  # improving tracker
                    if counter >= patience and early_stopping:
                        print('.................Early Stopping .....................')
                        break

                    resume_path = os.path.join(resume_dir, f'Epoch{epoch}.ckpt')
                    torch.save(self.model.state_dict(), resume_path)
                    print(f'Saving model to {resume_path}')



            #Saving the model for later use every 10 epochs:
            if epoch%save_every==0:
                resume_dir=os.path.join(self.save_dir,'resume_points')
                os.makedirs(resume_dir, exist_ok=True)
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

            self.metric.reset()
            self.scheduler.step()
        #choose the best model between the saved models in regard to r2 value
        best_path=r2_dict[max(r2_dict)]
        del r2_dict[max(r2_dict)]
        better_path=r2_dict[max(r2_dict)]

        os.rename(best_path,os.path.join(self.save_dir,'best.ckpt'))
        os.rename(better_path,os.path.join(self.save_dir,'better.ckpt'))
        print("Time Elapsed for all epochs : {:.4f}s".format(time.time() - start))

        return best_loss, best_path,better_path
        #TODO implement overfit batches
        #TODO savelast




    def training_step(self, batch,):


        train_loss = self._shared_step(batch, self.metric)
        self.opt.zero_grad()

        train_loss.backward()

        self.opt.step()

        # log the outputs!

        return train_loss


    def validation_step(self, batch,):
        loss = self._shared_step(batch, self.metric)

        return  loss

    def test_step(self, batch, ):
        loss = self._shared_step(batch, self.metric)

        return loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
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

