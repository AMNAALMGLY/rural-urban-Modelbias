import os
import shutil
import time
from collections import defaultdict

import numpy as np
from torch.cuda.amp import autocast
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR
import torch
import torch.nn as nn
from tqdm import tqdm

from batchers.dataset_constants_buildings import DHS_COUNTRIES
from utils.scheduler import StepLRScheduler
from utils.utils import Metric, save_results, get_sustain_labels
from configs import args
import wandb
import pandas as pd
from pl_bolts import optimizers
import tensorflow as tf
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import optuna
from optuna.trial import TrialState

patience = args.patience


class Trainer:
    """A trainer class for model training
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

    def __init__(self, model, lr, weight_decay, loss_type, num_outputs, metric, save_dir, sched,train_loader=None,valid_loader=None,test_loader=None, **kwargs):

        '''Initializes the Trainer.
        Args

        - model:PreAct Model class
        - lr: int learning_rate
        - weight_decay: int
        - loss_type: str, one of ['classification', 'mse','smoothL1']
        - num_outputs:output class  one of [None ,num_classes]
        - metric:List[str]  one of ['r2','R2' ,'mse', 'rank']


        '''
        super().__init__()

        self.label_name=args.label_name
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_type = loss_type
        self.save_dir = save_dir
        self.num_outputs = num_outputs


        if args.no_of_gpus > 1:
            self.model = nn.DataParallel(self.model)
            self.typeAs = self.model.module.fc
        else:
            self.typeAs = self.model.fc
        self.model.to(args.gpus)

        self.metric_str = metric
        self.metric = []
        for m in metric:
            self.metric.append(Metric(self.num_outputs).get_metric(m))

        self.scheduler = self.configure_optimizers()['lr_scheduler'][sched]
        ##Stochastic weight averaging
        self.swa_scheduler = self.configure_optimizers()['lr_scheduler']['swa_scheduler']

        self.swa_model = torch.optim.swa_utils.AveragedModel(self.model)
        self.opt = self.configure_optimizers()['optimizer']

        self.setup_criterion()

        #for raytune experiments
        self.loader_train=train_loader
        self.loader_valid=valid_loader
        self.loader_test=test_loader

    def _shared_step(self, batch, metric_fn, is_training=True):
        '''
        this functions assumes 4 types of input :building band (buildings), multispecttral(images)
        , Nightlight (images) , combined bands (nl,ms) and metadata (meta)
        split the batch input into these 4 inputs  or more into a dictionary as follows:
        'images' 'buildings' ,'ms' , 'merge' , 'locs' ,'country'
        if they are not none ofcourse, (images and ms are mutually exclusive)

        #TODO do this in the dataset class
        #TODO is grouping necessary?
        :param batch:
        :param metric_fn:
        :param is_training:
        :return:
        '''
        '''
        if args.include_buildings:
            if args.ls_bands or args.nl_band :
                x = torch.tensor(batch[0]['images'], )
                b = torch.tensor(batch[1]['buildings'], )
                x = torch.cat((x, b), dim=-1)
                target = torch.tensor(batch[0]['labels'], )



            else:
                x = torch.tensor(batch[1]['buildings'])
                target = torch.tensor(batch[0]['labels'])
            if 'urban_rural' in args.scaler_features_keys:
                group = torch.tensor(batch[0]['urban_rural'])



        else:
           if args.ls_bands or args.nl_band:
                x = torch.tensor(batch['images'])
           target = torch.tensor(batch['labels'], )
           if 'urban_rural' in args.scaler_features_keys:
                group = torch.tensor(batch['urban_rural'])

        x = x.type_as(self.model.fc.weight)

        target = target.type_as(self.model.fc.weight)
        x = x.reshape(-1, x.shape[-1], x.shape[-3],                        # from [batch_size,H,W,in_channels] to [batch_size ,in_channels, H ,W]

                      x.shape[-2])  if args.input=='images' else x
        '''
        x = defaultdict()

        if args.include_buildings:
            if args.ls_bands and args.nl_band:
                # 2 bands split them inot seperate inputs
                # assumes for now it is only merged nl_bands
                x[args.ls_bands] = torch.tensor(batch[0]['images'][:, :, :, :-1], )

                x[args.nl_band] = torch.tensor(batch[0]['images'][:, :, :, -1]).unsqueeze(-1)
            elif args.ls_bands or args.nl_band:
                # only one type of band
                x['images'] = torch.tensor(batch[0]['images'])
            if args.metadata:
                for meta in args.metadata:
                    if meta == 'country':
                        batch[0][meta] = tf.map_fn(fn=lambda t: DHS_COUNTRIES.index(t), elems=batch[0][meta],
                                                   fn_output_signature=tf.int32)
                        batch[0][meta] = (tf.reshape(batch[0][meta], [-1, 1])).numpy()

                    x[meta] = torch.tensor(batch[0][meta], dtype=torch.int32)
                    if x[meta].dim() < 2:  # usqueeze the last dimension if I have only one dimension
                        x[meta].unsqueeze_(-1)
            lats = batch[0]['locs'][:, 0]
            longs = batch[0]['locs'][:, 1]
            if np.any(np.isnan(batch[0]['labels'])):
                print('in get sustainlabels')
                label = get_sustain_labels(lats, longs, self.label_name)  # TODO change this
            else:
                label = batch[0]['labels']
            target = torch.tensor(label, )
            # target = torch.tensor:wq
            # (batch[0]['labels'], )
            target = target.type_as(self.typeAs.weight)
            x['buildings'] = torch.tensor(batch[1]['buildings'], )


        else:
            if args.ls_bands and args.nl_band:
                # 2 bands split them to seperate inputs
                # assumes for now it is only merged nl_bands
                x[args.ls_bands] = torch.tensor(batch['images'][:, :, :, :-1])
                x[args.nl_band] = torch.tensor(batch['images'][:, :, :, -1]).unsqueeze(-1)
            elif args.ls_bands or args.nl_band:
                # only one type of band
                x['images'] = torch.tensor(batch['images'])
            if args.metadata:
                for meta in args.metadata:
                    if meta == 'country':
                        batch[meta] = tf.map_fn(fn=lambda t: DHS_COUNTRIES.index(t), elems=batch[meta])
                        batch[meta] = tf.map_fn(fn=lambda t: DHS_COUNTRIES.index(t), elems=batch[meta],
                                                fn_output_signature=tf.int32)
                        batch[meta] = tf.reshape(batch[meta], [-1, 1])
                    x[meta] = torch.tensor(batch[meta].numpy(), dtype=torch.int32)
            # label=torch.zeros((x['images'].shape[0]))
            lats = batch['locs'][:, 0]
            longs = batch['locs'][:, 1]
            if np.any(np.isnan(batch['labels'])):
                print('in get sustainlabels')
                label = get_sustain_labels(lats, longs, self.label_name)  # TODO change this
            else:
                label = batch['labels']
            target = torch.tensor(label, )
            # target = torch.tensor(batch['labels'], )
            print('target values', target[:10])
            target = target.type_as(self.typeAs.weight)

        x = {key: value.type_as(self.typeAs.weight) for key, value in x.items()}

        for key, value in x.items():
            x[key] = value.reshape(-1, value.shape[-1], value.shape[-3], value.shape[-2]) if value.dim() >= 3 else value
            # x = {key: value.reshape(-1, value.shape[-1], value.shape[-3], value.shape[-2]) for key, value in x.items() if
            #     value.dim() >= 3 else key:value}

        outputs = self.model(x)
        outputs = outputs.squeeze(dim=-1)
        # Re-weighting data
        if hasattr(self,'class_model') and self.class_model is not None:
            Beta = self.weight_ex(x['images'], self.class_model)

            outputs = outputs * (Beta ** 0.5)
        # Loss

        loss = self.criterion(outputs, target)
        if self.loss_type == 'custom':
            custom_loss = self.custom_loss(x['images'], target)
            # print(loss,custom_loss)
            loss = loss + args.lamda * custom_loss
            # print('total_loss',loss)
        # elif self.loss_type == 'subshift' and is_training:
        #    trainloss = self.subshift(x, target, group)
        else:
            trainloss = loss
        # Metric calculation
        if self.loss_type == 'classification' and self.num_outputs > 1:

            preds = nn.functional.softmax(outputs, dim=1)
            target = target.long()

        elif self.loss_type == 'classification' and self.num_outputs == 1:
            preds = torch.sigmoid(outputs, )
            target = target.long()
        elif self.loss_type=='quantile':
            outputs= self.criterion.to_prediction(outputs)
            print(outputs.shape)
            preds=torch.tensor(outputs, device=args.gpus)

        else:
            preds = torch.tensor(outputs, device=args.gpus)

        for fn in metric_fn:
            fn.to(args.gpus)
            fn.update(preds, target)

        return loss, trainloss

    def training_step(self, batch, ):

        _, train_loss = self._shared_step(batch, self.metric)
        self.opt.zero_grad()

        train_loss.backward()

        self.opt.step()

        # log the outputs!

        return train_loss

    def validation_step(self, batch, ):
        loss, subshift_loss = self._shared_step(batch, self.metric, is_training=False)

        return subshift_loss

    def test_step(self, batch, ):
        loss, _ = self._shared_step(batch, self.metric, is_training=False)

        return loss

    def fit(self, trainloader, validloader, batcher_test, max_epochs, gpus, class_model=None, early_stopping=True,
            save_every=25, args=args):
        # Weighting model
        if class_model:
            self.class_model = class_model.to(gpus)
        else:
            self.class_model = None
        # log the gradients
        wandb.watch(self.model, log='all', log_freq=5)

        swa_start = int(0.75 * max_epochs)  # swa in 25% of the training

        train_steps = len(trainloader)

        valid_steps = len(validloader)  # the number of batches
        best_loss = float('inf')
        count2 = 0  # count loss improving times
        r2_dict = defaultdict(lambda x: '')
        resume_path = None
        val_list = defaultdict(lambda x: '')
        start = time.time()

        # building_sum=[]

        for epoch in range(max_epochs):
            # scheduler updates
            # num_updates=epoch*len(trainloader)
            epoch_start = time.time()

            with tqdm(trainloader, unit="batch") as tepoch:
                train_step = 0
                epoch_loss = 0

                print('-----------------------Training--------------------------------')
                self.model.train()
                self.opt.zero_grad()
                for record in tepoch:
                    tepoch.set_description(f"Epoch {epoch}")

                    _, train_loss = self._shared_step(record, self.metric)
                    train_loss.backward()
                    # Implementing gradient accumlation
                    if (train_step + 1) % args.accumlation_steps == 0:
                        self.opt.step()
                        self.opt.zero_grad()

                    epoch_loss += train_loss.item()
                    train_step += 1
                    # print statistics
                    print(f'Epoch {epoch} training Step {train_step}/{train_steps} train_loss {train_loss.item():.2f}')
                    if (train_step + 1) % 20 == 0:
                        running_train = epoch_loss / (train_step)
                        wandb.log({"train_loss": running_train, 'epoch': epoch})

                    tepoch.set_postfix(loss=train_loss.item())
                    time.sleep(0.1)

                    # b=torch.tensor(record[1]['buildings'])

                    # building_sum.append(torch.sum(b ,dim=(1,2,3)))
            '''
            building_sum=torch.cat(building_sum,dim=0)
            print('shape of sum ',building_sum.shape)
            np_dict=defaultdict()
            np_dict['building_sum']=building_sum.numpy()
            
            save_results(self.save_dir, np_dict, 'building_sum')
            '''
            # Metric calulation and average loss
            r2 = (self.metric[0].compute()) ** 2 if self.metric_str[0] == 'r2' else self.metric[0].compute()
            wandb.log({f'{self.metric_str[0]} train': r2, 'epoch': epoch})
            avgloss = epoch_loss / train_steps
            wandb.log({"Epoch_train_loss": avgloss, 'epoch': epoch})
            print(f'End of Epoch training average Loss is {avgloss:.2f} and {self.metric_str[0]} is {r2:.2f}')
            self.metric[0].reset()
            building_sum = []
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
                    if (valid_step + 1) % 20 == 0:
                        running_loss = valid_epoch_loss / (valid_step)
                        wandb.log({"valid_loss": running_loss, 'epoch': epoch})
                # b=torch.tensor(record[1]['buildings'])
                # if epoch==0:
                #   building_sum.append(torch.sum(b ,dim=(1,2,3)))
                """
                if epoch==0:
                    building_sum=torch.cat(building_sum,dim=0)
                    print('shape of sum ',building_sum.shape)
                    np_dict=defaultdict()
                    np_dict['building_sum']=building_sum.numpy()

                    save_results(self.save_dir, np_dict, 'building_sum_val')
                """

                avg_valid_loss = valid_epoch_loss / valid_steps

                # tune.report(mean_loss=avg_valid_loss)

                r2_valid = (self.metric[0].compute()) ** 2 if self.metric_str[0] == 'r2' else self.metric[0].compute()
                self.metric[0].reset()
                print(f'Validation {self.metric_str[0]}is {r2_valid:.2f} and loss {avg_valid_loss}')
                wandb.log({f'{self.metric_str[0]} valid': r2_valid, 'epoch': epoch})
                wandb.log({"Epoch_valid_loss": avg_valid_loss, 'epoch': epoch})

                # AGAINST ML RULES : moniter test values
                r2_test, test_loss = self.test(batcher_test)

                wandb.log({f'{self.metric_str[0]} test': r2_test, 'epoch': epoch})
                wandb.log({f'loss test': test_loss, 'epoch': epoch})

                # early stopping with loss
                if best_loss - avg_valid_loss >= 0:
                    print('in loss improving loop by ')
                    print(best_loss - avg_valid_loss)
                    # loss is improving
                    counter = 0
                    count2 += 1
                    best_loss = avg_valid_loss
                    # start saving after a threshold of epochs and a patience of improvement
                    if count2 >= 1:
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
                        # break

            # Saving the model for later use every 10 epochs:
            if epoch % save_every == 0:
                resume_dir = os.path.join(self.save_dir, 'resume_points')
                os.makedirs(resume_dir, exist_ok=True)
                resume_path = os.path.join(resume_dir, f'Epoch{epoch}.ckpt')
                torch.save(self.model.state_dict(), resume_path)
                print(f'Saving model to {resume_path}')

            self.metric[0].reset()
            self.scheduler.step()
            # if epoch >= swa_start:
            #     print('in SWA scheduler')
            #     self.swa_model.update_parameters(self.model)
            #      self.swa_scheduler.step()
            #   else:
            #        self.scheduler.step(epoch + 1)

            print("Time Elapsed for one epochs : {:.2f}m".format((time.time() - epoch_start) / 60))
        # UPDATE SWA MODEL RUNNIGN MEAN AND VARIANCE
        # with autocast():
        #     Trainer.update_bn(trainloader, self.swa_model)

        # choose the best model between the saved models in regard to r2 value or minimum loss
        if len(r2_dict.keys()) > 0:
            best_path = r2_dict[max(r2_dict.keys())]
            print(f'{self.metric_str[0]} of best model saved is {max(r2_dict.keys())} , path {best_path}')

            shutil.move(best_path,
                        os.path.join(self.save_dir, 'best.ckpt'))

        elif len(val_list.keys()) > 0:
            best_path = val_list[min(val_list.keys())]
            print(f'loss of best model saved is {min(val_list.keys())} , path {best_path}')

            shutil.move(best_path,
                        os.path.join(self.save_dir, 'best.ckpt'))

   

        else:
            # best path is the last path which is saved at resume_points dir
            best_path = resume_path
            print(f'loss of best model saved from resume_point is {avg_valid_loss}')
            shutil.move(os.path.join(self.save_dir, best_path.split('/')[-2], best_path.split('/')[-1]),
                        os.path.join(self.save_dir, 'best.ckpt'))

            # better_path=best_path

            print("Time Elapsed for all epochs : {:.2} H".format((time.time() - start) / 120))
        best_path = os.path.join(self.save_dir, 'best.ckpt')
        return best_loss, best_path,
        # TODO implement overfit batches
        # TODO savelast

    def test(self, batcher_test):

        with torch.no_grad():
            test_step = 0
            test_epoch_loss = 0
            print('--------------------------Testing-------------------- ')
            self.model.eval()
            r2_test = []
            for record in batcher_test:
                test_epoch_loss += self.test_step(record).item()
                test_step += 1

            for i, m in enumerate(self.metric):
                r2_test.append((m.compute()) ** 2 if self.metric_str[i] == 'r2' else m.compute())

                # wandb.log({f'{self.metric_str[i]} Test': r2_test[i], })

        return r2_test[0], (test_epoch_loss / test_step)

    def train_ray(self, config):
        '''
        A seperate training function to adapt to ray tune hyper paramenters optimization setup
        :param config: hyper parameters you want to tune
        :return: loss
        '''

        self.lr = config['lr']
        self.wd = config['wd']
        self.configure_optimizers()
        args.accumlation_steps = config['bs']

        wandb.watch(self.model, log='all', log_freq=5)

        train_steps = len(self.loader_train)

        valid_steps = len(self.loader_valid)  # the number of batches




        for epoch in range(epochs):

            epoch_start = time.time()

            with tqdm(self.loader_train, unit="batch") as tepoch:
                train_step = 0
                epoch_loss = 0

                print('-----------------------Training--------------------------------')
                self.model.train()
                self.opt.zero_grad()
                for record in tepoch:
                    tepoch.set_description(f"Epoch {epoch}")

                    _, train_loss = self._shared_step(record, self.metric)
                    train_loss.backward()
                    # Implementing gradient accumlation
                    if (train_step + 1) % args.accumlation_steps == 0:
                        self.opt.step()
                        self.opt.zero_grad()

                    epoch_loss += train_loss.item()
                    train_step += 1
                    # print statistics
                    print(f'Epoch {epoch} training Step {train_step}/{train_steps} train_loss {train_loss.item():.2f}')
                    if (train_step + 1) % 20 == 0:
                        running_train = epoch_loss / (train_step)
                        #wandb.log({"train_loss": running_train, 'epoch': epoch})

                    tepoch.set_postfix(loss=train_loss.item())
                    time.sleep(0.1)

            # Metric calulation and average loss
            r2 = (self.metric[0].compute()) ** 2 if self.metric_str[0] == 'r2' else self.metric[0].compute()
            wandb.log({f'{self.metric_str[0]} train': r2, 'epoch': epoch})
            avgloss = epoch_loss / train_steps
            wandb.log({"Epoch_train_loss": avgloss, 'epoch': epoch})
            print(f'End of Epoch training average Loss is {avgloss:.2f} and {self.metric_str[0]} is {r2:.2f}')
            self.metric[0].reset()

            with torch.no_grad():
                valid_step = 0
                valid_epoch_loss = 0
                print('--------------------------Validation-------------------- ')
                self.model.eval()
                for record in self.loader_valid:

                    valid_loss = self.validation_step(record)
                    valid_epoch_loss += valid_loss.item()
                    valid_step += 1
                    print(
                        f'Epoch {epoch} validation Step {valid_step}/{valid_steps} validation_loss {valid_loss.item():.2f}')
                    if (valid_step + 1) % 20 == 0:
                        running_loss = valid_epoch_loss / (valid_step)
                #        wandb.log({"valid_loss": running_loss, 'epoch': epoch})

                avg_valid_loss = valid_epoch_loss / valid_steps
                with tune.checkpoint_dir(epoch) as checkpoint_dir:
                    path = os.path.join(self.save_dir, "ray_checkpoint")
                    torch.save((self.model.state_dict(), self.opt.state_dict()), path)



                r2_valid = (self.metric[0].compute()) ** 2 if self.metric_str[0] == 'r2' else self.metric[0].compute()
                self.metric[0].reset()
                print(f'Validation {self.metric_str[0]}is {r2_valid:.2f} and loss {avg_valid_loss}')
                wandb.log({f'{self.metric_str[0]} valid': r2_valid, 'epoch': epoch})
                wandb.log({"Epoch_valid_loss": avg_valid_loss, 'epoch': epoch})
                # AGAINST ML RULES : moniter test values
                r2_test, test_loss = self.test(self.loader_test)

                wandb.log({f'{self.metric_str[0]} test': r2_test, 'epoch': epoch})
                wandb.log({f'loss test': test_loss, 'epoch': epoch})

                tune.report(loss=avg_valid_loss, accuracy=r2_valid)

        #val_loss, _ = self.fit(data[0], data[1] ,data[2], epochs, tune_gpu, args=args)
        #val_loss, _ = self.fit_ray( epochs, tune_gpu, args=args)
        #tune.report(loss=val_loss)

    def tune_run(self, max_epochs, gpus):

        # just to use it in raytune function make loaders as global varibles
        global  epochs, tune_gpu
        #t_load, v_load, epochs, tune_gpu, test_load = trainloader, validloader, max_epochs, gpus, batcher_test
        epochs, tune_gpu=max_epochs, gpus
        tune_config = {"lr": tune.loguniform(1e-4, 1e-1),
                       "wd": tune.loguniform(1e-4, 1e1),
                       "bs": tune.choice([1, 2, 3, 4, 5])}

        reporter = CLIReporter(

            metric_columns=["loss", "training_iteration"])
        scheduler = ASHAScheduler(
            metric="loss",
            mode="min",
            max_t=200,
            grace_period=1,
            reduction_factor=2)
        result = tune.run(
            tune.with_parameters(self.train_ray),
            resources_per_trial={"cpu": 4, "gpu": args.gpus},
            config=tune_config,
            progress_reporter=reporter,
            metric="loss",
            mode="min",

            scheduler=scheduler,
            stop={
                "loss": 0.001,
                "training_iteration": 200
            },
            num_samples=1

        )
        best_trial = result.get_best_trial("loss", "min", "last")
        print("Best trial config: {}".format(best_trial.config))
        print("Best trial final validation loss: {}".format(
            best_trial.last_result["loss"]))
        return best_trial
    def train_optuna(self,trial):
        '''
        A seperate training function to adapt to ray tune hyper paramenters optimization setup
        :param config: hyper parameters you want to tune
        :return: loss
        '''

        self.lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)


        self.wd =  trial.suggest_float("wd", 1e-5, 1e1, log=True)
        self.opt=self.configure_optimizers()['optimizer']
        args.accumlation_steps = trial.suggest_int("bs", 1,5)

        wandb.watch(self.model, log='all', log_freq=5)

        train_steps = len(self.loader_train)

        valid_steps = len(self.loader_valid)  # the number of batches
        epochs=args.max_epochs
        for epoch in range(epochs):



            with tqdm(self.loader_train, unit="batch") as tepoch:
                train_step = 0
                epoch_loss = 0

                print('-----------------------Training--------------------------------')
                self.model.train()
                self.opt.zero_grad()
                for record in tepoch:
                    tepoch.set_description(f"Epoch {epoch}")

                    _, train_loss = self._shared_step(record, self.metric)
                    train_loss.backward()
                    # Implementing gradient accumlation
                    if (train_step + 1) % args.accumlation_steps == 0:
                        self.opt.step()
                        self.opt.zero_grad()

                    epoch_loss += train_loss.item()
                    train_step += 1
                    # print statistics
                    print(f'Epoch {epoch} training Step {train_step}/{train_steps} train_loss {train_loss.item():.2f}')


                    tepoch.set_postfix(loss=train_loss.item())
                    time.sleep(0.1)

            # Metric calulation and average loss
            r2 = (self.metric[0].compute()) ** 2 if self.metric_str[0] == 'r2' else self.metric[0].compute()

            avgloss = epoch_loss / train_steps

            print(f'End of Epoch training average Loss is {avgloss:.2f} and {self.metric_str[0]} is {r2:.2f}')
            self.metric[0].reset()

            with torch.no_grad():
                valid_step = 0
                valid_epoch_loss = 0
                print('--------------------------Validation-------------------- ')
                self.model.eval()
                for record in self.loader_valid:

                    valid_loss = self.validation_step(record)
                    valid_epoch_loss += valid_loss.item()
                    valid_step += 1
                    print(
                        f'Epoch {epoch} validation Step {valid_step}/{valid_steps} validation_loss {valid_loss.item():.2f}')

                avg_valid_loss = valid_epoch_loss / valid_steps
                #with tune.checkpoint_dir(epoch) as checkpoint_dir:
                 #   path = os.path.join(self.save_dir, "ray_checkpoint")
                  #  torch.save((self.model.state_dict(), self.opt.state_dict()), path)



                r2_valid = (self.metric[0].compute()) ** 2 if self.metric_str[0] == 'r2' else self.metric[0].compute()
                self.metric[0].reset()
                print(f'Validation {self.metric_str[0]}is {r2_valid:.2f} and loss {avg_valid_loss}')


                trial.report(avg_valid_loss,epoch)
                # Handle pruning based on the intermediate value.
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
        return avg_valid_loss
    def run_optuna(self):
        study = optuna.create_study(direction="minimize")
        study.optimize(self.train_optuna, n_trials=5)

        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
        return trial
    def configure_optimizers(self):
        if args.scheduler == 'cyclic':
            opt = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        else:
            opt = torch.optim.Adam(self.model.parameters(), lr=self.lr,
                                   weight_decay=self.weight_decay)

        return {
            'optimizer': opt,
            'lr_scheduler': {
                'exp': ExponentialLR(opt,
                                     gamma=args.lr_decay),
                'cos': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=args.max_epochs),
                'warmup_cos': optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR(opt, warmup_epochs=5,
                                                                                    max_epochs=200,
                                                                                    warmup_start_lr=1e-8),
                'warmup_step': StepLRScheduler(opt, decay_t=1, decay_rate=args.lr_decay, warmup_t=5,
                                               warmup_lr_init=1e-7),
                'step': torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=args.lr_decay, ),
                'cyclic': torch.optim.lr_scheduler.CyclicLR(opt, base_lr=1e-4, max_lr=self.lr, cycle_momentum=False),
                # 'scheduler':torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min'),

                'swa_scheduler': torch.optim.swa_utils.SWALR(opt, anneal_strategy="cos", anneal_epochs=5, swa_lr=0.05)

            }
        }

    def setup_criterion(self):

        if self.loss_type == 'classification' and self.num_outputs > 1:
            self.criterion = nn.CrossEntropyLoss()
        elif self.loss_type == 'classification' and self.num_outputs == 1:
            self.criterion = nn.BCEWithLogitsLoss()

        elif self.loss_type == 'mse':
            self.criterion = nn.MSELoss()
        elif self.loss_type=='l1':
            self.criterion=nn.L1Loss()
        elif self.loss_type == 'smoothL1':
            self.criterion = nn.SmoothL1Loss(beta=3)
        elif self.loss_type=='quantile':
            self.criterion=QuantileLoss()

    @torch.no_grad()
    def update_bn(loader, model, device=None):
        r"""Updates BatchNorm running_mean, running_var buffers in the model.
        It performs one pass over data in `loader` to estimate the activation
        statistics for BatchNorm layers in the model.
        Args:
            loader (torch.utils.data.DataLoader): dataset loader to compute the
                activation statistics on. Each data batch should be either a
                tensor, or a list/tuple whose first element is a tensor
                containing data.
            model (torch.nn.Module): model for which we seek to update BatchNorm
                statistics.
            device (torch.device, optional): If set, data will be transferred to
                :attr:`device` before being passed into :attr:`model`.
        Example:

        .. note::
            The `update_bn` utility assumes that each data batch in :attr:`loader`
            is either a tensor or a list or tuple of tensors; in the latter case it
            is assumed that :meth:`model.forward()` should be called on the first
            element of the list or tuple corresponding to the data batch.
        """
        momenta = {}
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.running_mean = torch.zeros_like(module.running_mean)
                module.running_var = torch.ones_like(module.running_var)
                momenta[module] = module.momentum

        if not momenta:
            return

        was_training = model.training
        model.train()
        for module in momenta.keys():
            module.momentum = None
            module.num_batches_tracked *= 0
        x = defaultdict()
        for input in loader:

            key = list(input.keys())[0]
            # if isinstance(input, (list, tuple)):

            # x['buildings']=torch.tensor( input[1]['buildings'])
            x[key] = torch.tensor(input[key])
            x = {key: value.reshape(-1, value.shape[-1], value.shape[-2], value.shape[-3]) for key, value in x.items()}
            # x={key:value.type_as(model.fc.weight) for key , value in x.items()}
            input = x
            if device is not None:
                input = input.to(device)
            model(input)
        print('......in SWA...............')
        for bn_module in momenta.keys():
            bn_module.momentum = momenta[bn_module]
        model.train(was_training)

    def custom_loss(self, batch, target):
        losses = []

        sorted, indices = torch.sort(target, descending=False, dim=0)
        batch = batch[indices]
        target = target[indices]

        # print(sorted==target)
        quantiles_x = torch.split(batch, 5)
        quantiles_y = torch.split(target, 5)
        for i in range(5):
            losses.append(torch.nn.functional.mse_loss(self.model(quantiles_x[i]).squeeze(-1), quantiles_y[i]))
        # print('losses_list',losses)
        return max(losses)

    def weight_ex(self, x, class_model):
        '''
        use binary classification for finiding weights for data
        :return: weghing factor Beta that can be added to loss function
        '''

        hx = torch.sigmoid(class_model(x))

        Beta = torch.exp(hx.squeeze(-1))
        return Beta


    def fit_wilds(self, trainloader, validloader, max_epochs, gpus, class_model=None, early_stopping=True,
                  save_every=10):
        self.model.to(gpus)

        # Weighting model
        if class_model:
            self.class_model = class_model.to(gpus)
        else:
            self.class_model = None
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
                for x, y, in tepoch:
                    tepoch.set_description(f"Epoch {epoch}")

                    x = x.type_as(self.typeAs.weight)
                    y = y.type_as(self.typeAs.weight)
                    # x = dict(images=x)
                    outputs = self.model(x)
                    outputs = outputs.squeeze(dim=-1)
                    y = y.squeeze(dim=-1)
                    train_loss = self.criterion(outputs, y)
                    train_loss.backward()
                    # Implementing gradient accumlation
                    if (train_step + 1) % args.accumlation_steps == 0:
                        self.opt.step()
                        self.opt.zero_grad()

                    epoch_loss += train_loss.item()
                    train_step += 1
                    # print statistics
                    print(f'Epoch {epoch} training Step {train_step}/{train_steps} train_loss {train_loss.item():.2f}')
                    if (train_step + 1) % 20 == 0:
                        running_train = epoch_loss / (train_step)
                        wandb.log({"train_loss": running_train, 'epoch': epoch})

                    tepoch.set_postfix(loss=train_loss.item())
                    time.sleep(0.1)

                    preds = torch.tensor(outputs, device=args.gpus)
                    self.metric[0].to(args.gpus)

                    self.metric[0].update(preds, y)
                    self.scheduler.step()

            # Metric calulation and average loss
            r2 = (self.metric[0].compute()) ** 2 if self.metric_str[0] == 'r2' else self.metric[0].compute()
            wandb.log({f'{self.metric_str[0]} train': r2, 'epoch': epoch})
            avgloss = epoch_loss / train_steps
            wandb.log({"Epoch_train_loss": avgloss, 'epoch': epoch})
            print(f'End of Epoch training average Loss is {avgloss:.2f} and {self.metric_str[0]} is {r2:.2f}')
            self.metric[0].reset()
            with torch.no_grad():
                valid_step = 0
                valid_epoch_loss = 0
                print('--------------------------Validation-------------------- ')
                self.model.eval()
                for x, y, in validloader:
                    x = x.type_as(self.typeAs.weight)
                    y = y.type_as(self.typeAs.weight)
                    # x=dict(images=x)
                    outputs = self.model(x)
                    outputs = outputs.squeeze(dim=-1)
                    y = y.squeeze(dim=-1)
                    valid_loss = self.criterion(outputs, y)

                    valid_epoch_loss += valid_loss.item()
                    valid_step += 1
                    print(
                        f'Epoch {epoch} validation Step {valid_step}/{valid_steps} validation_loss {valid_loss.item():.2f}')
                    if (valid_step + 1) % 20 == 0:
                        running_loss = valid_epoch_loss / (valid_step)
                        wandb.log({"valid_loss": running_loss, 'epoch': epoch})
                    preds = torch.tensor(outputs, device=args.gpus)
                    self.metric[0].to(args.gpus)
                    self.metric[0].update(preds, y)

                avg_valid_loss = valid_epoch_loss / valid_steps

                r2_valid = (self.metric[0].compute()) ** 2 if self.metric_str[0] == 'r2' else self.metric[0].compute()

                print(f'Validation {self.metric_str[0]}is {r2_valid:.2f} and loss {avg_valid_loss}')
                wandb.log({f'{self.metric_str[0]} valid': r2_valid, 'epoch': epoch})
                wandb.log({"Epoch_valid_loss": avg_valid_loss, 'epoch': epoch})

                # early stopping with loss
                if best_loss - avg_valid_loss >= 0:
                    print('in loss improving loop by ')
                    print(best_loss - avg_valid_loss)
                    # loss is improving
                    counter = 0
                    count2 += 1
                    best_loss = avg_valid_loss
                    # start saving after a threshold of epochs and a patience of improvement
                    if count2 >= 1:
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
                        # break

            # Saving the model for later use every 10 epochs:
            if epoch % save_every == 0:
                resume_dir = os.path.join(self.save_dir, 'resume_points')
                os.makedirs(resume_dir, exist_ok=True)
                resume_path = os.path.join(resume_dir, f'Epoch{epoch}.ckpt')
                torch.save(self.model.state_dict(), resume_path)
                print(f'Saving model to {resume_path}')

            self.metric[0].reset()
            # self.scheduler.step()

            print("Time Elapsed for one epochs : {:.2f}m".format((time.time() - epoch_start) / 60))

        # choose the best model between the saved models in regard to r2 value or minimum loss
        if len(val_list.keys()) > 0:
            best_path = val_list[min(val_list.keys())]
            print(f'loss of best model saved is {min(val_list.keys())} , path {best_path}')

            shutil.move(best_path,
                        os.path.join(self.save_dir, 'best.ckpt'))

        elif len(r2_dict.keys()) > 0:
            best_path = r2_dict[max(r2_dict.keys())]
            print(f'{self.metric_str[0]} of best model saved is {max(r2_dict.keys())} , path {best_path}')

            shutil.move(best_path,
                        os.path.join(self.save_dir, 'best.ckpt'))


        else:
            # best path is the last path which is saved at resume_points dir
            best_path = resume_path
            print(f'loss of best model saved from resume_point is {avg_valid_loss}')
            shutil.move(os.path.join(self.save_dir, best_path.split('/')[-2], best_path.split('/')[-1]),
                        os.path.join(self.save_dir, 'best.ckpt'))

            # better_path=best_path

            print("Time Elapsed for all epochs : {:.2} H".format((time.time() - start) / 120))
        best_path = os.path.join(self.save_dir, 'best.ckpt')
        return best_loss, best_path,

    '''
    def subshift(self, x, y, group):
        sorted, indices = torch.sort(y, descending=False, dim=0)
        x = x[indices]
        y = y[indices]

        losses = []
        urban_index = (group == 1.0).nonzero(as_tuple=True)[0]
        rural_index = (group == 0.0).nonzero(as_tuple=True)[0]
        urban_x, urban_y = x[urban_index], y[urban_index]
        rural_x, rural_y = x[rural_index], y[rural_index]

        urban_grouped = torch.split(urban_x, 2)
        rural_grouped = torch.split(rural_x, 2)
        urban_y_grouped = torch.split(urban_y, 2)
        rural_y_grouped = torch.split(rural_y, 2)
        for i in range(2):
            losses.append(torch.nn.functional.mse_loss(self.model(urban_grouped[i]).squeeze(-1), urban_y_grouped[i]))
        for i in range(2):
            losses.append(torch.nn.functional.mse_loss(self.model(rural_grouped[i]).squeeze(-1), rural_y_grouped[i]))
        print(losses)
        return max(losses)
    '''
class QuantileLoss():
    """
    Quantile loss, i.e. a quantile of ``q=0.5`` will give half of the mean absolute error as it is calcualted as

    Defined as ``max(q * (y-y_pred), (1-q) * (y_pred-y))``
    """

    def __init__(
        self,
        quantiles = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98],
        **kwargs,
    ):
        """
        Quantile loss

        Args:
            quantiles: quantiles for metric
        """
        
        super(QuantileLoss, self).__init__()
        self.quantiles=quantiles

    def __call__(self, y_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # calculate quantile loss
        assert y_pred.size(0) == target.size(0) ,'prediction shape is not as target shape for quantile loss'

        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - y_pred[..., i]
            losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(-1))
        losses = torch.cat(losses, dim=1)
        losses= torch.mean(torch.sum(losses, dim=-1))
        
        return losses


    def to_prediction(self, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Convert network prediction into a point prediction.

        Args:
            y_pred: prediction output of network

        Returns:
            torch.Tensor: point prediction
        """
        if y_pred.shape[1] > 1:
            idx = self.quantiles.index(0.5)
            y_pred = y_pred[..., idx]
        return y_pred


    def to_quantiles(self, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Convert network prediction into a quantile prediction.

        Args:
            y_pred: prediction output of network

        Returns:
            torch.Tensor: prediction quantiles
        """
        return y_pred