# Setup Experiment
import pytorch_lightning as pl
#from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from batchers.dataset import Batcher
from models.resnet import resnet50
from src.trainer import ResTrain
from utils.utils import get_paths



#logger = TensorBoardLogger("resnet_logs", name="resnet50")

# model
model = resnet50(in_channels=7, pretrained=True)
# fc=nn.Linear(model.fc.in_features,1)

# metric
#metric = torchmetrics.PearsonCorrcoef()

# dataloader
paths_train = get_paths('DHS_OOC', ['train'], 'A', '/content/drive/MyDrive/dhs_tfrecords')
paths_valid = get_paths('DHS_OOC', ['val'], 'A', '/content/drive/MyDrive/dhs_tfrecords')

batcher_train = Batcher(paths_train, None, 'ms', None, 'wealthpooled', None, 'DHS', False, 64, )
batcher_valid = Batcher(paths_valid, None, 'ms', None, 'wealthpooled', None, 'DHS', False, 64, )
# lightning model , trainer

litmodel = ResTrain(model, 0.01, 0.001, 'regression', 1, metric='r2')
checkpoint_callback = ModelCheckpoint(monitor="train_loss", mode='min',
                                      filename="ResnetFineTune-{epoch:02d}-{train-loss:.2f}",
                                      dirpath='/content/drive/MyDrive/checkpoints_resnet',
                                      verbose=True,
                                      save_last=True)
ckpt_path = None
trainer = pl.Trainer(max_epochs=400, gpus=1, auto_select_gpus=True,
                     logger=logger,
                     callbacks=[checkpoint_callback],
                     resume_from_checkpoint=ckpt_path,
                     accumulate_grad_batches=16)  # understand what it does exactly
trainer.fit(litmodel, batcher_train, batcher_valid)
