import os.path
from glob import glob
import numpy as np
import pandas as pd
import torch.utils.data
from torch.utils.data import Dataset
from batchers.dataset import Batcher
from src.configs import args
from utils.utils import get_paths

data_dir='./np_data'


class Data(Dataset):
    def __init__(self,data_dir):
            self.data_dir=data_dir
            data = pd.DataFrame(columns=['images', 'locs', 'years', 'labels'])
            length=self.get_length()
            for i in range(length):
                print(i)
                with np.load(os.path.join(data_dir, f'{i:05d}.npz')) as d:
                    data.loc[i, 'images'] = d['images']
                    data.loc[i, 'locs'] = d['locs']
                    data.loc[i, 'years'] = d['years']
                    data.loc[i, 'labels'] = d['labels']
            self.data = data
            self.x = data['images']
            self.y = data['labels']

    def __len__(self):
            return self.get_length()

    def __getitem__(self, item):
            return self.get_length()

    def get_length(self):
        file_path = os.path.join(self.data_dir, '*.npz')
        all_files = glob(file_path)
        length = len(all_files)
        return length

paths_train = get_paths(args.dataset, ['train'], args.fold, args.data_path)
batcher_train = Batcher(paths_train, args.scaler_features_keys, args.ls_bands, args.nl_band, args.label_name,
                            args.nl_label, 'DHS', args.augment, args.batch_size, groupby=args.group,
                            cache=True,save_dir=data_dir)
batcher_train.tfDatase_to_np() #TODO this doesn't need to be called on batcher object

dataloader=torch.utils.data.DataLoader(Data(data_dir=data_dir),batch_size=16,num_workers=3,pin_memory=True, prefetch_factor=2,shuffle=True)
for x, y in dataloader:
    print(x,y)



