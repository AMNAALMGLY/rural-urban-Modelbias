# have list of checkpoints
# have your model architucture to load on
# have your dataloader
# have output directory
# load your checkpoint and fit your model
# find the output  of the input
# save results to .npz file
import json
import os.path
from collections import defaultdict
from typing import Iterable, Mapping
from glob import glob

import numpy as np
import torch
from torch import nn
import tensorflow as tf
import batchers
from batchers.dataset import Batcher
from models.model_generator import get_model
from configs import args
from utils.utils import save_results, get_paths, load_from_checkpoint

OUTPUTS_ROOT_DIR = args.out_dir

DHS_MODELS = [
    # put paths to DHS models here (relative to OUTPUTS_ROOT_DIR)
    #'DHS_OOC_A_nl_random_weighted_b32_fc01_conv01_lr0001',
   # 'DHS_OOC_B_nl_random_weighted_b32_fc01_conv01_lr0001',
  # 'DHS_OOC_C_nl_random_weighted_b32_fc01_conv01_lr0001',
  # 'DHS_OOC_D_nl_random_weighted_b32_fc01_conv01_lr0001',
  #  'DHS_OOC_E_nl_random_weighted_b32_fc01_conv01_lr0001',
#'DHS_OOC_A_building_random_b32_fc01_conv01_lr0001',
#'DHS_OOC_B_building_random_b32_fc01_conv01_lr0001',
#'DHS_OOC_C_building_random_b32_fc01_conv01_lr0001',
#'DHS_OOC_D_building_random_b32_fc01_conv01_lr0001',
#'DHS_OOC_E_building_random_b32_fc01_conv01_lr0001',
#'DHS_OOC_A_nl_custom_b90_fc001_conv001_lr0001',
#'DHS_OOC_B_nl_custom_b90_fc001_conv001_lr0001',
#'DHS_OOC_C_nl_custom_b90_fc001_conv001_lr0001',
#'DHS_OOC_D_nl_custom_b90_fc01_conv01_lr0001',
#'DHS_OOC_E_nl_custom_b90_fc01_conv01_lr0001',

    #'DHS_OOC_A_ms_samescaled_b32_fc1_conv1_lr0001',
    #  'DHS_OOC_B_ms_samescaled_b32_fc1_conv1_lr0001',
    #  'DHS_OOC_C_ms_samescaled_b32_fc1_conv1_lr0001',
   # 'DHS_OOC_D_ms_samescaled_b32_fc1_conv1_lr0001',
  #  'DHS_OOC_E_ms_samescaled_b32_fc1_conv1_lr0001',

     'DHS_OOC_A_ms_samescaled_b64_fc01_conv01_lr0001',
     'DHS_OOC_B_ms_samescaled_b64_fc001_conv001_lr0001',
     'DHS_OOC_C_ms_samescaled_b64_fc001_conv001_lr001',
     'DHS_OOC_D_ms_samescaled_b64_fc001_conv001_lr01',
     'DHS_OOC_E_ms_samescaled_b64_fc01_conv01_lr001',
   # 'DHS_OOC_A_nl_random_b32_fc1.0_conv1.0_lr0001',
   # 'DHS_OOC_B_nl_random_b32_fc1.0_conv1.0_lr0001',
  #  'DHS_OOC_c_nl_random_b32_fc1.0_conv1.0_lr0001',
   # 'DHS_OOC_D_nl_random_b32_fc1.0_conv1.0_lr0001',
 #  '/DHS_OOC_E_nl_random_b32_fc1.0_conv1.0_lr0001',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
#'DHS_OOC_A_nl_random_b_b32_fc1.0_conv1.0_lr0001',
#'DHS_OOC_B_nl_random_b_b32_fc1.0_conv1.0_lr0001',
#'DHS_OOC_C_nl_random_b_b32_fc1.0_conv1.0_lr0001',
#'DHS_OOC_D_nl_random_b_b32_fc1.0_conv1.0_lr0001',
#'DHS_OOC_E_nl_random_b_b32_fc1.0_conv1.0_lr0001',
#'DHS_OOC_E_nl_random_b32_fc1.0_conv1.0_lr0001'

    # 'dhs_ooc/DHS_OOC_A_rgb_same_b64_fc0001_conv0001_lr001',
    # 'dhs_ooc/DHS_OOC_B_rgb_same_b64_fc001_conv001_lr0001',
    # 'dhs_ooc/DHS_OOC_C_rgb_same_b64_fc001_conv001_lr0001',
    # 'dhs_ooc/DHS_OOC_D_rgb_same_b64_fc1.0_conv1.0_lr01',
    # 'dhs_ooc/DHS_OOC_E_rgb_same_b64_fc001_conv001_lr0001',
]


def run_extraction_on_models(model_dir: str,
                             model_params: Mapping,
                             data_params,
                             batcher,
                             out_root_dir: str,
                             save_filename: str,
                             batch_keys: Iterable[str] = (),
                             ) -> None:
    '''Runs feature extraction on the given models, and saves the extracted
    features as a compressed numpy .npz file.

    Args
    - model_dirs: list of str, names of folders where models are saved, should
        be subfolders of out_root_dir
    - model_params: dict, parameters to pass to ModelClass constructor
    - batcher: Batcher, whose batch_op includes 'images' key
    - out_root_dir: str, path to main directory where all model checkpoints and
        TensorBoard logs are saved
    - save_filename: str, name of file to save
    - batch_keys: list of str to columns to be saved with the images

    '''

    print(f'Building model from {model_dir} checkpoint')

    model = get_model(**model_params)
    # redefine the model according to num_outputs
    fc = nn.Linear(model.fc.in_features, args.num_outputs)
    model.fc = fc

    checkpoint_pattern = os.path.join(out_root_dir, model_dir, 'best.ckpt')
    checkpoint_path = glob(checkpoint_pattern)
    print(checkpoint_path)
    model = load_from_checkpoint(path=checkpoint_path[-1], model=model)
    # freeze the last layer for feature extraction
    model.fc = nn.Sequential()
    model.to('cuda')
    model.eval()
    # model.freeze()
    for p in model.parameters():
        p.requires_grad=False
    with torch.no_grad():
        # initalizating
        np_dict = defaultdict()
        for i, record in enumerate(batcher):
            if data_params['include_buildings']:
                if data_params['ls_bands'] or data_params['nl_band']:
                    x = torch.tensor(record[0]['images'], )
                    b = torch.tensor(record[1]['buildings'], )
                    x = torch.cat((x, b), dim=-1)
                else:
                    x=torch.tensor(record[1]['buildings'], )


            else:
                x = torch.tensor(record['images'])

            x = x.type_as(model.conv1.weight)
            x = x.reshape(-1, x.shape[-1], x.shape[-3], x.shape[-2])  # [batch_size ,in_channels, H ,W]

            output = model(x)

            for key in batch_keys:
                if i == 0:
                    if data_params['include_buildings']:
                            np_dict[key] = record[0][key]
                    else:
                        np_dict[key] = record[key]
                else:
                    if data_params['include_buildings']:
                        np_dict[key] = np.append(np_dict[key], record[0][key], axis=0)
                    else:
                        np_dict[key] = np.append(np_dict[key], record[key], axis=0)
            features = output.to('cpu').numpy()
            if i == 0:
                np_dict['features'] = features
            else:
                np_dict['features'] = np.append(np_dict['features'], features, axis=0)

    save_dir = os.path.join(out_root_dir, model_dir)

    print(f'saving features to {save_dir} named {save_filename}')

    save_results(save_dir, np_dict, save_filename)


# best at Epoch 177 loss 0.3836648819908019.ckpt  B
# best at Epoch 117 loss 0.38881643032354696.ckpt C
# best at Epoch 160 loss 0.4024718254804611.ckpt D
# best at Epoch 149 loss 0.4684090583074477.ckpt E
def main(args):
    for model_dir in DHS_MODELS:
        # TODO check existing
        json_path = os.path.join(OUTPUTS_ROOT_DIR, model_dir, 'params.json')
        with open(json_path, 'r') as f:
            model_params = json.load(f)

        json_data_path = os.path.join(OUTPUTS_ROOT_DIR, model_dir, 'data_params.json')
        with open(json_data_path, 'r') as f:
            data_params = json.load(f)
        paths = get_paths(data_params['dataset'], 'all', 'A', args.data_path)
        if data_params['include_buildings']:
           paths_b = get_paths(data_params['dataset'], 'all', data_params['fold'], args.buildings_records)
        else:
            paths_b=None
        #TODO save path of building_reocrds or make it doesn't imply any thing in the dataset class
        batcher = Batcher(paths, None, data_params['ls_bands'], data_params['nl_band'],
                          data_params['label_name'],
                          data_params['nl_label'],data_params['include_buildings'],paths_b,normalize='DHS', augment=False, clipn=True,
                          batch_size=128, groupby=data_params['groupby'],
                          cache=False, shuffle=False)  # assumes no scalar features are present

        ## TODO fix in the future
        print('===Current Config ===')
        print(data_params)
        print(model_params)
        print(model_dir)
        run_extraction_on_models(model_dir,
                                 model_params,
                                 data_params,
                                 batcher=batcher,
                                 out_root_dir=OUTPUTS_ROOT_DIR,
                                 save_filename='features.npz',
                                 batch_keys=['labels', 'locs', 'years'],
                                 )


if __name__ == '__main__':
    main(args)
