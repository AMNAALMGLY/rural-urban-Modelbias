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
from batchers.dataset_constants_buildings import DHS_COUNTRIES
from models.model_generator import get_model, Encoder
from configs import args
from utils.utils import save_results, get_paths

OUTPUTS_ROOT_DIR = args.out_dir

DHS_MODELS = [
    # put paths to DHS models here (relative to OUTPUTS_ROOT_DIR)
    #    'DHS_OOC_A_NL_NOattn_355_PE25_b32_fc01_conv01_lr0001_crop355',
    #   'DHS_OOC_B_NL_NOattn_355_PE25_b32_fc01_conv01_lr0001_crop355',
    #   'DHS_OOC_C_NL_NOattn_355_PE25_b32_fc01_conv01_lr0001_crop355',
    #  'DHS_OOC_D_NL_NOattn_355_PE25_b32_fc01_conv01_lr0001_crop355',
  #  'DHS_OOC_E_NL_NOattn_355_PE25_b32_fc01_conv01_lr0001_crop355',
    # 'DHS_OOC_A_building_larger_p120_b32_fce-05_conve-05_lr0001/',
    # 'DHS_OOC_B_BUILD_larger_PE120_b32_fce-05_conve-05_lr0001/',
    # 'DHS_OOC_C_BUILD_larger_PE120_b32_fce-05_conve-05_lr0001/',
    # 'DHS_OOC_D_BUILD_larger_PE120_b32_fce-05_conve-05_lr0001/',
    # 'DHS_OOC_E_BUILD_larger_PE120_b32_fc1_conv1_lr0001/',

    # 'DHS_OOC_A_NL_larger_PE120_b32_fc1.0_conv1.0_lr0001/',
    # 'DHS_OOC_B_NL_larger_PE120_b32_fc1.0_conv1.0_lr0001//',
    # 'DHS_OOC_C_NL_larger_PE120_b32_fc01_conv01_lr0001/',
    # 'DHS_OOC_D_NL_larger_PE120_b32_fc1_conv1_lr0001/',
    # 'DHS_OOC_E_NL_larger_PE120_b32_fc01_conv01_lr0001/',

    # 'DHS_OOC_A_encoder_b_nl_geo_b128_fc001_conv001_lre-05',
    # 'DHS_OOC_B_encoder_b_nl_geo_b128_fc001_conv001_lre-05',
    # 'DHS_OOC_C_encoder_b_nl_geo_b128_fc001_conv001_lre-05',
    # 'DHS_OOC_D_encoder_b_nl_geo_b128_fc001_conv001_lre-05',
    # 'DHS_OOC_E_encoder_b_nl_geo_b128_fc001_conv001_lre-05',

 #   'DHS_OOC_A_NL_Noise_validation_offset20_150_b64_fc01_conv01_lr0001_crop355',
  #  'DHS_OOC_B_NL_Noise_validation_offset20_150_b64_fc01_conv01_lr0001_crop355',
   # 'DHS_OOC_C_NL_Noise_validation_offset20_150_b64_fc01_conv01_lr0001_crop355',
   # 'DHS_OOC_D_NL_Noise_validation_offset20_150_b64_fc01_conv01_lr0001_crop355',
   # 'DHS_OOC_E_NL_Noise_validation_offset20_150_b64_fc01_conv01_lr0001_crop355',


   # 'DHS_OOC_A_NL_Noise_cent_nooffset_attn_uni_b64_fc01_conv01_lr0001_crop0',
    #'DHS_OOC_B_NL_Noise_cent_nooffset_attn_uni_b64_fc01_conv01_lr0001_crop0',
    #'DHS_OOC_C_NL_Noise_cent_nooffset_attn_uni_b64_fc01_conv01_lr0001_crop0',
    #'DHS_OOC_D_NL_Noise_cent_nooffset_attn_uni_b64_fc01_conv01_lr0001_crop0',
    #'DHS_OOC_E_NL_Noise_cent_nooffset_attn_uni_b64_fc01_conv01_lr0001_crop0',

    #'DHS_OOC_A_NL_Noise_cent_nooffset_attn_b64_fc01_conv01_lr0001_crop0',
    #'DHS_OOC_B_NL_Noise_cent_nooffset_attn_b64_fc01_conv01_lr0001_crop0',
    #'DHS_OOC_C_NL_Noise_cent_nooffset_attn_b64_fc01_conv01_lr0001_crop0',
    #'DHS_OOC_D_NL_Noise_cent_nooffset_attn_b64_fc01_conv01_lr0001_crop0',
    #'DHS_OOC_E_NL_Noise_cent_nooffset_attn_b64_fc01_conv01_lr0001_crop0',

    #'DHS_OOC_A_NL_Noise_offset0_attn_uni_b64_fc01_conv01_lr0001_crop355',
     #'DHS_OOC_B_NL_Noise_offset0_attn_uni_b64_fc01_conv01_lr0001_crop355',
     #'DHS_OOC_C_NL_Noise_offset0_attn_uni_b64_fc01_conv01_lr0001_crop355',
     #'DHS_OOC_D_NL_Noise_offset0_attn_uni_b64_fc01_conv01_lr0001_crop355',
     #'DHS_OOC_E_NL_Noise_offset0_attn_uni_b64_fc01_conv01_lr0001_crop355',

    #'DHS_OOC_A_NL_Noise_offset0_attn_b64_fc01_conv01_lr0001_crop355',
     #'DHS_OOC_B_NL_Noise_offset0_attn_b64_fc01_conv01_lr0001_crop355',
     #'DHS_OOC_C_NL_Noise_offset0_attn_b64_fc01_conv01_lr0001_crop355',
     #'DHS_OOC_D_NL_Noise_offset0_attn_b64_fc01_conv01_lr0001_crop355',
     #'DHS_OOC_E_NL_Noise_offset0_attn_b64_fc01_conv01_lr0001_crop355',

    #'DHS_OOC_A_NL_Noise_offset105_nei480__attn_b64_fc01_conv01_lr0001_crop355',
    #'DHS_OOC_B_NL_Noise_offset105_nei480__attn_b64_fc01_conv01_lr0001_crop355',
    #'DHS_OOC_C_NL_Noise_offset105_nei480__attn_b64_fc01_conv01_lr0001_crop355',
    #'DHS_OOC_D_NL_Noise_offset105_nei480__attn_b64_fc01_conv01_lr0001_crop355',
    #'DHS_OOC_E_NL_Noise_offset105_nei480__attn_b64_fc01_conv01_lr0001_crop355',

    #'DHS_OOC_A_NL_Noise_offset105_nei480__attn_uni_b64_fc01_conv01_lr0001_crop355',
    #'DHS_OOC_B_NL_Noise_offset105_nei480__attn_uni_b64_fc01_conv01_lr0001_crop355',
    #'DHS_OOC_C_NL_Noise_offset105_nei480__attn_uni_b64_fc01_conv01_lr0001_crop355',
    #'DHS_OOC_D_NL_Noise_offset105_nei480__attn_uni_b64_fc01_conv01_lr0001_crop355',
    #'DHS_OOC_E_NL_Noise_offset105_nei480__attn_uni_b64_fc01_conv01_lr0001_crop355',


  #  'DHS_OOC_A_NL_new_pooling_wealth_511_b32_fce-05_conve-05_lr0001_crop0',
  #  'DHS_OOC_B_NL_new_pooling_wealth_511_b32_fce-05_conve-05_lr0001_crop0',
  #  'DHS_OOC_C_NL_new_pooling_wealth_511_b32_fce-05_conve-05_lr0001_crop0',
   # 'DHS_OOC_D_NL_new_pooling_wealth_511_b32_fce-05_conve-05_lr0001_crop0',
  #  'DHS_OOC_E_NL_new_pooling_wealth_511_b32_fce-05_conve-05_lr0001_crop0',

#'DHS_OOC_A_NL_new_attn_wealth_511_b32_fce-05_conve-05_lr0001_crop0',
#'DHS_OOC_B_NL_new_attn_wealth_511_b32_fce-05_conve-05_lr0001_crop0',
#'DHS_OOC_C_NL_new_attn_wealth_511_b32_fce-05_conve-05_lr0001_crop0',
#'DHS_OOC_D_NL_new_attn_wealth_511_b32_fce-05_conve-05_lr0001_crop0',
#'DHS_OOC_E_NL_new_attn_wealth_511_b32_fce-05_conve-05_lr0001_crop0',


#'DHS_OOC_A_NL_new_pooling_wealth_511_P100_b32_fce-05_conve-05_lr0001_crop0',
#'DHS_OOC_B_NL_new_pooling_wealth_511_P100_b32_fce-05_conve-05_lr0001_crop0',
#'DHS_OOC_C_NL_new_pooling_wealth_511_P100_b32_fce-05_conve-05_lr0001_crop0',
#'DHS_OOC_D_NL_new_pooling_wealth_511_P100_b32_fce-05_conve-05_lr0001_crop0',
#'DHS_OOC_E_NL_new_pooling_wealth_511_P100_b32_fce-05_conve-05_lr0001_crop0',

#'DHS_OOC_A_NL_new_attn_wealth_511_P100_b32_fce-05_conve-05_lr0001_crop0',
#'DHS_OOC_B_NL_new_early_attn_wealth_511_P224_b32_fce-05_conve-05_lr0001_crop0',
#'DHS_OOC_C_NL_new_early_attn_wealth_511_P224_b32_fce-05_conve-05_lr0001_crop0',
#'DHS_OOC_D_NL_new_early_attn_wealth_511_P224_b32_fce-05_conve-05_lr0001_crop0',
#'DHS_OOC_E_NL_new_early_attn_wealth_511_P224_b32_fce-05_conve-05_lr0001_crop0',

'DHS_OOC_A_NL_no_attn_PE100_b32_fce-05_conve-05_lr0001_crop0',
'DHS_OOC_B_NL_no_attn_PE100_b32_fce-05_conve-05_lr0001_crop0',
'DHS_OOC_C_NL_new_no_attn_PE100_b32_fce-05_conve-05_lr0001_crop0',
'DHS_OOC_D_NL_no_attn_PE100_b32_fce-05_conve-05_lr0001_crop0',
'DHS_OOC_E_NL_no_attn_PE100_b32_fce-05_conve-05_lr0001_crop0',

'DHS_OOC_A_NL_new_no_attn_P224_b32_fce-05_conve-05_lr0001_crop0',
'DHS_OOC_B_NL_new_no_attn_P224_b32_fce-05_conve-05_lr0001_crop0',
'DHS_OOC_C_NL_new_no_attn_P224_b32_fce-05_conve-05_lr0001_crop0',
'DHS_OOC_D_NL_new_no_attn_P224_b32_fce-05_conve-05_lr0001_crop0',
'DHS_OOC_E_NL_new_no_attn_P224_b32_fce-05_conve-05_lr0001_crop0',


    'DHS_OOC_A_NL_new_no_attn_P511_b32_fce-05_conve-05_lr0001_crop0',
    'DHS_OOC_B_NL_new_no_attn_P511_b32_fce-05_conve-05_lr0001_crop0',
    'DHS_OOC_C_NL_new_no_attn_P511_b32_fce-05_conve-05_lr0001_crop0',
    'DHS_OOC_D_NL_new_no_attn_P511_b32_fce-05_conve-05_lr0001_crop0',
    'DHS_OOC_E_NL_new_no_attn_P511_b32_fce-05_conve-05_lr0001_crop0',

'DHS_OOC_A_NL_new_linear_resnet_P511_b32_fce-05_conve-05_lr0001_crop0',
'DHS_OOC_B_NL_new_linear_resnet_P511_b32_fce-05_conve-05_lr0001_crop0',
'DHS_OOC_C_NL_new_linear_resnet_P511_b32_fce-05_conve-05_lr0001_crop0',
'DHS_OOC_D_NL_new_linear_resnet_P511_b32_fce-05_conve-05_lr0001_crop0',
'DHS_OOC_E_NL_new_linear_resnet_P511_b32_fce-05_conve-05_lr0001_crop0',

'DHS_OOC_A_NL_new_linear_PE100_b32_fce-05_conve-05_lr0001_crop0',
'DHS_OOC_B_NL_new_linear_PE100_b32_fce-05_conve-05_lr0001_crop0',
'DHS_OOC_C_NL_new_linear_PE100_b32_fce-05_conve-05_lr0001_crop0',
'DHS_OOC_D_NL_new_linear_PE100_b32_fce-05_conve-05_lr0001_crop0',
'DHS_OOC_E_NL_new_linear_PE100_b32_fce-05_conve-05_lr0001_crop0',
#'DHS_OOC_A_NL_uniform_wealth_511_P100_b32_fce-05_conve-05_lr0001_crop',
# 'DHS_OOC_B_NL_new_uniform_attn_PE100_b32_fce-05_conve-05_lr0001_crop'  ,
 #   'DHS_OOC_C_NL_new_uniform_attn_PE100_b32_fce-05_conve-05_lr0001_crop',
  #  'DHS_OOC_D_NL_new_uniform_attn_PE100_b32_fce-05_conve-05_lr0001_crop',
#'DHS_OOC_E_NL_new_uniform_attn_PE100_b32_fce-05_conve-05_lr0001_crop',

#'DHS_OOC_A_NL_new_uniform_attn_PE224_b32_fce-05_conve-05_lr0001_crop0',
 #   'DHS_OOC_B_NL_new_uniform_attn_PE224_b32_fce-05_conve-05_lr0001_crop0',
  #  'DHS_OOC_C_NL_new_uniform_attn_PE224_b32_fce-05_conve-05_lr0001_crop0',
   # 'DHS_OOC_D_NL_new_uniform_attn_PE224_b32_fce-05_conve-05_lr0001_crop0',
    #'DHS_OOC_E_NL_new_uniform_attn_PE224_b32_fce-05_conve-05_lr0001_crop0',

    # DHS_OOC_A_encoder_b_nl_concat
    #  'DHS_OOC_A_ms_samescaled_b64_fc01_conv01_lr0001',
    # 'DHS_OOC_B_ms_samescaled_b64_fc001_conv001_lr0001',
    # 'DHS_OOC_C_ms_samescaled_b64_fc001_conv001_lr001',
    # 'DHS_OOC_D_ms_samescaled_b64_fc001_conv001_lr01',
    # 'DHS_OOC_E_ms_samescaled_b64_fc01_conv01_lr001',
    # 'DHS_OOC_A_nl_random_b32_fc1.0_conv1.0_lr0001',
    # 'DHS_OOC_B_nl_random_b32_fc1.0_conv1.0_lr0001',
    #  'DHS_OOC_c_nl_random_b32_fc1.0_conv1.0_lr0001',
    # 'DHS_OOC_D_nl_random_b32_fc1.0_conv1.0_lr0001',
    #  '/DHS_OOC_E_nl_random_b32_fc1.0_conv1.0_lr0001',
    # 'DHS_OOC_A_nl_random_b_b32_fc1.0_conv1.0_lr0001',
    # 'DHS_OOC_B_nl_random_b_b32_fc1.0_conv1.0_lr0001',
    # 'DHS_OOC_C_nl_random_b_b32_fc1.0_conv1.0_lr0001',
    # 'DHS_OOC_D_nl_random_b_b32_fc1.0_conv1.0_lr0001',
    # 'DHS_OOC_E_nl_random_b_b32_fc1.0_conv1.0_lr0001',
    # 'DHS_OOC_E_nl_random_b32_fc1.0_conv1.0_lr0001'

    # 'dhs_ooc/DHS_OOC_A_rgb_same_b64_fc0001_conv0001_lr001',
    # 'dhs_ooc/DHS_OOC_B_rgb_same_b64_fc001_conv001_lr0001',
    # 'dhs_ooc/DHS_OOC_C_rgb_same_b64_fc001_conv001_lr0001',
    # 'dhs_ooc/DHS_OOC_D_rgb_same_b64_fc1.0_conv1.0_lr01',
    # 'dhs_ooc/DHS_OOC_E_rgb_same_b64_fc001_conv001_lr0001',
]

#helper function
def load_from_checkpoint(path, model):
    print(f'initializing model from pretrained weights at {path}')
    ckpt = torch.load(path)

    model.load_state_dict(ckpt)
    # model.load_state_dict(torch.load(path))
    model.eval()
    return model
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
    encoder_params = defaultdict()
    # model params is a dictionary of dictionary that have 2 main keys(
    # model_dict(which is a dict of models itself) and self_attn)
    # for key, value in model_params['model_dict'].items():
    #        encoder_params[key] = get_model(**value)
    for key, value in model_params.items():

        if 'resnet' in key:
            encoder_params[key] = get_model(**value)
        else:
            encoder_params[key] = value
    # model = get_model(**model_params)
    # encoder = Encoder(encoder_params,model_params['self_attn'])
    encoder = Encoder(**encoder_params)
    # redefine the model according to num_outputs
    fc = nn.Linear(encoder.fc.in_features, args.num_outputs)
    print('fc shape', encoder.fc.in_features)
    # model.fc = fc
    encoder.fc = fc
    checkpoint_pattern = os.path.join(out_root_dir, model_dir, 'best.ckpt')

    checkpoint_path = glob(checkpoint_pattern)
    print(checkpoint_path)
    # model = load_from_checkpoint(path=checkpoint_path[-1], model=model)
    # freeze the last layer for feature extraction
    # model.fc = nn.Sequential()
    # model.to(args.gpus)
    if checkpoint_path:
        encoder = load_from_checkpoint(path=checkpoint_path[-1], model=encoder)
        # freeze the last layer for feature extraction
        encoder.fc = nn.Sequential()
        encoder.to(args.gpus)

        # model.freeze()

        with torch.no_grad():
            # initalizating
            np_dict = defaultdict()
            for i, record in enumerate(batcher):

                x = defaultdict()
                if data_params['include_buildings']:
                    if data_params['ls_bands'] and data_params['nl_band']:
                        # 2 bands split them into separate inputs
                        # assumes for now it is only merged nl_bands
                        x[data_params['ls_bands']] = torch.tensor(record[0]['images'][:, :, :-1], device=args.gpus)
                        x[data_params['nl_band']] = torch.tensor(record[0]['images'][:, :, -1], device=args.gpus)
                    elif data_params['ls_bands'] or data_params['nl_band']:
                        # only one type of band
                        x['images'] = torch.tensor(record[0]['images'], device=args.gpus)

                    x['buildings'] = torch.tensor(record[1]['buildings'], device=args.gpus)

                else:
                    if data_params['ls_bands'] and data_params['nl_band']:
                        # 2 bands split them inot seperate inputs
                        # assumes for now it is only merged nl_bands
                        x[data_params['ls_bands']] = torch.tensor(record['images'][:, :, :-1], device=args.gpus)
                        x[data_params['nl_band']] = torch.tensor(record['images'][:, :, -1], device=args.gpus)
                    elif data_params['ls_bands'] or data_params['nl_band']:
                        # only one type of band
                        x['images'] = torch.tensor(record['images'], device=args.gpus)

                # x = {key: value.type_as(encoder.fc.weight) for key, value in x.items()}
                for key, value in x.items():
                    x[key] = value.reshape(-1, value.shape[-1], value.shape[-3],
                                           value.shape[-2]) if value.dim() >= 3 else value

                output = encoder(x)

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
        print(np_dict['features'].shape)
        save_dir = os.path.join(out_root_dir, model_dir)

        print(f'saving features to {save_dir} named {save_filename}')
        save_results(save_dir, np_dict, save_filename, )


def main(args):
    for model_dir in DHS_MODELS:
        # TODO check existing
        '''
        json_path = os.path.join(OUTPUTS_ROOT_DIR, model_dir, 'params.json')
        with open(json_path, 'r') as f:
            model_params = json.load(f)
        '''
        json_data_path = os.path.join(OUTPUTS_ROOT_DIR, model_dir, 'data_params.json')
        with open(json_data_path, 'r') as f:
            data_params = json.load(f)

        json_path = os.path.join(OUTPUTS_ROOT_DIR, model_dir, 'encoder_params.json')
        with open(json_path, 'r') as f:
            model_params = json.load(f)

        paths = get_paths(data_params['dataset'], 'all', 'A', args.data_path)
        if data_params['include_buildings']:
            paths_b = get_paths(data_params['dataset'], 'all', data_params['fold'], args.buildings_records)
        else:
            paths_b = None
        # TODO save path of building_reocrds or make it doesn't imply any thing in the dataset class
        batcher = Batcher(paths, None, data_params['ls_bands'], data_params['nl_band'],
                          data_params['label_name'],
                          data_params['nl_label'], data_params['include_buildings'], paths_b, normalize='DHS',
                          augment=False, clipn=True,
                          batch_size=128, groupby=data_params['groupby'],
                          cache=False, shuffle=False, img_size=data_params['img_size'], crop=data_params['crop'],
                          rand_crop=data_params['rand_crop'],
                          offset=data_params['offset'])  # assumes no scalar features are present

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

"""
'DHS_OOC_B_NL_Noise_validation_offset0_nei355_b64_fc01_conv01_lr0001_crop355',
'DHS_OOC_C_NL_Noise_validation_offset0_nei355_b64_fc01_conv01_lr0001_crop355',
'DHS_OOC_D_NL_Noise_validation_offset0_nei355_b64_fc01_conv01_lr0001_crop355',
'DHS_OOC_E_NL_Noise_validation_offset0_nei355_b64_fc01_conv01_lr0001_crop355',

'DHS_OOC_A_NL_Noise_validation_offset60-nei480_b64_fc01_conv01_lr0001_crop355',
'DHS_OOC_B_NL_Noise_validation_offset60-nei480_b64_fc01_conv01_lr0001_crop355',
'DHS_OOC_C_NL_Noise_validation_offset60-nei480_b64_fc01_conv01_lr0001_crop355',
'DHS_OOC_D_NL_Noise_validation_offset60-nei480_b64_fc01_conv01_lr0001_crop355',
'DHS_OOC_E_NL_Noise_validation_offset60-nei480_b64_fc01_conv01_lr0001_crop355',

'DHS_OOC_A_NL_Noise_validation_offset40-nei480_b64_fc01_conv01_lr0001_crop355',
'DHS_OOC_B_NL_Noise_validation_offset40-nei480_b64_fc01_conv01_lr0001_crop355',
'DHS_OOC_C_NL_Noise_validation_offset40-nei480_b64_fc01_conv01_lr0001_crop355',
'DHS_OOC_D_NL_Noise_validation_offset40-nei480_b64_fc01_conv01_lr0001_crop355',
'DHS_OOC_E_NL_Noise_validation_offset40-nei480_b64_fc01_conv01_lr0001_crop355',

'DHS_OOC_A_NL_Noise_validation_offset105-nei480_b64_fc01_conv01_lr0001_crop355',
'DHS_OOC_B_NL_Noise_validation_offset105-nei480_b64_fc01_conv01_lr0001_crop355',
'DHS_OOC_C_NL_Noise_validation_offset105-nei480_b64_fc01_conv01_lr0001_crop355',
'DHS_OOC_D_NL_Noise_validation_offset105-nei480_b64_fc01_conv01_lr0001_crop355',
'DHS_OOC_E_NL_Noise_validation_offset105-nei480_b64_fc01_conv01_lr0001_crop355',

'DHS_OOC_A_NL_Noise_validation_offset0_b64_fc01_conv01_lr0001_crop355',
'DHS_OOC_B_NL_Noise_validation_offset0_b64_fc01_conv01_lr0001_crop355',
'DHS_OOC_C_NL_Noise_validation_offset0_b64_fc01_conv01_lr0001_crop355',
'DHS_OOC_D_NL_Noise_validation_offset0_b64_fc01_conv01_lr0001_crop355',
'DHS_OOC_E_NL_Noise_validation_offset0_b64_fc01_conv01_lr0001_crop355',

'DHS_OOC_A_NL_Noise_validation_offset100_0_b64_fc01_conv01_lr0001_crop355',
'DHS_OOC_B_NL_Noise_validation_offset100_0_b64_fc01_conv01_lr0001_crop355',
'DHS_OOC_C_NL_Noise_validation_offset100_0_b64_fc01_conv01_lr0001_crop355',
'DHS_OOC_D_NL_Noise_validation_offset100_0_b64_fc01_conv01_lr0001_crop355',
'DHS_OOC_E_NL_Noise_validation_offset100_0_b64_fc01_conv01_lr0001_crop355',

'DHS_OOC_A_NL_Noise_validation_offset130_b64_fc01_conv01_lr0001_crop355',
'DHS_OOC_B_NL_Noise_validation_offset130_b64_fc01_conv01_lr0001_crop355',
'DHS_OOC_C_NL_Noise_validation_offset130_b64_fc01_conv01_lr0001_crop355',
'DHS_OOC_D_NL_Noise_validation_offset130_b64_fc01_conv01_lr0001_crop355',
'DHS_OOC_E_NL_Noise_validation_offset130_b64_fc01_conv01_lr0001_crop355',

'DHS_OOC_A_NL_Noise_validation_offset50_100_b64_fc01_conv01_lr0001_crop355',
'DHS_OOC_B_NL_Noise_validation_offset50_100_b64_fc01_conv01_lr0001_crop355',
'DHS_OOC_C_NL_Noise_validation_offset50_100_b64_fc01_conv01_lr0001_crop355',
'DHS_OOC_D_NL_Noise_validation_offset50_100_b64_fc01_conv01_lr0001_crop355',
'DHS_OOC_E_NL_Noise_validation_offset50_100_b64_fc01_conv01_lr0001_crop355',

'DHS_OOC_A_NL_Noise_validation_offset20_b64_fc01_conv01_lr0001_crop355',
'DHS_OOC_B_NL_Noise_validation_offset20_b64_fc01_conv01_lr0001_crop355',
'DHS_OOC_C_NL_Noise_validation_offset20_b64_fc01_conv01_lr0001_crop355',
'DHS_OOC_D_NL_Noise_validation_offset20_b64_fc01_conv01_lr0001_crop355',
'DHS_OOC_E_NL_Noise_validation_offset20_b64_fc01_conv01_lr0001_crop355',
"""
