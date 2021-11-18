# have list of checkpoints
# have your model architucture to load on
# have your dataloader
# have output directory
# load your checkpoint and fit your model
# find the output  of the input
# save results to .npz file
import json
import os.path
from typing import Iterable, Mapping
from glob import glob

import torch
from torch import nn

import batchers
from batchers.dataset import Batcher
from models.model_generator import get_model
from configs import args
from utils.utils import save_results, get_paths, load_from_checkpoint

OUTPUTS_ROOT_DIR = args.out_dir
print(OUTPUTS_ROOT_DIR)
DHS_MODELS = [
    # put paths to DHS models here (relative to OUTPUTS_ROOT_DIR)
    # 'dhs_ooc/DHS_OOC_A_ms_samescaled_b64_fc01_conv01_lr0001',
    # 'dhs_ooc/DHS_OOC_B_ms_samescaled_b64_fc001_conv001_lr0001',
    # 'dhs_ooc/DHS_OOC_C_ms_samescaled_b64_fc001_conv001_lr001',
    # 'dhs_ooc/DHS_OOC_D_ms_samescaled_b64_fc001_conv001_lr01',
    # 'dhs_ooc/DHS_OOC_E_ms_samescaled_b64_fc01_conv01_lr001',
    # 'dhs_ooc/DHS_OOC_A_nl_random_b64_fc1.0_conv1.0_lr0001',
    # 'dhs_ooc/DHS_OOC_B_nl_random_b64_fc1.0_conv1.0_lr0001',
    # 'dhs_ooc/DHS_OOC_C_nl_random_b64_fc1.0_conv1.0_lr0001',
    # 'dhs_ooc/DHS_OOC_D_nl_random_b64_fc1.0_conv1.0_lr01',
    # 'dhs_ooc/DHS_OOC_E_nl_random_b64_fc1.0_conv1.0_lr0001',
    'dhs_ooc/DHS_OOC_A_rgb_same_b64_fc0001_conv0001_lr001',
    # 'dhs_ooc/DHS_OOC_B_rgb_same_b64_fc001_conv001_lr0001',
    # 'dhs_ooc/DHS_OOC_C_rgb_same_b64_fc001_conv001_lr0001',
    # 'dhs_ooc/DHS_OOC_D_rgb_same_b64_fc1.0_conv1.0_lr01',
    #  'dhs_ooc/DHS_OOC_E_rgb_same_b64_fc001_conv001_lr0001',
]


def run_extraction_on_models(model_dir: str,

                             model_params: Mapping,
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
    - ModelClass: class, an instance `model` of ModelClass
    - model_params: dict, parameters to pass to ModelClass constructor
    - batcher: Batcher, whose batch_op includes 'images' key
    - out_root_dir: str, path to main directory where all model checkpoints and
        TensorBoard logs are saved
    - save_filename: str, name of file to save
    - batch_keys: list of str

    '''

    print(f'Building model from {model_dir} checkpoint')

    model = get_model(**model_params)
    # redefine the model according to num_outputs
    fc = nn.Linear(model.fc.in_features, args.num_outputs)
    model.fc = fc

    checkpoint_pattern = os.path.join(out_root_dir, model_dir, '*.ckpt')
    checkpoint_path = glob(checkpoint_pattern)
    model = load_from_checkpoint(path=checkpoint_path[0], model=model)
    # freeze the last layer for feature extraction
    model.fc = nn.Sequential()
    model.to('cuda')
    # model.eval()
    # model.freeze()
    with torch.no_grad():
        for record in batcher:
            np_dict = {}
            # feature
            x=torch.tensor(record['images'], device='cuda')
            x = x.reshape(-1, x.shape[-1], x.shape[-3], x.shape[-2])  # [batch_size ,in_channels, H ,W]
            output = model(x)
            for key in batch_keys:
                np_dict[key] += [record[key]]
            np_dict['features'] += [output.to('cpu').numpy()]
    save_dir = os.path.join(out_root_dir, model_dir)
    print(f'saving features to {save_dir} under the name {save_filename}')
    save_results(model_dir, np_dict, save_filename)


def main(args):
    for model_dir in DHS_MODELS:
        # TODO check existing
        json_path = os.path.join(OUTPUTS_ROOT_DIR, model_dir, 'params.json')
        with open(json_path, 'r') as f:
            model_params = json.load(f)

        json_data_path = os.path.join(OUTPUTS_ROOT_DIR, model_dir, 'data_params.json')
        with open(json_data_path, 'r') as f:
            data_params = json.load(f)
        paths = get_paths(data_params['dataset'], 'all', data_params['fold'], args.data_path)

        batcher = Batcher(paths, None, data_params['ls_bands'], data_params['nl_band'], data_params['label_name'],
                          data_params['nl_label'], 'DHS', augment=False, clipn=True,
                          batch_size=data_params['batch_size'], groupby=data_params['groupby'],
                          cache=True)  # assumes no scalar features are present

        ## TODO fix in the future
        print('===Current Config ===')
        print(data_params)
        print(model_params)
        print(model_dir)
        run_extraction_on_models(model_dir,
                                 model_params,
                                 batcher=batcher,
                                 out_root_dir=OUTPUTS_ROOT_DIR,
                                 save_filename='features.npz',
                                 batch_keys=['labels', 'locs', 'years'],
                                 )


if __name__ == '__main__':
    main(args)
