import os
import json
import torch
import numpy as np

from pathlib import Path
from os.path import join, exists


def add_cmd_line_args(parser):
    parser.add('-c', '--config', required=False, is_config_file=True)

    parser.add_argument('--dr', type=str, default='pdr3')
    parser.add_argument('--data_dir', type=str, default='../../../data/astro')

    # experiment setup
    parser.add_argument('--para_nms', nargs='+')
    parser.add_argument('--tile_id', type=str, required=True)
    parser.add_argument('--footprint', type=str, required=True)
    parser.add_argument('--subtile_id', type=str, required=True)
    parser.add_argument('--trail_id', type=str, default='trail_dum')
    parser.add_argument('--experiment_id', type=str, default='exp_dum')

    # img args
    parser.add_argument('--sensors_full_name', nargs='+', required=True)
    parser.add_argument('--sensor_collection_name', type=str, required=True)

    parser.add_argument('--img_sz',type=int, default=64)
    parser.add_argument('--start_r', type=int, default=0)
    parser.add_argument('--start_c', type=int, default=0)

    # train and infer args
    parser.add_argument('--lr',type=float, default=1e-3)
    parser.add_argument('--dim',type=int, default=3)
    parser.add_argument('--num_epochs', type=int, default=4000)
    parser.add_argument('--verbose', action='store_true', default=False)

    # inpainting args
    parser.add_argument('--inpaint_cho', type=int, default=0, help='0-no inpaint, 1-spatial, 2-spectral')
    parser.add_argument('--mask_cho', type=int, default=0, help='0-diff across bands,1-same across bands,2-region')
    parser.add_argument('--mask_band_cho', type=int, default=0)
    parser.add_argument('--mask_sz', type=int, default=1)
    parser.add_argument('--mask_seed', type=int, default=0)
    parser.add_argument('--m_start_r', type=int, default=1)
    parser.add_argument('--m_start_c', type=int, default=1)
    parser.add_argument('--sample_ratio', type=float, default=1.0)
    parser.add_argument('--sample_ratio_cho', type=int, required=False)
    parser.add_argument('--train_bands', nargs='+', required=False)
    parser.add_argument('--inpaint_bands', nargs='+', required=False)

    args = parser.parse_args()
    config = vars(args)
    return config


def add_input_paths(config):
    dr = config['dr']
    img_id = config['img_id']
    data_dir = config['data_dir']

    dim = str(config['dim'])
    img_sz = str(config['img_sz'])
    start_r = str(config['start_r'])
    start_c = str(config['start_c'])
    num_bands = str(config['num_bands'])
    sensor_col_nm = config['sensor_collection_name']
    suffx = img_id +'_'+ img_sz +'_'+ start_r +'_'+ start_c + '.npy'

    input_dir = join(data_dir, dr +'_input')
    img_data_dir = join(input_dir, sensor_col_nm, 'img_data')
    mask_dir = join(input_dir, 'sampled_pixl_ids',
                    'cutout_' + suffx[:-4] +
                    '_mask_' + str(config['mask_band_cho']) +'_'
                    + str(config['inpaint_cho']) + '_'
                    + str(config['mask_cho']) + '_'
                    + str(config['mask_seed']))

    for path in [input_dir, mask_dir, img_data_dir]:
        Path(path).mkdir(parents=True, exist_ok=True)

    config['data_dir'] = data_dir
    config['mask_dir'] = mask_dir
    config['input_dir'] = input_dir
    config['dud_dir'] = join(input_dir, dr+'_dud')
    config['gt_img_fn'] = join(img_data_dir, 'gt_img_'+ suffx)

def add_train_infer_args(config):
    num_bands = config['num_bands']
    num_epochs = config['num_epochs']
    num_pixls = config['img_sz']**2

    # train and infer args
    if config['inpaint_cho'] == 1: # spatial inpainting
        config['npixls'] = int(num_pixls * config['sample_ratio'])
        config['num_train_pixls'] = int(num_pixls * config['sample_ratio'])
    else:
        config['npixls'] = num_pixls
        config['num_train_pixls'] = int(num_pixls)

    model_smpl_intvl = max(1,num_epochs//10)
    config['model_smpl_intvl'] = model_smpl_intvl

    config['cuda'] = torch.cuda.is_available()
    if config['cuda']:
        config['device'] = torch.device('cuda')
        config['float_tensor'] = torch.cuda.FloatTensor
        config['double_tensor'] = torch.cuda.DoubleTensor
    else:
        config['device'] =  torch.device('cpu')
        config['float_tensor'] = torch.FloatTensor
        config['double_tensor'] = torch.DoubleTensor

    # multi-band image recon args
    config['loss_options'] = [1,2,4]
    config['metric_names'] = ['mse','psnr','ssim']

    # inpaint specification
    tb = config['train_bands']
    ib = config['inpaint_bands']
    if ib is None:
        config['inpaint_bands'] = []
    elif config['inpaint_cho'] == 1:
        config['inpaint_bands'] = list(np.arange(config['num_bands']))
    else:
        config['inpaint_bands'] = [int(i) for i in ib]
    if tb is None or config['inpaint_cho'] == 1:
        config['train_bands'] = list(np.arange(config['num_bands']))
    else:
        config['train_bands'] = [int(i) for i in tb]

def add_output_paths(config):
    output_dir = join\
        (config['data_dir'], 'pnp' +'_output',
         config['sensor_collection_name'],
         'trail_'+ config['trail_id'], config['experiment_id'])

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if config['verbose']:
        print('--- output dir', output_dir)

    for path_nm, folder_nm, in zip\
        (['recon_dir','metric_dir'], ['recons','metrics']):
        path = join(output_dir, folder_nm)
        config[path_nm] = path
        Path(path).mkdir(parents=True, exist_ok=True)

    if config['mask_cho'] != 2:
        mask_str = '_' + str(float(100 * config['sample_ratio']))
    else:
        mask_str = '_' + str(config['m_start_r'])+'_' +\
            str(config['m_start_c'])+'_'+str(config['mask_sz'])

    config['sampled_pixl_id_fn'] = join\
        (config['mask_dir'], str(config['img_sz']) + mask_str + '.npy')

''' redefine some args'''
def process_config(config):
    config['num_bands'] = len(config['sensors_full_name'])

    # redefine experiment id if para_nms specified
    if config['para_nms'] is not None:
        config['experiment_id'] = [
            str(config[para_nm]) + '_' for para_nm in config['para_nms']
        ][:-1]

    # define img id
    config['img_id'] = config['footprint'] + config['tile_id'] + config['subtile_id']

    # redefine sample ratio if do inpainting and cho specified
    if config['inpaint_cho'] != 0 and config['sample_ratio_cho'] is not None:
        ratios = [0, 0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 1.0]
        config['sample_ratio'] = ratios[config['sample_ratio_cho']]

''' parse all command arguments and generate all needed ones '''
def parse_args(parser):
    print('=== Parsing')
    config = add_cmd_line_args(parser)
    process_config(config)
    add_input_paths(config)
    add_train_infer_args(config)
    add_output_paths(config)
    return config
