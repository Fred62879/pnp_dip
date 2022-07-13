import os
import json
import torch
import numpy as np

from pathlib import Path
from os.path import join, exists


def add_cmd_line_args(parser):
    parser.add('-c', '--config', required=False, is_config_file=True)

    parser.add_argument('--model', type=str, default='pnp_dip')
    parser.add_argument('--submodel', type=str, required=True)
    parser.add_argument('--astro', action='store_true',default=False)

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
    parser.add_argument('--normal_img_id', type=str, default='1')

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
    parser.add_argument('--inpaint_cho', type=str, default='no_inpaint')
    parser.add_argument('--mask_config', type=str, default='rand_diff')
    parser.add_argument('--mask_bandset_cho', type=str,default='None')
    parser.add_argument('--mask_bands', type=str,default='None')
    parser.add_argument('--mask_seed', type=int, default=0)
    parser.add_argument('--sample_ratio', type=float, default=1.0)
    parser.add_argument('--sample_ratio_cho', type=int)
    parser.add_argument('--current_bands', nargs='+', type=int)

    args = parser.parse_args()
    config = vars(args)
    return config


def add_input_paths(config):
    data_dir = config['data_dir']

    if config['astro']:
        dr = config['dr']
        img_id = config['img_id']

        dim = str(config['dim'])
        img_sz = str(config['img_sz'])
        start_r = str(config['start_r'])
        start_c = str(config['start_c'])
        num_bands = str(config['num_bands'])
        sensor_col_nm = config['sensor_collection_name']
        suffx = img_id +'_'+ img_sz +'_'+ start_r +'_'+ start_c + '.npy'

        input_dir = join(data_dir, dr +'_input')
        img_data_dir = join(input_dir, sensor_col_nm, 'img_data')

        dirs = [input_dir, img_data_dir]

        if config['inpaint_cho'] == 'spectral_inpaint':
            mask_dir = join(input_dir, 'sampled_pixl_ids',
                            'cutout_' + suffx[:-4] + '_mask_'
                            + str(config['mask_config']) + '_'
                            + str(config['mask_bandset_cho']) +'_'
                            + str(config['mask_bands']) + '_'
                            + str(config['mask_seed']))
            config['mask_dir'] = mask_dir
            dirs.append(mask_dir)

        for path in dirs:
            Path(path).mkdir(parents=True, exist_ok=True)

        config['data_dir'] = data_dir
        config['input_dir'] = input_dir
        config['dud_dir'] = join(input_dir, dr+'_dud')
        config['gt_img_fn'] = join(img_data_dir, 'gt_img_'+ suffx)

    else:
        input_dir = join(data_dir, 'normal_input')
        Path(input_dir).mkdir(parents=True, exist_ok=True)
        config['gt_img_fn'] = join(input_dir, 'gt_img',
                                   config['normal_img_id'] + '.jpg')

def add_train_infer_args(config):
    num_epochs = config['num_epochs']

    model_smpl_intvl = max(1, num_epochs//10)
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

    if not config['astro']:
        return

    num_bands = config['num_bands']
    num_pixls = config['img_sz']**2

    # train and infer args
    if config['inpaint_cho'] == 'spatial_inpaint':
        config['npixls'] = int(num_pixls * config['sample_ratio'])
        config['num_train_pixls'] = int(num_pixls * config['sample_ratio'])
    else:
        config['npixls'] = num_pixls
        config['num_train_pixls'] = int(num_pixls)

    # multi-band image recon args
    config['loss_options'] = [1,2,4]
    config['metric_names'] = ['mse','psnr','ssim']

    # inpaint specification
    if ['inpaint_cho'] != 'no_inpaint':
        if config['mask_bandset_cho'] == '10':
            config['mask_bandset'] = [0,1,2,3,4,5,6,7,8,9]
        elif config['mask_bandset_cho'] == '5grizy':
            config['mask_bandset'] = [0,1,2,3,4]
        else:
            raise Exception('Unsupported mask bandsets')

def add_output_paths(config):
    if config['astro']:
        output_dir = join\
            (config['data_dir'], config['dr'] +'_output',
             config['model'] + '_output',
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

        if config['mask_config'] != 'region':
            mask_str = '_' + str(float(100 * config['sample_ratio']))
        else:
            mask_str = '_' + str(config['m_start_r'])+'_' +\
                str(config['m_start_c'])+'_'+str(config['mask_sz'])

        config['sampled_pixl_id_fn'] = join\
            (config['mask_dir'], str(config['img_sz']) + mask_str + '.npy')

    else: # normal image training
        output_dir = join\
            (config['data_dir'], 'normal_output',
             config['model'] + '_output',
             'trail_'+ config['trail_id'], config['experiment_id'])

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        if config['verbose']:
            print('--- output dir', output_dir)

        for path_nm, folder_nm, in zip\
            (['recon_dir','metric_dir'], ['recons','metrics']):
            path = join(output_dir, folder_nm)
            config[path_nm] = path
            Path(path).mkdir(parents=True, exist_ok=True)


''' redefine some args'''
def process_config(config):
    config['num_bands'] = len(config['sensors_full_name'])

    # define img id
    config['img_id'] = config['footprint'] + config['tile_id'] + config['subtile_id']

    # redefine sample ratio if do inpainting and cho specified
    if config['inpaint_cho'] != 'no_inpaint' and config['sample_ratio_cho'] is not None:
        ratios = [0, 0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 1.0]
        config['sample_ratio'] = ratios[config['sample_ratio_cho']]

    # redefine experiment id if para_nms specified
    if config['para_nms'] is not None:
        eid = ''
        for para_nm in config['para_nms']:
            eid += str(config[para_nm]) + '_'
        config['experiment_id'] = eid[:-1]


''' parse all command arguments and generate all needed ones '''
def parse_args(parser):
    print('=== Parsing')
    config = add_cmd_line_args(parser)
    if config['astro']:
        process_config(config)
    add_input_paths(config)
    add_train_infer_args(config)
    add_output_paths(config)
    return config
