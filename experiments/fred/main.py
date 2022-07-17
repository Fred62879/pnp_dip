
#import os
#import sys
import time
#import glob
#import math
#import torch
import argparse
#import numpy as np
import configargparse

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#os.chdir(os.path.dirname(os.path.abspath(__file__)))

#from torch import optim
from parser import parse_args
#from skimage.transform import resize
#from matplotlib.pyplot import imread, imsave
#from skimage.metrics import structural_similarity
#from skimage.metrics import peak_signal_noise_ratio

from batch_DIP import run as run_dip
from batch_DIP_TV import run as run_diptv
from batch_DIP_NLM import run as run_dipnlm
from batch_DIP_BM3D import run as run_dipbm3d

#sys.path.append('../')
#from models import *
#from admm_utils import *

if __name__ == '__main__':
    parser = configargparse.ArgumentParser()
    config = parse_args(parser)
    args = argparse.Namespace(**config)

    start = time.time()

    if args.submodel == 'DIP':
        run_dip(args, noise_sigma = 10/255)
    elif args.submodel == 'DIP-TV':
        run_diptv(args, noise_sigma=10/255, rho=1, sigma_0=1, shrinkage_param=0.01, prior='tv_prox')
    elif args.submodel == 'DIP-NLM':
        run_dipnlm(args, noise_sigma=10/255, rho=1, sigma_0=1, shrinkage_param=0.01, prior='nlm_prox') # nlm=0.01
    elif args.submodel == 'DIP-BM3D':
        run_dipbm3d(args, noise_sigma=10/255, rho=1, sigma_0=1, shrinkage_param=0.05, prior='bm3d_prox')
    else:
        raise Exception('Unsupported submodel')

    print("Duration ", time.time() - start)
    print()
