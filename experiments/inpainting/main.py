
import os
import sys
import time
import glob
import math
import torch
import argparse
import numpy as np
import configargparse

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from torch import optim
from parser import parse_args
from skimage.transform import resize
from matplotlib.pyplot import imread, imsave
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio

sys.path.append('../')
from models import *
from admm_utils import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def func(A, b, x):
    return torch.norm(A(x.reshape(-1)) - b) ** 2 / 2

def run(img_fn, mask_fn, img_sz, noise_sigma, num_iter, rho, sigma_0, L, shrinkage_param,
        prior, sample_ratio, nchls, sample_intvl, recon_dir, metric_dir, dtype):

    img = np.load(img_fn)
    nchls = img.shape[0]
    x_true = torch.from_numpy(img).unsqueeze(0).type(dtype) # [nb,nchls,sz,sz]
    A, At, A_diag = A_inpainting(mask_fn, sample_ratio, img_sz**2, nchls, dtype)
    b = A((x_true[0]).permute(1,2,0))

    # model
    G = skip(nchls, nchls, num_channels_down=[16, 32, 64, 128, 128],
             num_channels_up=[16, 32, 64, 128, 128],  # [16, 32, 64, 128, 128],
             num_channels_skip=[0, 0, 0, 0, 0],
             filter_size_up=3, filter_size_down=3, filter_skip_size=1,
             upsample_mode='nearest',  # downsample_mode='avg',
             need1x1_up=False, need_sigmoid=False, need_bias=True,
             pad='reflection', act_fun='LeakyReLU').type(dtype)

    z = torch.zeros_like(x_true).type(dtype).normal_()
    x = G(z).clone().detach()
    scaled_lambda_ = torch.zeros_like(x, requires_grad=False).type(dtype)

    x.requires_grad, z.requires_grad = False, True

    # since we use exact minimization over x, we don't need the grad of x
    z.requires_grad = False
    opt_z = optim.Adam(G.parameters(), lr=L)

    sigma_0 = torch.tensor(sigma_0).type(dtype)
    prox_op = eval(prior)
    Gz = G(z)

    results = None
    mses, psnrs, ssims = [], [], []

    for t in range(num_iter):
        # for x
        with torch.no_grad():
            x = prox_op(Gz.detach() - scaled_lambda_, shrinkage_param / rho)

        # for z (GD)
        opt_z.zero_grad()
        Gz = G(z)
        loss_z = torch.norm(b- A((Gz[0]).permute(1,2,0))) ** 2 / 2 + (rho / 2) * torch.norm(x - G(z) + scaled_lambda_) ** 2
        loss_z.backward()
        opt_z.step()

        # for dual var(lambda)
        with torch.no_grad():
            Gz = G(z).detach()
            x_Gz = x - Gz
            scaled_lambda_.add_(sigma_0 * rho * x_Gz)

        if results is None:
            results = Gz.detach()
        else:
            results = results * 0.99 + Gz.detach() * 0.01

        if (t + 1) % sample_intvl == 0 or t == 0:
            gt = x_true.cpu().numpy()
            recon = results.cpu().numpy()

            id = (t + 1) // sample_intvl
            recon_path = os.path.join(recon_dir, str(id))
            losses = reconstruct(gt[0], recon[0], recon_path)

            #print(losses[0], losses[1], losses[2])
            print('[Iteration/Total] [%d/%d]' % (t + 1, num_iter))

            mses.append(losses[0]);psnrs.append(losses[1]);ssims.append(losses[2])

    np.save(os.path.join(metric_dir, 'mse.npy'), np.array(mses))
    np.save(os.path.join(metric_dir, 'psnr.npy'), np.array(psnrs))
    np.save(os.path.join(metric_dir, 'ssim.npy'), np.array(ssims))


if __name__ == '__main__':
    parser = configargparse.ArgumentParser()
    config = parse_args(parser)
    args = argparse.Namespace(**config)

    rho = 1
    sigma_0 = 1
    num_iters = 4000
    prior = 'nlm_prox'
    noise_sigma = 10/255
    shrinkage_param = 0.01

    start = time.time()
    run(args.gt_img_fn, args.sampled_pixl_id_fn, args.img_sz, noise_sigma, args.num_epochs,
        rho, sigma_0, args.lr, shrinkage_param, prior, args.sample_ratio, args.num_bands,
        args.model_smpl_intvl, args.recon_dir, args.metric_dir, args.float_tensor)
    print("Duration ", time.time() - start)
    print()
