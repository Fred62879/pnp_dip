import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import sys
import time
import torch
import numpy as np

sys.path.append('../')

from utils import *
from models import *
from torch import optim

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def run(args, noise_sigma):

    def fn(x):
        return torch.norm(A(x.reshape(-1)) - b) ** 2 / 2

    G = skip(3, 3,
             num_channels_down=[16, 32, 64, 128, 128],
             num_channels_up=[16, 32, 64, 128, 128],#[16, 32, 64, 128, 128],
             num_channels_skip=[0, 0, 0, 0, 0],
             filter_size_up=3, filter_size_down=3, filter_skip_size=1,
             upsample_mode='nearest',  # downsample_mode='avg',
             need1x1_up=False,
             need_sigmoid=True, need_bias=True, pad='reflection', act_fun='LeakyReLU')\
             .type(args.float_tensor)

    opt = optim.Adam(G.parameters(), args.lr)

    num_channels, x_true, A, _, _, b, _, z, _ = load_data(G, noise_sigma, args)
    z.requires_grad = False

    results = None
    mses, psnrs, ssims = [], [], []
    for epoch in range(args.num_epochs):
        x = G(z)
        fidelity_loss = fn(x)

        # prior_loss = (torch.sum(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) + torch.sum(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])))
        total_loss = fidelity_loss #+ 0.01 * prior_loss
        opt.zero_grad()
        total_loss.backward()
        opt.step()

        if results is None:
            results = x.detach() #.cpu().numpy()
        else:
            #results = results * 0.99 + x.detach().cpu().numpy() * 0.01
            results = results * 0.99 + x.detach() * 0.01

        if (epoch + 1) % args.model_smpl_intvl == 0 or epoch == 0:
            gt = x_true.cpu().numpy()
            recon = results.cpu().numpy()

            metrics = reconstruct(epoch, gt, recon, num_channels, args)
            if args.astro:
                mses.append(metrics[0])
                psnrs.append(metrics[1])
                ssims.append(metrics[2])

    if args.astro:
        np.save(os.path.join(args.metric_dir, 'mse.npy'), np.array(mses))
        np.save(os.path.join(args.metric_dir, 'psnr.npy'), np.array(psnrs))
        np.save(os.path.join(args.metric_dir, 'ssim.npy'), np.array(ssims))
