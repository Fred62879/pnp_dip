import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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


def run(args, noise_sigma, rho, sigma_0, shrinkage_param, prior):

    def fn(x):
        return torch.norm(A(x.reshape(-1)) - b) ** 2 / 2

    G = skip(args.num_bands, args.num_bands,
             num_channels_down=[16, 32, 64, 128, 128],
             num_channels_up=[16, 32, 64, 128, 128],  # [16, 32, 64, 128, 128],
             num_channels_skip=[0, 0, 0, 0, 0],
             filter_size_up=3, filter_size_down=3, filter_skip_size=1,
             upsample_mode='nearest',  # downsample_mode='avg',
             need1x1_up=False,
             need_sigmoid=True, need_bias=True, pad='reflection', act_fun='LeakyReLU')\
             .type(args.float_tensor)

    opt_z = optim.Adam(G.parameters(), lr=args.lr)

    num_channels, x_true, A, _, _, b, x, z, scaled_lambda_ = load_data(G, noise_sigma, args)
    # since we use exact minimization over x, we don't need the grad of x
    gt = x_true.cpu().numpy()
    z.requires_grad = False

    sigma_0 = torch.tensor(sigma_0).type(args.float_tensor)
    prox_op = eval(prior)
    Gz = G(z)

    results = None
    mses, psnrs, ssims = [], [], []
    for epoch in range(args.num_epochs):
        # for x
        with torch.no_grad(): # [bsz,nchls,sz,sz]
            x = prox_op(Gz.detach() - scaled_lambda_, shrinkage_param / rho)

        # for z (GD)
        opt_z.zero_grad()
        Gz = G(z) # [bsz,nchls,sz,sz]
        if args.astro: sampled_pixls = A(Gz[0].permute(1,2,0)) # [sz,sz,nchls]
        else:          sampled_pixls = A(Gz.view(-1))
        loss_z = torch.norm(b- sampled_pixls) ** 2 / 2 \
            + (rho / 2) * torch.norm(x - G(z) + scaled_lambda_) ** 2
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

        if (epoch + 1) % args.model_smpl_intvl == 0 or epoch == 0:
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
