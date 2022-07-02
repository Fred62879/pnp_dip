import os
import sys
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('../')

import time
import torch
import numpy as np

from models import *
from torch import optim
from admm_utils import *
from skimage.transform import resize
from matplotlib.pyplot import imread, imsave
from skimage.metrics import peak_signal_noise_ratio


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def run(f_name, recon_dir, metrics_dir, noise_sigma, num_iter, rho, sigma_0, L, shrinkage_param, prior, num_ratio):

    if torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    img = imread(f_name)
    if img.dtype == 'uint8':
        img = img.astype('float32') / 255  # scale to [0, 1]
    elif img.dtype == 'float32':
        img = img.astype('float32')
    else:
        raise TypeError()

    img = np.clip(resize(img, (128, 128)), 0, 1) ## CHANGED
    imsave(recon_dir + 'true.png', img)
    img = img.transpose((2, 0, 1))
    x_true = torch.from_numpy(img).unsqueeze(0).type(dtype)

    A, At, A_diag = A_inpainting(num_ratio, x_true.numel())

    b = A(x_true.reshape(-1, ))
    b = torch.clamp(b + noise_sigma * torch.randn(b.shape).type(dtype), 0, 1)
    imsave(recon_dir + 'corrupted.png',
           At(b).reshape(1, 3, 128, 128)[0].permute((1, 2, 0)).cpu().numpy()) ## CHANGED

    def fn(x):
        return torch.norm(A(x.reshape(-1)) - b) ** 2 / 2

    G = skip(3, 3,
             num_channels_down=[16, 32, 64, 128, 128],
             num_channels_up=[16, 32, 64, 128, 128],  # [16, 32, 64, 128, 128],
             num_channels_skip=[0, 0, 0, 0, 0],
             filter_size_up=3, filter_size_down=3, filter_skip_size=1,
             upsample_mode='nearest',  # downsample_mode='avg',
             need1x1_up=False,
             need_sigmoid=True, need_bias=True, pad='reflection', act_fun='LeakyReLU').type(dtype)
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
        loss_z = torch.norm(b- A(Gz.view(-1))) ** 2 / 2 + (rho / 2) * torch.norm(x - G(z) + scaled_lambda_) ** 2
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

        psnr_gt = peak_signal_noise_ratio(x_true.cpu().numpy(), results.cpu().numpy())
        mse_gt = np.mean((x_true.cpu().numpy() - results.cpu().numpy()) ** 2)
        fidelity_loss = fn(results).detach()

        if (t + 1) % 100 == 0:
            mses.append(mse_gt); psnrs.append(psnr_gt)
            imsave(recon_dir + 'iter%d_PSNR_%.2f.png' % (t, psnr_gt), results[0].cpu().numpy().transpose((1, 2, 0)))

        if (t + 1) % 50 == 0:
            print('[Iteration/Total/MSE/PSNR] [%d/%d/%.3f/%.3f] ' % (t + 1, num_iter, mse_gt, psnr_gt))

    np.save(os.path.join(metrics_dir, 'mse.npy'), np.array(mses))
    np.save(os.path.join(metrics_dir, 'psnr.npy'), np.array(psnrs))


if __name__ == '__main__':
    torch.manual_seed(500)
    if torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    img_sz = 128
    rho = 1
    nfls = 10
    L = 0.001
    sigma_0 = 1
    num_iters = 5000
    mask_ratio = 100.0
    spectral = False
    prior = 'nlm_prox'
    noise_sigma = 10/255
    shrinkage_param = 0.01
    sample_intvl = num_iters // 4

    loss = 'l1_'
    dim = '2d_'+ str(nfls)
    data_dir = '../../../../data'
    #sz_str = str(img_sz) + ('_spectra' if spectral else '')

    mask_dir = os.path.join(data_dir, 'pdr3_output/sampled_id')
    output_dir = os.path.join(data_dir, 'pdr3_output/'+dim+'/PNP/'+str(img_sz)+'/l1_'+str(mask_ratio))
    recon_dir = os.path.join(output_dir, 'recons')
    metric_dir = os.path.join(output_dir, 'metrics')

    mask_fn = os.path.join(mask_dir, str(img_sz)+'_'+str(mask_ratio)+'.npy')
    img_fn = os.path.join(data_dir, 'pdr3_output', dim, 'orig_imgs/celeba.jpg')
    #img_fn = os.path.join(data_dir, 'pdr3_output', dim, 'orig_imgs/0_'+str(img_sz)+'.npy')

    run(img_fn, recon_dir, metric_dir, noise_sigma=10/255, num_iter=num_iters, rho=1, sigma_0=1,
        L=0.001, shrinkage_param=0.01, prior='nlm_prox', num_ratio=mask_ratio/100)
