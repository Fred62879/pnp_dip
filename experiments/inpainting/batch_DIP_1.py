import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio
from matplotlib.pyplot import imread, imsave
from skimage.transform import resize
import time
import sys
import glob

sys.path.append('../')

from admm_utils import *
from torch import optim
from models import *
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def run(f_name, recon_dir, metric_dir, noise_sigma, num_iter, num_ratio, GD_lr):
    img = imread(f_name)
    if img.dtype == 'uint8':
        img = img.astype('float32') / 255  # scale to [0, 1]
    elif img.dtype == 'float32':
        img = img.astype('float32')
    else:
        raise TypeError()
    img = np.clip(resize(img, (128, 128)), 0, 1)
    imsave(recon_dir + 'true.png', img)
    if len(img.shape) == 2:
        img = img[:,:,np.newaxis]
        num_channels = 1
    else:
        num_channels = 3

    img = img.transpose((2, 0, 1))
    x_true = torch.from_numpy(img).unsqueeze(0).type(dtype)

    # A = torch.zeros(num_measurements, x_true.numel()).normal_().type(dtype) / math.sqrt(num_measurements)
    A, At, _ = A_inpainting(num_ratio, x_true.numel())
    b = A(x_true.reshape(-1,))
    b = torch.clamp(b + noise_sigma * torch.randn(b.shape).type(dtype), 0, 1)
    if num_channels == 3:
        imsave(recon_dir+'corrupted.png', At(b).reshape(1, num_channels, 128, 128)[0].permute((1,2,0)).cpu().numpy())
    else:
        imsave(recon_dir + 'corrupted.png', At(b).reshape(1, num_channels, 128, 128)[0, 0].cpu().numpy(), cmap='gray')

    def fn(x): return torch.norm(A(x.reshape(-1)) - b) ** 2 / 2

    # G = skip(3, 3,
    #            num_channels_down = [16, 32, 64, 128, 128, 128],
    #            num_channels_up =   [16, 32, 64, 128, 128, 128],
    #            num_channels_skip =    [4, 4, 4, 4, 4, 4],
    #            filter_size_up = [7, 7, 5, 5, 3, 3],filter_size_down = [7, 7, 5, 5, 3, 3],  filter_skip_size=1,
    #            upsample_mode='bilinear', # downsample_mode='avg',
    #            need1x1_up=False,
    #            need_sigmoid=True, need_bias=True, pad='reflection', act_fun='LeakyReLU').type(dtype)
    G = skip(3, 3,
             num_channels_down=[16, 32, 64, 128, 128],
             num_channels_up=[16, 32, 64, 128, 128],#[16, 32, 64, 128, 128],
             num_channels_skip=[0, 0, 0, 0, 0],
             filter_size_up=3, filter_size_down=3, filter_skip_size=1,
             upsample_mode='nearest',  # downsample_mode='avg',
             need1x1_up=False,
             need_sigmoid=True, need_bias=True, pad='reflection', act_fun='LeakyReLU').type(dtype)
    z = torch.zeros_like(x_true).type(dtype).normal_()

    z.requires_grad = False
    opt = optim.Adam(G.parameters(), lr=GD_lr)



    results = None
    mses, psnrs, ssims = [], [], []
    for t in range(num_iter):
        x = G(z)
        fidelity_loss = fn(x)

        # prior_loss = (torch.sum(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) + torch.sum(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])))
        total_loss = fidelity_loss #+ 0.01 * prior_loss
        opt.zero_grad()
        total_loss.backward()
        opt.step()


        if results is None:
            results = x.detach().cpu().numpy()
        else:
            results = results * 0.99 + x.detach().cpu().numpy() * 0.01

        psnr_gt = peak_signal_noise_ratio(x_true.cpu().numpy(), results)
        mse_gt = np.mean((x_true.cpu().numpy() - results) ** 2)

        if (t + 1) % 100 == 0:
            if num_channels == 3:
                imsave(recon_dir + 'iter%d_PSNR_%.2f.png'%(t, psnr_gt), results[0].transpose((1,2,0)))
            else:
                imsave(recon_dir + 'iter%d_PSNR_%.2f.png'%(t, psnr_gt), results[0, 0], cmap='gray')

            print('[Iteration/Total/MSE/PSNR] [%d/%d/%.3f/%.3f] ' % (t + 1, num_iter, mse_gt, psnr_gt))

    np.save(os.path.join(metric_dir, 'mse.npy'), np.array(mses))
    np.save(os.path.join(metric_dir, 'psnr.npy'), np.array(psnrs))



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

    run(img_fn, recon_dir, metric_dir, noise_sigma = 10 / 255, num_iter = 5000, num_ratio = 0.5, GD_lr=0.001)
