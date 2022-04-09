
import os
import sys
import time
import glob
import math
import torch
import numpy as np

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from torch import optim
from skimage.transform import resize
from matplotlib.pyplot import imread, imsave
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio

sys.path.append('../')
from models import *
from admm_utils import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def func(x):
    return torch.norm(A(x.reshape(-1)) - b) ** 2 / 2

def load_data(img_fn, mask_fn, output_dir, ratio, noise_sigma):
    img = imread(img_fn)
    if img.dtype == 'uint8':
        img = img.astype('float32') / 255  # scale to [0, 1]
    elif img.dtype == 'float32':
        img = img.astype('float32')
    else:
        raise TypeError()

    img = np.clip(resize(img, (128, 128)), 0, 1) ## CHANGED
    imsave(output_dir + '/true.png', img)
    img = img.transpose((2, 0, 1))
    print(img.shape)
    x_true = torch.from_numpy(img).unsqueeze(0).type(dtype)

    A, At, A_diag = A_inpainting(mask_fn, ratio, x_true.numel(), dtype)

    b = A(x_true.reshape(-1, ))
    #b = torch.clamp(b + noise_sigma * torch.randn(b.shape).type(dtype), 0, 1)
    masked_img = At(b).reshape(1, 3, 128, 128)[0].permute((1, 2, 0)).cpu().numpy()
    imsave(output_dir + '/corrupted.png', masked_img)
    return b, x_true, A

def init_model(x_true, sigma_0, L, prior):
    G = skip(3, 3,
             num_channels_down=[16, 32, 64, 128, 128],
             num_channels_up=[16, 32, 64, 128, 128],  # [16, 32, 64, 128, 128],
             num_channels_skip=[0, 0, 0, 0, 0],
             filter_size_up=3, filter_size_down=3, filter_skip_size=1,
             upsample_mode='nearest',  # downsample_mode='avg',
             need1x1_up=False,
             need_sigmoid=True, need_bias=True, pad='reflection', act_fun='LeakyReLU')\
             .type(dtype)

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

    return z, G, Gz, opt_z, scaled_lambda_, prox_op


def train(b, z, A, sample_intvl, num_iter, opt_z, G, Gz, scaled_lambda_, shrinkage_param, rho, prox_op):
    results = None
    mses, psnrs, ssims = [], [], []

    for t in range(num_iter):
        # for x
        with torch.no_grad():
            x = prox_op(Gz.detach() - scaled_lambda_, shrinkage_param / rho)

        # for z (GD)
        opt_z.zero_grad()
        Gz = G(z)
        loss_z = torch.norm(b- A(Gz.view(-1))) ** 2 / 2 + \
            (rho / 2) * torch.norm(x - G(z) + scaled_lambda_) ** 2
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

        if t == 0 or (t + 1) % sample_intvl == 0:
            fidelity_loss = func(results).detach()
            gt = x_true.cpu().numpy()[0]
            recon = results.cpu().numpy()[0]

            id = (t + 1) // sample_intvl
            fn = os.path.join(output_dir, 'recons', str(id) + '.png')
            imsave(fn, recon.transpose((1, 2, 0)))

            print('[%d / %d] '% (t+1, num_iter))

            if t == sample_intvl - 1:
                metric_dir = os.path.join(output_dir, 'metrics')
                recon_dir = os.path.join(output_dir, 'recons')
                reconstruct(gt, recon, recon_dir, metric_dir)

            #mses.append(mse);psnrs.append(psnr);ssims.append(ssim)

    #np.save(os.path.join(outputdir, metrics, 'mse.npy'), np.array(mses))
    #np.save(os.path.join(outputdir, metrics, 'psnr.npy'), np.array(psnrs))
    #np.save(os.path.join(outputdir, metrics, 'ssim.npy'), np.array(ssims))


if __name__ == '__main__':

    global dtype
    if torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    '''
    nfls = args.nfls      # num bands
    ratio = float(args.ratio)    # ratio %
    img_sz = args.imgsz
    n_iters = args.niters
    spectral = args.spectral
    '''
    img_sz = 128
    rho = 1
    nfls = 5
    L = 0.001
    sigma_0 = 1
    num_iters = 500
    mask_ratio = 10.0
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
    #output_dir = os.path.join(data_dir, 'pdr3_output/'+dim+'/PNP',
    #                          sz_str, loss + str(mask_ratio))
    output_dir = os.path.join(data_dir, 'pdr3_output/'+dim+'/PNP/trail/0')

    img_fn = os.path.join(data_dir, 'pdr3_output', dim, 'orig_imgs/celeba.jpg')
    mask_fn = os.path.join(mask_dir, str(img_sz)+'_'+str(mask_ratio)+'.npy')
    #img_fn = os.path.join(data_dir, 'pdr3_output', dim, 'orig_imgs/0_'+str(img_sz)+'.npy')

    b, x_true, A = load_data(img_fn, mask_fn, output_dir, mask_ratio, noise_sigma)
    z, G, Gz, opt_z, scaled_lambda_, prox_op = init_model(x_true, sigma_0, L, prior)
    train(b, z, A, sample_intvl, num_iters, opt_z, G, Gz,
          scaled_lambda_, shrinkage_param, rho, prox_op)
