
#import bm3d
import math
import torch
#import prox_tv
import numpy as np

from os.path import join
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from skimage.transform import resize
from matplotlib.pyplot import imread, imsave
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio
from skimage.restoration import denoise_tv_chambolle
from skimage.restoration import denoise_nl_means, estimate_sigma


def load_data(G, noise_sigma, args):
    if args.astro:
        img = np.load(args.gt_img_fn) # [nchls,sz,sz]
        num_channels = args.num_bands
    else:
        img = imread(args.gt_img_fn) # [sz,sz,nchls]
        if img.dtype == 'uint8':
            img = img.astype('float32') / 255  # scale to [0, 1]
        elif img.dtype == 'float32':
            img = img.astype('float32')
        else:
            raise TypeError()

        img = np.clip(resize(img, (128, 128)), 0, 1)
        imsave(args.recon_dir + 'true.png', img)
        if len(img.shape) == 2:
            img = img[:,:,np.newaxis]
            num_channels = 1
        else:
            num_channels = 3
        img = img.transpose((2, 0, 1))

    x_true = torch.from_numpy(img).unsqueeze(0).type(args.float_tensor)
    z = torch.zeros_like(x_true).type(args.float_tensor).normal_()
    x = G(z).clone().detach()
    scaled_lambda_ = torch.zeros_like(x, requires_grad=False)\
                          .type(args.float_tensor)
    x.requires_grad, z.requires_grad = False, True

    if args.astro:
        A, At, A_diag = A_inpainting_astro\
            (args.sampled_pixl_id_fn, args.img_sz, args.num_bands)
        b = A((x_true[0]).permute(1,2,0)) # [sz,sz,nchls]
        return num_channels, x_true, A, At, A_diag, b, x, z, scaled_lambda_

    A, At, A_diag = A_inpainting_normal(args.sample_ratio, x_true.numel())
    b = A(x_true.reshape(-1,))
    b = torch.clamp(b + noise_sigma * torch.randn(b.shape).\
                    type(args.float_tensor), 0, 1)

    if num_channels == 3:
        imsave(args.recon_dir+'corrupted.png',
               At(b).reshape(1, num_channels, 128, 128)[0]\
               .permute((1,2,0)).cpu().numpy())
    else:
        imsave(args.recon_dir + 'corrupted.png',
               At(b).reshape(1, num_channels, 128, 128)[0, 0]\
               .cpu().numpy(), cmap='gray')
    return num_channels, x_true, A, At, A_diag, b, x, z, scaled_lambda_


def A_inpainting_astro(mask_fn, img_sz, num_bands):

    mask = np.load(mask_fn)
    if mask.ndim == 2: # [npixls,nchls]
        mask = mask.reshape(img_sz, img_sz, num_bands)

    print('Mask shape ', mask.shape)
    print([np.count_nonzero(mask[:,:,i]) for i in range(mask.shape[-1])])

    mask = mask.astype(bool)

    # x [sz,sz,nchls]
    def A(x): # return unmasked pixel values
        res = x[mask]
        return res

    return A, None, None

def A_inpainting_normal(num_ratio, img_dim):
    if torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    num_measurements = np.round(img_dim * num_ratio).astype('int')
    chosen_ind = np.random.permutation(img_dim)[:num_measurements]
    A_diag= torch.zeros(img_dim).type(dtype)
    A_diag[chosen_ind] = 1

    def A(x):
        return x[chosen_ind]
    def At(b):
        b_ = torch.zeros(img_dim, device=b.device)
        b_[chosen_ind] = b
        return b_
    return A, At, A_diag

def A_superresolution(down_ratio, img_shape):
    '''
    img_shape: (1, 3, h, w), h and w should be divided by 2
    '''
    if torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    img_dim = img_shape[1]*img_shape[2]*img_shape[3]
    
    h_sampling_position = np.arange(0, img_shape[2], down_ratio).reshape((-1,1))
    w_sampling_position = np.arange(0, img_shape[3], down_ratio).reshape((-1,1))

    subsampled_h = len(h_sampling_position)
    subsampled_w = len(w_sampling_position)
    mask = np.zeros(img_shape)
    for h_p in h_sampling_position:
        for w_p in w_sampling_position:
            mask[:,:,h_p, w_p] = 1

    mask = mask.reshape((-1,))
    chosen_ind = np.where(mask==1)
    A_diag= torch.zeros(img_dim).type(dtype)
    A_diag[chosen_ind] = 1

    def A(x):
        return x[chosen_ind]
    def At(b):
        b_ = torch.zeros(img_dim, device=b.device)
        b_[chosen_ind] = b
        return b_
    def down_img(img):
        ## img: (1, 3, h, w)
        img = img.reshape((-1,))
        subsampled_img = img[chosen_ind]
        subsampled_img = subsampled_img.reshape((1,3, subsampled_h, subsampled_w))
        return subsampled_img
    return A, At, A_diag, down_img

def l1_prox(input, lamda):
    t = torch.abs(input) - lamda
    t = t * (t > 0).type(t.dtype)
    return torch.sign(input) * t


def tv_prox(input, lamda):
    numpy_input = input.detach().cpu().numpy()
    numpy_input = numpy_input[0]
    result = []
    for i in numpy_input:
        result.append(prox_tv.tv1_2d(i, w=lamda, n_threads=16, method='pd')[np.newaxis])
    result = np.concatenate(result, 0)
    result = result[np.newaxis]
    return input.new(result)

def bm3d_prox(input, lamda):
    numpy_input = input.detach().cpu().numpy()
    numpy_input = numpy_input[0].transpose((1,2,0))
    result = bm3d.bm3d(numpy_input, lamda)
    result = result.transpose((2,0,1))[np.newaxis]
    return input.new(result)

def nlm_prox(input, lamda):
    numpy_input = input.detach().cpu().numpy()
    numpy_input = numpy_input[0].transpose((1,2,0))
    multichannel = numpy_input.shape[-1] != 1
    result = denoise_nl_means(numpy_input, multichannel=multichannel, sigma=lamda, patch_distance=2, h=0.05)
    result = result.transpose((2,0,1))[np.newaxis]
    return input.new(result)

def linf_proj(input, center, bound):
    numpy_input = input.clone().detach()
    numpy_center = center.clone().detach()
    inp_minus_center = numpy_input - numpy_center

    no_change_part = ((inp_minus_center<=bound) * (inp_minus_center>=-bound))
    above_part = (inp_minus_center>bound)
    below_part = (inp_minus_center<-bound)

    upper_value = numpy_center + bound
    lower_value = numpy_center - bound

    result = no_change_part.float() * numpy_input + above_part.float() * upper_value + below_part.float() * lower_value
    return result.detach()

def projection_simplex_sort(v):
    v = v.clone().detach()
    v_flatten = v.reshape(-1)
    n = v_flatten.shape[0]
    u = torch.sort(v_flatten, descending=True)[0]
    cssv = torch.cumsum(u, dim=0) - 1.
    ind = torch.arange(n, device=u.device) + 1.
    cond = (u - cssv / ind.float() > 0).long()
    rho = torch.nonzero(cond).max() + 1
    theta = cssv[rho - 1] / rho
    w = torch.clamp(v - theta, min=0)
    return w.detach()
    
def proj_l1(v):
    u = torch.abs(v)
    if torch.sum(u) <= 1.:
        return v
    w = projection_simplex_sort(u)
    w *= torch.sign(v)
    return w

def linf_prox(x, shrinkage_param):
    return x - shrinkage_param * proj_l1(x / shrinkage_param)

def linf_prox_x_b(x, b, shrinkage_param):
    return linf_prox(x-b, shrinkage_param) + b


### reconstruction
def get_header(dir, sz):
    hdu = fits.open(join(dir, 'pdr3_dud/calexp-HSC-G-9813-0%2C0.fits'))[1]
    header = hdu.header
    cutout = Cutout2D(hdu.data, position=(sz//2, sz//2),
                      size=sz, wcs=WCS(header))
    return cutout.wcs.to_header()

def reconstruct(epoch, gt, recon, num_channels, args):
    if args.astro:
        id = (epoch + 1) // args.model_smpl_intvl
        recon_path = join(args.recon_dir, str(id))
        (mse, psnr, ssim) = reconstruct_(gt[0], recon[0], recon_path)

        print('[Epoch/Total] [%d/%d] [MSE %.3f] [PSNR %.2f] [SSIM %.3f]'
              % (epoch + 1, args.num_epochs, mse[0], psnr[0], ssim[0]))
        return [mse, psnr, ssim]

    else:
        psnr_gt = peak_signal_noise_ratio(gt, recon)
        mse_gt = np.mean((gt - recon) ** 2)
        #fidelity_loss = fn(results).detach()

        if num_channels == 3:
            imsave(join(args.recon_dir, 'iter%d_PSNR_%.2f.png'%(epoch, psnr_gt)),
                   recon[0].transpose((1,2,0)))
        else:
            imsave(join(args.recon_dir, 'iter%d_PSNR_%.2f.png'%(epoch, psnr_gt)),
                   recon[0,0], cmap='gray')

        #record["psnr_gt"].append(psnr_gt)
        #record["mse_gt"].append(mse_gt)
        #record["fidelity_loss"].append(fidelity_loss.item())
        print('[Epoch/Total] [%d/%d] [PSNR %.2f] [MSE %.3f]'
              % (epoch + 1, args.num_epochs, psnr_gt, mse_gt))
        return None

def reconstruct_(gt, recon, recon_path=None, header=None):
    # gt/recon, [c,h,w]
    sz = gt.shape[1]

    if recon_path is not None:
        np.save(recon_path + '.npy', recon)

    #print('GT max', np.round(np.max(gt, axis=(1,2)), 3) )
    #print('Recon pixl max ', np.round(np.max(recon, axis=(1,2)), 3) )
    #print('Recon stat ', round(np.min(recon), 3), round(np.median(recon), 3),
    #      round(np.mean(recon), 3), round(np.max(recon), 3))

    # [noptions,nchls]
    losses = get_losses(gt, recon, None, [1,2,4])

    if header is not None:
        hdu = fits.PrimaryHDU(data=recon, header=header)
        hdu.writeto(recon_path + '.fits', overwrite=True)

    return losses

def calculate_ssim(gt, gen):
    rg = np.max(gt)-np.min(gt)
    return structural_similarity(gt, gen, data_range=rg)
                                 #win_size=len(org_img))
def calculate_psnr(gen, gt):
    mse = calculate_mse(gen, gt)
    mx = np.max(gt)
    return 20 * np.log10(mx / np.sqrt(mse))

def calculate_mse(gen, gt):
    mse = np.mean((gen - gt)**2)
    return mse

# calculate normalized cross correlation between given 2 imgs
def calculate_ncc(img1, img2):
    a, b = img1.flatten(), img2.flatten()
    n = len(a)
    return 1/n * np.sum( (a-np.mean(a)) * (b-np.mean(b)) /
                         np.sqrt(np.var(a)*np.var(b)) )

def get_loss(gt, gen, mx, j, option):
    if option == 0:
        loss = np.abs(gt[j] - gen[j]).mean()
    elif option == 1:
        loss = calculate_mse(gen[j], gt[j])
    elif option == 2:
        loss = calculate_psnr(gen[j], gt[j])
    elif option == 3:
        loss = calculate_sam(gen[:,:,j:j+1], gt[:,:,j:j+1])
    elif option == 4:
        loss = calculate_ssim(gen[j], gt[j])
    elif option == 5: # min
        loss = np.min(gen[j])
    elif option == 6: # max
        loss = np.max(gen[j])
    elif option == 7: # meam
        loss = np.mean(gen[j])
    elif option == 8: # median
        loss = np.median(gen[j])
    return loss

# calculate losses between gt and gen based on options
def get_losses(gt, gen, mx, options):
    nchl = gen.shape[0]
    losses = np.zeros((len(options), nchl))

    for i, option in enumerate(options):
        for j in range(nchl):
            losses[i, j] = get_loss(gt, gen, mx, j, option)
    return losses
