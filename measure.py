import argparse
import glob
import os
import platform

import cv2
import lpips
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# default folders: enhanced images written by eval.py and the corresponding ground truth
DEFAULT_INPUT = {
    'lol_v1': './output/lol_v1/',
    'lolv2_real': './output/lolv2_real/',
    'lolv2_syn': './output/lolv2_syn/',
    'fivek': './output/fivek/',
}
DEFAULT_GT = {
    'lol_v1': './datasets/LOLdataset/eval15/high/',
    'lolv2_real': './datasets/LOLv2/Real_captured/Test/Normal/',
    'lolv2_syn': './datasets/LOLv2/Synthetic/Test/Normal/',
    'fivek': './datasets/FiveK/test/target/',
}


def ssim(prediction, target):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = prediction.astype(np.float64)
    img2 = target.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(target, ref):
    """SSIM, matching the implementation used in the paper."""
    img1 = np.array(target, dtype=np.float64)
    img2 = np.array(ref, dtype=np.float64)
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    if img1.ndim == 3:
        if img1.shape[2] == 3:
            return np.array([ssim(img1[:, :, i], img2[:, :, i]) for i in range(3)]).mean()
        if img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
        raise ValueError('Wrong number of channels.')
    raise ValueError('Wrong input image dimensions.')


def calculate_psnr(target, ref):
    img1 = np.array(target, dtype=np.float32)
    img2 = np.array(ref, dtype=np.float32)
    diff = img1 - img2
    return 10.0 * np.log10(255.0 * 255.0 / (np.mean(np.square(diff)) + 1e-8))


def metrics(im_dir, label_dir, use_GT_mean=False):
    """Return the average PSNR, SSIM and LPIPS over the images matched by ``im_dir``."""
    avg_psnr = avg_ssim = avg_lpips = 0.0
    n = 0
    loss_fn = lpips.LPIPS(net='alex').cuda()
    for item in tqdm(sorted(glob.glob(im_dir))):
        n += 1
        im1 = Image.open(item).convert('RGB')
        sep = '\\' if platform.system().lower() == 'windows' else '/'
        name = item.split(sep)[-1]
        im2 = Image.open(os.path.join(label_dir, name)).convert('RGB')

        (h, w) = im2.size
        im1 = im1.resize((h, w))
        im1, im2 = np.array(im1), np.array(im2)

        if use_GT_mean:
            mean_restored = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY).mean()
            mean_target = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY).mean()
            im1 = np.clip(im1 * (mean_target / mean_restored), 0, 255)

        avg_psnr += calculate_psnr(im1, im2)
        avg_ssim += calculate_ssim(im1, im2)
        avg_lpips += loss_fn.forward(lpips.im2tensor(im2).cuda(), lpips.im2tensor(im1).cuda()).item()
        torch.cuda.empty_cache()

    if n == 0:
        raise RuntimeError('no image found in %s' % im_dir)
    return avg_psnr / n, avg_ssim / n, avg_lpips / n


def main():
    p = argparse.ArgumentParser(description='Measure PSNR / SSIM / LPIPS for FPA-Net')
    p.add_argument('--dataset', type=str, default='lol_v1',
                   choices=['lol_v1', 'lolv2_real', 'lolv2_syn', 'fivek'])
    p.add_argument('--input', type=str, default='', help='folder of enhanced images')
    p.add_argument('--gt', type=str, default='', help='folder of ground-truth images')
    p.add_argument('--use_GT_mean', action='store_true',
                   help='rescale each output to the mean of the ground truth (disabled for the paper)')
    mea = p.parse_args()

    im_dir = mea.input if mea.input else DEFAULT_INPUT[mea.dataset]
    label_dir = mea.gt if mea.gt else DEFAULT_GT[mea.dataset]
    if not im_dir.endswith('*'):
        im_dir = os.path.join(im_dir, '*')

    avg_psnr, avg_ssim, avg_lpips = metrics(im_dir, label_dir, mea.use_GT_mean)
    print('===> Avg.PSNR: {:.4f} dB'.format(avg_psnr))
    print('===> Avg.SSIM: {:.4f}'.format(avg_ssim))
    print('===> Avg.LPIPS: {:.4f}'.format(avg_lpips))


if __name__ == '__main__':
    main()
