import os
import random
import datetime

import numpy as np
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from data.options import option
from data.data import *
from net.FPANet import FPANet
from loss.losses import *
from data.scheduler import *

opt = option().parse_args()


# --------------------------------------------------------------------------- utils
def seed_torch(seed):
    """Fix every random source. A negative seed draws a new one at every run."""
    if seed is None or seed < 0:
        seed = random.randint(1, 1000000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print('===> Random seed: %d' % seed)
    return seed


def train_init():
    seed_torch(opt.seed)
    cudnn.benchmark = True
    if opt.gpu_mode and not torch.cuda.is_available():
        raise Exception('No GPU found, please run with --gpu_mode False')


# --------------------------------------------------------------------------- train
def train(epoch, model, training_data_loader, optimizer, losses):
    """One epoch of training.

    The gradient handling follows the standard order: zero_grad -> backward ->
    (optional) clip -> step.  In the previous release the clipping was applied
    before the backward pass and therefore had no effect; see the README.
    """
    L1_loss, D_loss, E_loss, P_loss = losses
    model.train()
    loss_last_10 = 0.0
    pic_last_10 = 0
    train_len = len(training_data_loader)
    it = 0

    for batch in tqdm(training_data_loader):
        im1, im2 = batch[0].cuda(), batch[1].cuda()

        # optional random gamma applied to the input; disabled in the reported runs
        if opt.gamma:
            gamma = random.randint(opt.start_gamma, opt.end_gamma) / 100.0
            output_rgb = model(im1 ** gamma)
        else:
            output_rgb = model(im1)

        gt_rgb = im2
        output_hvi = model.HVIT(output_rgb)
        gt_hvi = model.HVIT(gt_rgb)

        loss_hvi = (L1_loss(output_hvi, gt_hvi) + D_loss(output_hvi, gt_hvi)
                    + E_loss(output_hvi, gt_hvi) + opt.P_weight * P_loss(output_hvi, gt_hvi)[0])
        loss_rgb = (L1_loss(output_rgb, gt_rgb) + D_loss(output_rgb, gt_rgb)
                    + E_loss(output_rgb, gt_rgb) + opt.P_weight * P_loss(output_rgb, gt_rgb)[0])
        loss = loss_rgb + opt.HVI_weight * loss_hvi

        optimizer.zero_grad()
        loss.backward()
        if opt.grad_clip:  # disabled in the configuration of the reported results
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip_value, norm_type=2)
        optimizer.step()

        it += 1
        loss_last_10 += loss.item()
        pic_last_10 += 1

        if it == train_len:
            print('===> Epoch[{}]: Loss: {:.4f} || lr={}'.format(
                epoch, loss_last_10 / pic_last_10, optimizer.param_groups[0]['lr']))

    # keep one sample pair from the last batch for a quick visual sanity check
    out_dir = os.path.join(opt.val_folder, 'training')
    os.makedirs(out_dir, exist_ok=True)
    transforms.ToPILImage()(output_rgb[0].detach().cpu().squeeze(0).clamp(0, 1)).save(
        os.path.join(out_dir, 'train_output.png'))
    transforms.ToPILImage()(gt_rgb[0].cpu().squeeze(0)).save(os.path.join(out_dir, 'train_gt.png'))
    return loss_last_10 / max(pic_last_10, 1)


# --------------------------------------------------------------------------- io
def checkpoint(epoch, model):
    out_dir = os.path.join(opt.ckpt_folder, opt.dataset)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'epoch_{}.pth'.format(epoch))
    torch.save(model.state_dict(), path)
    print('Checkpoint saved to {}'.format(path))
    return path


def load_datasets():
    """Build the training loader and the loader of the evaluation split."""
    print('===> Loading datasets')
    if opt.dataset == 'lol_v1':
        train_set = get_lol_training_set(opt.data_train_lol_v1, size=opt.cropSize)
        eval_dir = opt.data_eval_lol_v1
        norm_size = True
    elif opt.dataset == 'lolv2_real':
        train_set = get_lol_v2_training_set(opt.data_train_lolv2_real, size=opt.cropSize)
        eval_dir = opt.data_eval_lolv2_real
        norm_size = True
    elif opt.dataset == 'lolv2_syn':
        train_set = get_lol_v2_syn_training_set(opt.data_train_lolv2_syn, size=opt.cropSize)
        eval_dir = opt.data_eval_lolv2_syn
        norm_size = True
    elif opt.dataset == 'fivek':
        train_set = get_fivek_training_set(opt.data_train_fivek, size=opt.cropSize)
        eval_dir = opt.data_eval_fivek
        norm_size = False
    else:
        raise Exception('unknown dataset: %s' % opt.dataset)

    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads,
                                      batch_size=opt.batchSize, shuffle=opt.shuffle)
    if norm_size:
        eval_set = get_eval_set(eval_dir)
    else:
        eval_set = get_fivek_eval_set(eval_dir)
    testing_data_loader = DataLoader(dataset=eval_set, num_workers=1, batch_size=1, shuffle=False)
    return training_data_loader, testing_data_loader, norm_size


def build_model():
    print('===> Building model')
    model = FPANet().cuda()
    if opt.start_epoch > 0:
        pth = os.path.join(opt.ckpt_folder, opt.dataset, 'epoch_{}.pth'.format(opt.start_epoch))
        model.load_state_dict(torch.load(pth, map_location=lambda storage, loc: storage))
        print('===> Resumed from', pth)
    return model


def make_scheduler(model):
    optz = optim.Adam(model.parameters(), lr=opt.lr)
    if opt.cos_restart_cyclic:
        if opt.start_warmup:
            sched_step = CosineAnnealingRestartCyclicLR(
                optimizer=optz,
                periods=[(opt.nEpochs // 4) - opt.warmup_epochs, (opt.nEpochs * 3) // 4],
                restart_weights=[1, 1], eta_mins=[0.0002, 0.0000001])
            sched = GradualWarmupScheduler(optz, multiplier=1, total_epoch=opt.warmup_epochs,
                                           after_scheduler=sched_step)
        else:
            sched = CosineAnnealingRestartCyclicLR(
                optimizer=optz, periods=[opt.nEpochs // 4, (opt.nEpochs * 3) // 4],
                restart_weights=[1, 1], eta_mins=[0.0002, 0.0000001])
    elif opt.cos_restart:
        if opt.start_warmup:
            sched_step = CosineAnnealingRestartLR(
                optimizer=optz, periods=[opt.nEpochs - opt.warmup_epochs - opt.start_epoch],
                restart_weights=[1], eta_min=1e-7)
            sched = GradualWarmupScheduler(optz, multiplier=1, total_epoch=opt.warmup_epochs,
                                           after_scheduler=sched_step)
        else:
            sched = CosineAnnealingRestartLR(
                optimizer=optz, periods=[opt.nEpochs - opt.start_epoch],
                restart_weights=[1], eta_min=1e-7)
    else:
        raise Exception('should choose a scheduler')
    return optz, sched


def init_loss():
    """Loss terms. The weights follow Section 3.5 of the paper."""
    L1_loss = L1Loss(loss_weight=opt.L1_weight, reduction='mean').cuda()
    D_loss = SSIM(weight=opt.D_weight).cuda()
    E_loss = EdgeLoss(loss_weight=opt.E_weight).cuda()
    P_loss = PerceptualLoss({'conv1_2': 1, 'conv2_2': 1, 'conv3_4': 1, 'conv4_4': 1},
                            perceptual_weight=1.0, criterion='mse').cuda()
    return L1_loss, D_loss, E_loss, P_loss


def gt_dir(dataset):
    return {
        'lol_v1': opt.data_evalgt_lol_v1,
        'lolv2_real': opt.data_evalgt_lolv2_real,
        'lolv2_syn': opt.data_evalgt_lolv2_syn,
        'fivek': opt.data_evalgt_fivek,
    }[dataset]


# --------------------------------------------------------------------------- main
if __name__ == '__main__':
    train_init()
    training_data_loader, testing_data_loader, norm_size = load_datasets()
    model = build_model()
    optimizer, scheduler = make_scheduler(model)
    losses = init_loss()
    torch.autograd.set_detect_anomaly(opt.grad_detect)


    from eval import eval as evaluate
    from measure import metrics

    label_dir = gt_dir(opt.dataset)
    out_root = os.path.join(opt.val_folder, opt.dataset)
    os.makedirs(out_root, exist_ok=True)
    log_dir = os.path.join(opt.val_folder, 'training')
    os.makedirs(log_dir, exist_ok=True)

    psnr, ssim, lpips_scores = [], [], []
    for epoch in range(opt.start_epoch + 1, opt.nEpochs + opt.start_epoch + 1):
        epoch_loss = train(epoch, model, training_data_loader, optimizer, losses)
        scheduler.step()

        if epoch % opt.snapshots == 0:
            ckpt_path = checkpoint(epoch, model)
         
            evaluate(model, testing_data_loader, ckpt_path, out_root + '/',
                     norm_size=norm_size,
                     LOL=(opt.dataset == 'lol_v1'),
                     v2=(opt.dataset == 'lolv2_real'),
                     alpha=0.8)
            avg_psnr, avg_ssim, avg_lpips = metrics(
                os.path.join(out_root, '*'), label_dir, use_GT_mean=False)
            print('===> Avg.PSNR: {:.4f} dB'.format(avg_psnr))
            print('===> Avg.SSIM: {:.4f}'.format(avg_ssim))
            print('===> Avg.LPIPS: {:.4f}'.format(avg_lpips))
            psnr.append(avg_psnr)
            ssim.append(avg_ssim)
            lpips_scores.append(avg_lpips)
        torch.cuda.empty_cache()

    now = datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S')
    log_path = os.path.join(log_dir, 'metrics_{}_{}.md'.format(opt.dataset, now))
    with open(log_path, 'w') as f:
        f.write('dataset: %s\n' % opt.dataset)
        f.write('lr: %s\n' % opt.lr)
        f.write('batch size: %s\n' % opt.batchSize)
        f.write('crop size: %s\n' % opt.cropSize)
        f.write('seed: %s\n' % opt.seed)
        f.write('HVI_weight: %s\nL1_weight: %s\nD_weight: %s\nE_weight: %s\nP_weight: %s\n' % (
            opt.HVI_weight, opt.L1_weight, opt.D_weight, opt.E_weight, opt.P_weight))
        f.write('| Epoch | PSNR | SSIM | LPIPS |\n|---|---|---|---|\n')
        for i in range(len(psnr)):
            f.write('| %d | %.4f | %.4f | %.4f |\n' % (
                opt.start_epoch + (i + 1) * opt.snapshots, psnr[i], ssim[i], lpips_scores[i]))
    print('===> Log written to', log_path)
