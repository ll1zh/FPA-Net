
import argparse
import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from data.data import get_eval_set, get_fivek_eval_set
from net.FPANet import FPANet

# default evaluation inputs and output folders
EVAL_DIR = {
    'lol_v1': './datasets/LOLdataset/eval15/low',
    'lolv2_real': './datasets/LOLv2/Real_captured/Test/Low',
    'lolv2_syn': './datasets/LOLv2/Synthetic/Test/Low',
    'fivek': './datasets/FiveK/test/input',
}


def eval(model, testing_data_loader, model_path, output_folder,
         norm_size=True, LOL=False, v2=False, unpaired=False, alpha=1.0, gamma=1.0):

    model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
    model.eval()
    print('===> Loaded checkpoint:', model_path)

    if LOL:
        model.trans.gated = True
    elif v2 or unpaired:
        model.trans.gated2 = True
        model.trans.alpha = alpha

    os.makedirs(output_folder, exist_ok=True)
    with torch.no_grad():
        for batch in tqdm(testing_data_loader):
            if norm_size:
                inp, name = batch[0], batch[1]
            else:
                inp, name, h, w = batch[0], batch[1], batch[2], batch[3]

            inp = inp.cuda()
            output = model(inp ** gamma)
            output = torch.clamp(output, 0, 1)
            if not norm_size:
                output = output[:, :, :h, :w]

            transforms.ToPILImage()(output.squeeze(0).cpu()).save(os.path.join(output_folder, name[0]))
            torch.cuda.empty_cache()

    if LOL:
        model.trans.gated = False
    elif v2 or unpaired:
        model.trans.gated2 = False
    print('===> Evaluation finished, results in', output_folder)


def main():
    p = argparse.ArgumentParser(description='Evaluate FPA-Net')
    p.add_argument('--dataset', type=str, default='lol_v1',
                   choices=['lol_v1', 'lolv2_real', 'lolv2_syn', 'fivek'])
    p.add_argument('--ckpt', type=str, required=True, help='path of the checkpoint to evaluate')
    p.add_argument('--input', type=str, default='', help='override the evaluation input folder')
    p.add_argument('--out', type=str, default='', help='override the output folder')
    p.add_argument('--alpha', type=float, default=0.8, help='global gain used on LOLv2-Real')
    p.add_argument('--gamma', type=float, default=1.0, help='gamma applied to the input (1.0 in the paper)')
    ep = p.parse_args()

    if not torch.cuda.is_available():
        raise Exception('No GPU found, or need to change CUDA_VISIBLE_DEVICES')

    data_dir = ep.input if ep.input else EVAL_DIR[ep.dataset]
    out_dir = ep.out if ep.out else os.path.join('./output', ep.dataset)
    norm_size = ep.dataset != 'fivek'

    eval_set = get_eval_set(data_dir) if norm_size else get_fivek_eval_set(data_dir)
    loader = DataLoader(dataset=eval_set, num_workers=1, batch_size=1, shuffle=False)

    model = FPANet().cuda()
    eval(model, loader, ep.ckpt, out_dir,
         norm_size=norm_size,
         LOL=(ep.dataset == 'lol_v1'),
         v2=(ep.dataset == 'lolv2_real'),
         alpha=ep.alpha,
         gamma=ep.gamma)


if __name__ == '__main__':
    main()
