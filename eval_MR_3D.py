import math
import torch
import numpy as np
from signal_utils import get_psnr, get_ssim

if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    subjects = ['0']
    all_steps = [1, 2, 3, 4, 5]

    for subject in subjects:
        path = f'{subject}.h5_FLAIR_3D_BraTS_multi_a0.0_un4_L2_lr1e-05'
        figures_np_t2 = []
        rs = []
        lines = [f'{subject}_{i}' for i in [0, 1, 2]]
        for steps in all_steps:
            r = 1 / steps
            r = math.floor(r * 100) / 100
            interval = r
            highest_psnr_t2 = 0

            i = 0
            while r <= 1:
                u_mask = torch.tensor(np.load('vd.npy')).to(device)
                full_t2 = torch.tensor(np.load(f'outputs/BraTS_3D/{path}/{subject}.h5_{steps}_{r:.2f}_i.npy')).to(device)
                gt = torch.tensor(np.load(f'outputs/BraTS_3D/{path}/{subject}.h5_gt_FLAIR_i.npy')).to(device)
                full_t2 = abs(full_t2[...,0] + full_t2[...,1] * 1j)[...,None]
                gt = abs(gt[...,0] + gt[...,1] * 1j)[...,None]

                t2_psnr = get_psnr(full_t2, gt, 1)
                t2_ssim = get_ssim(full_t2, gt, 1)
                if t2_psnr > highest_psnr_t2:
                    best_r_t2 = i
                    curr_best_fig_t2 = full_t2.clone()
                    highest_psnr_t2 = t2_psnr
                    highest_ssim_t2 = t2_ssim
                r += interval
                i += 1

            print(lines)