import math
import torch
import numpy as np
from signal_utils import get_psnr, get_ssim


if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    subjects = ['0_0', '0_1', '0_2']
    all_steps = [1, 2, 3, 4, 5]

    for subject in subjects:
        path = f'{subject}.h5_T2_2D_BraTS_multi_a0.0_un8_L2_lr5e-05'
        figures_np_t2 = []
        rs = []
        line = f'{subject}'
        for steps in all_steps:
            r = 1 / steps
            r = math.floor(r * 100) / 100
            interval = r
            highest_psnr_t2 = 0

            i = 0
            while r <= 1:
                u_mask = torch.tensor(np.load('vd.npy')).to(device)
                full_t2 = torch.tensor(np.load(f'outputs/BraTS_2D/{path}/{subject}.h5_{steps}_{r:.2f}_i.npy')).to(device)
                gt = torch.tensor(np.load(f'outputs/BraTS_2D/{path}/{subject}.h5_gt_T2_i.npy')).to(device)
                full_t2 = abs(full_t2[...,0] + full_t2[...,1] * 1j)
                gt = abs(gt[...,0] + gt[...,1] * 1j)

                t2_psnr = get_psnr(full_t2, gt, 1)
                t2_ssim = get_ssim(full_t2[...,None], gt[...,None], 1)
                if t2_psnr > highest_psnr_t2:
                    best_r_t2 = i
                    curr_best_fig_t2 = full_t2.clone()
                    highest_psnr_t2 = t2_psnr
                    highest_ssim_t2 = t2_ssim
                print(f'{r}, {t2_psnr:.2f}, {t2_ssim:.3f}')
                r += interval
                i += 1

        print(line)