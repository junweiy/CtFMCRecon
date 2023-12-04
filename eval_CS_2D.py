import torch
import numpy as np
from signal_utils import get_psnr, get_ssim

if __name__ == '__main__':
    subjects = ['0_0', '0_1', '0_2']
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    for subject in subjects:
        path = f'{subject}.h5_FLAIR_2D_BraTS_a0.01_un8_wav'
        figures_np_t2 = []
        rs = []
        line = f'{subject}'
        full_t2 = torch.tensor(np.load(f'outputs/CS_BraTS_2D/{path}/pred.npy')).to(device)[...,None]
        gt = torch.tensor(np.load(f'outputs/CS_BraTS_2D/{path}/gt.npy')).to(device)[0,...,None]

        t2_psnr = get_psnr(full_t2, gt, 1)
        t2_ssim = get_ssim(full_t2, gt, 1)
        print(line)
        print(f'PSNR: {t2_psnr:.2f}, SSIM: {t2_ssim:.3f}')