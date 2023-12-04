import os
import argparse
import h5py
import sys

os.environ['TOOLBOX_PATH'] = './bart-0.8.00/'
sys.path.append('./bart-0.8.00/python')

from bart import bart

import numpy as np

from utils import get_config, prepare_sub_folder

import random

np.random.seed(13)
random.seed(13)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/mr_recon_BraTS_2D.yaml', help='path to the config file.')
    parser.add_argument('--output_path', type=str, default='.', help="outputs path")
    parser.add_argument('--dim', type=str, default='2D', help="dimension of the data")
    parser.add_argument('--contrast', type=str, default='T2', help='contrast to be reconstructed')
    parser.add_argument('--un_rate', type=int, default=4, help='undersampling rate, e.g., 4 or 8.')
    parser.add_argument('--reg', type=str, default='wav', help='regularizer, TV or wavelet')
    parser.add_argument('--files', nargs='+', type=str, default=[])
    parser.add_argument('--alpha', type=float, default=0.01, help='regularizer term.')

    # Load experiment setting
    opts = parser.parse_args()
    config = get_config(opts.config)
    file_list = os.listdir(config['data_folder']) if len(opts.files) == 0 else opts.files

    for curr_image_file in file_list:
        output_folder = os.path.splitext(os.path.basename(opts.config))[0]

        model_name = '{}_{}_{}_{}_a{}_un{}_{}' \
            .format(curr_image_file, opts.contrast, opts.dim, config['dataset_name'],
                    opts.alpha, opts.un_rate, opts.reg)
        print(model_name)

        output_directory = os.path.join(opts.output_path + "/outputs/" + model_name)
        checkpoint_directory, image_directory = prepare_sub_folder(output_directory)

        h5_file = h5py.File(config['data_folder'] + curr_image_file, 'r')
        if config['dataset_name'] == 'BraTS':
            t2_k = h5_file['T2'][()]
            t2f_k = h5_file['FLAIR'][()]
        else:
            raise NotImplementedError

        h5_file.close()

        t2_img = np.fft.ifft2(np.fft.ifftshift(t2_k))
        t2f_img = np.fft.ifft2(np.fft.ifftshift(t2f_k))

        t2_k = t2_k[0, ..., None]
        t2f_k = t2f_k[0, ..., None]


        nx, ny, nc = t2_k.shape
        sens = np.ones_like(t2_k)

        if config['dataset_name'] == 'BraTS':
            mask = np.load(f'vd_{opts.un_rate}_BraTS.npy')[0]
            mask[nx // 2 - 4:nx // 2 + 4, ny // 2 - 4:ny // 2 + 4] = 1
        else:
            raise NotImplementedError
        t2_k_un = t2_k * mask
        t2f_k_un = t2f_k * mask
        inp = t2_k_un if opts.contrast == 'T2' else t2f_k_un
        gt = t2_img if opts.contrast == 'T2' else t2f_img

        if opts.reg == 'TV':
            recon = bart(1, f'pics -d5 -S -i 100 -R T:7:0:{opts.alpha}', inp[:,:,None,:], sens[:,:,None,:])
        elif opts.reg == 'wav':
            recon = bart(1, f'pics -d5 -S -i 100 -R W:7:0:{opts.alpha}', inp[:, :, None, :], sens[:, :, None, :])
        else:
            raise NotImplementedError

        recon /= abs(recon).max()
        recon = abs(np.fft.ifftshift(recon))
        np.save(f'{output_directory}/pred.npy', recon)
        np.save(f'{output_directory}/gt.npy', abs(gt))
