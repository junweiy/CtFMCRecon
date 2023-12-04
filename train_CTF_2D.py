import os
import argparse
import shutil

import torch
import torch.backends.cudnn as cudnn
import tensorboardX

import numpy as np

from networks import Positional_Encoder, FFN, SIREN_BI
from utils import get_config, prepare_sub_folder, get_mr_data_loader
from common_masks import get_vd_mask
from signal_utils import get_ssim, i_to_k, k_to_i, get_psnr, DC
from utils import k_loss
import random
import math


torch.manual_seed(13)
torch.cuda.manual_seed(13)
np.random.seed(13)
random.seed(13)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/mr_recon_BraTS_2D.yaml', help='path to the config file.')
    parser.add_argument('--output_path', type=str, default='.', help="outputs path")
    parser.add_argument('--dim', type=str, default='2D', help="dimension of the data")
    parser.add_argument('--multi_contrast', type=bool, default=True, help="whether use T1 to assist")
    parser.add_argument('--contrast', type=str, default='T2', help='contrast to be reconstructed')
    parser.add_argument('--alpha', type=float, default=0.1, help='coefficient to balance between T1 and query contrast')
    parser.add_argument('--un_rate', type=int, default=8, help='undersampling rate, e.g., 4 or 8.')
    parser.add_argument('--eps', type=float, default=1e-8, help='eps used in the training loss.')
    parser.add_argument('--steps', type=int, default=1, help='steps of coarse to fine.')
    parser.add_argument('--files', nargs='+', type=str, default=[])

    # Load experiment setting
    opts = parser.parse_args()
    config = get_config(opts.config)
    max_iter = config['max_iter']
    steps = 1.0 / opts.steps
    steps = math.floor(steps * 100) / 100
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    config['net']['multi_contrast'] = opts.multi_contrast

    cudnn.benchmark = True
    file_list = os.listdir(config['data_folder']) if len(opts.files) == 0 else opts.files

    for curr_image_file in file_list:
        # Setup output folder

        output_folder = os.path.splitext(os.path.basename(opts.config))[0]

        model_name = '{}_{}_{}_{}_{}_a{}_un{}_{}_lr{:.2g}' \
            .format(curr_image_file, opts.contrast, opts.dim, config['dataset_name'], 'multi' if opts.multi_contrast else '', \
                    opts.alpha, opts.un_rate, config['loss'], config['lr'])
        print(model_name)

        train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path + "/logs/" + model_name + '/' + str(opts.steps)))
        output_directory = os.path.join(opts.output_path + "/outputs/" + model_name)
        checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
        shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml'))  # copy config file to output folder

        # Setup input encoder:
        encoder = Positional_Encoder(config['encoder'])
        # Setup model
        if config['model'] == 'SIREN':
            model = SIREN_BI(config['net'])
        elif config['model'] == 'FFN':
            model = FFN(config['net'])
        else:
            raise NotImplementedError
        model.to(device)
        model.train()

        # Setup optimizer
        if config['optimizer'] == 'Adam':
            optim = torch.optim.AdamW(model.parameters(), lr=config['lr'], betas=(config['beta1'], config['beta2']),
                                     weight_decay=config['weight_decay'])
        else:
            NotImplementedError

        # Setup loss function
        if config['loss'] == 'L2':
            loss_fn = torch.nn.MSELoss()
        elif config['loss'] == 'L1':
            loss_fn = torch.nn.L1Loss()
        elif config['loss'] == 'kloss':
            loss_fn = k_loss
        else:
            NotImplementedError


        # Setup data loader
        print('Load image: {}'.format(curr_image_file))
        data_loader = get_mr_data_loader(opts.dim, config['data_folder'] + curr_image_file, train=True,
                                         batch_size=config['batch_size'], dataset_name=config['dataset_name'])

        grid, (t1_k, t2_k, t2f_k), (t1_img, t2_img, t2f_img) = data_loader.dataset.__getitem__(0)
        np.save(output_directory + f'/{curr_image_file}_gt_T2_i.npy', t2_img.cpu().numpy()[np.newaxis,...])
        np.save(output_directory + f'/{curr_image_file}_gt_T2f_i.npy', t2f_img.cpu().numpy()[np.newaxis, ...])
        np.save(output_directory + f'/{curr_image_file}_gt_T1_i.npy', t1_img.cpu().numpy()[np.newaxis, ...])
        # continue

        #

        for un_rate in [opts.un_rate]:
            for it, data in enumerate(data_loader):
                # image shape [bs, h, w, 2]
                # grid shape [bs, h, w, 2]
                grid, (t1_k, t2_k, t2f_k), (t1_img, t2_img, t2f_img) = data

                mask = torch.tensor(get_vd_mask(t1_k.shape[1:-1], un_rate, square_area=4)[np.newaxis, ..., np.newaxis]).to(device)

                grid = grid.to(device)
                t1_k = t1_k.to(device)
                t1_img = t1_img.to(device)

                if opts.contrast == 'T2':
                    query_k = t2_k.to(device)
                    query_img = k_to_i(query_k)
                elif opts.contrast == 'FLAIR':
                    query_k = t2f_k.to(device)
                    query_img = k_to_i(query_k)
                else:
                    raise NotImplementedError


                print(grid.shape, t1_k.shape)
                # for early stopping
                best_loss = 10000000000
                best_t1_loss = 10000000000
                ep = 0
                t1_losses = []

                # Current coarse level
                r = steps
                dist_sq = (grid[..., 0] * grid[..., 0] + grid[..., 1] * grid[..., 1])[..., None]
                tmp_mask = torch.zeros_like(t1_k[..., :1])
                tmp_mask[dist_sq <= 2 * r] = 1

                # Train model
                for iterations in range(max_iter):
                    optim.zero_grad()
                    model.train()

                    train_embedding = encoder.embedding(grid)  # [B, H, W, embedding*2]
                    train_output = model(train_embedding)  # [B, H, W, 2/4]


                    if opts.multi_contrast:
                        t1_output_k = i_to_k(train_output[..., :2])
                        query_output_k = i_to_k(train_output[..., 2:])
                        t1_loss = 0.5 * loss_fn(t1_output_k, t1_k)
                        query_loss = 0.5 * loss_fn(query_output_k * mask * tmp_mask, query_k * mask * tmp_mask) / (mask * tmp_mask).mean()
                        train_loss = opts.alpha * t1_loss + (1 - opts.alpha) * query_loss
                    else:
                        query_output_k = i_to_k(train_output)
                        train_loss = 0.5 * loss_fn(query_output_k * mask, query_k * mask)

                    train_loss.backward()
                    optim.step()

                    # Compute training psnr
                    if (iterations + 1) % config['log_iter'] == 0:

                        train_loss = train_loss.item()
                        t1_loss = t1_loss.item()

                        if opts.multi_contrast:
                            pred = train_output[..., 2:]
                        else:
                            pred = train_output
                        query_i = k_to_i(query_k)
                        train_ssim = get_ssim(pred, query_i, 1)
                        train_psnr = get_psnr(pred, query_i, 1)
                        train_writer.add_scalar('train_loss_T1', t1_loss, iterations + 1)
                        train_writer.add_scalar('train_loss', train_loss, iterations + 1)

                        zf_psnr = get_psnr(query_k * mask, query_k, 0)
                        zf_ssim = get_ssim(query_k * mask, query_k, 0)

                        train_writer.add_scalar('train_psnr', train_psnr, iterations + 1)
                        train_writer.add_scalar('train_ssim', train_ssim, iterations + 1)

                        print(
                            "[Iteration: {}/{}] Train loss: {:.4g} | Train psnr: {:.4g}, Train ssim: {:.4g} | ZF PSNR: {:.4g} | SSIM: {:.4g}".format(
                                iterations + 1, max_iter, train_loss, train_psnr, train_ssim, zf_psnr, zf_ssim))

                    # Compute testing psnr
                    if iterations == 0 or (iterations + 1) % config['val_iter'] == 0:
                        model.eval()

                        with torch.no_grad():
                            test_embedding = encoder.embedding(grid)
                            test_output = model(test_embedding)  # [B, H, W, 2/4]



                            if opts.multi_contrast:
                                pred = test_output[..., 2:]
                                pred_T1 = test_output[..., :2]

                                T1_ssim = get_ssim(pred_T1, t1_img, 1)
                                T1_psnr = get_psnr(pred_T1, t1_img, 1)
                            else:
                                pred = test_output
                            query_i = k_to_i(query_k)
                            test_ssim = get_ssim(DC(pred, query_i, mask, 1), query_i, 1)
                            test_psnr = get_psnr(DC(pred, query_i, mask, 1), query_i, 1)


                        val_psnr = test_psnr
                        train_writer.add_scalar('test_psnr', test_psnr, iterations + 1)
                        train_writer.add_scalar('test_ssim', test_ssim, iterations + 1)
                        if opts.multi_contrast:
                            train_writer.add_scalar('test_psnr_T1', T1_psnr, iterations + 1)
                            train_writer.add_scalar('test_ssim_T1', T1_ssim, iterations + 1)
                            print("T1 | PSNR {:.4g} | SSIM {:.4g}".format(T1_psnr, T1_ssim))
                        print("[Iteration: {}/{}] | Test psnr: {:.4g}, Test ssim: {:.4g}".format(iterations + 1,
                                                                                                 max_iter,
                                                                                                 test_psnr,
                                                                                                               test_ssim))
                        if train_loss < best_loss:
                            best_loss = train_loss
                            best_t1_loss = t1_loss
                            ep = 0
                            model_name = os.path.join(checkpoint_directory, f'model_{opts.steps}_{r:.2f}_es.pt')
                            np.save(output_directory + f'/{curr_image_file}_{opts.steps}_{r:.2f}_i.npy',
                                    DC(pred, query_i, mask, 1).detach().cpu().numpy())
                            np.save(output_directory + f'/{curr_image_file}_{opts.steps}_{r:.2f}_T1_i.npy',
                                    pred_T1.detach().cpu().numpy())
                            torch.save({'net': model.state_dict(),
                                        'enc': encoder.B,
                                        'opt': optim.state_dict(),
                                        }, model_name)
                        else:
                            ep += 1
                            if ep > 2:
                                t1_losses.append(best_t1_loss)
                                ep = 0
                                best_loss = 10000000000
                                best_t1_loss = 10000000000
                                print('Early stopping at iteration {}, r={}.'.format(iterations + 1, r))
                                # load weight
                                model_name = os.path.join(checkpoint_directory, f'model_{opts.steps}_{r:.2f}_es.pt')
                                state_dict = torch.load(model_name)
                                model.load_state_dict(state_dict['net'])
                                encoder.B = state_dict['enc']
                                # reset r
                                r += steps
                                tmp_mask[dist_sq <= 2 * r] = 1
                                # check r
                                if r > 1:
                                    np.savetxt(output_directory + f'/{curr_image_file}_{opts.steps}_{(r-steps):.2f}_t1_losses.npy', t1_losses)
                                    print('Final early stopped.')
                                    break


                    if (iterations + 1) % config['log_iter'] == 0 and (iterations + 1) > (max_iter - 50):
                        # Save final model
                        model_name = os.path.join(checkpoint_directory, 'model_%06d.pt' % (iterations + 1))
                        torch.save({'net': model.state_dict(),
                                    'enc': encoder.B,
                                    'opt': optim.state_dict(),
                                    }, model_name)

