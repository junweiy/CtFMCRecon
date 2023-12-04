import os
import argparse
import shutil

import torch
import torch.backends.cudnn as cudnn
import tensorboardX

import numpy as np

from networks import Positional_Encoder, FFN, SIREN_BI
from utils import get_config, prepare_sub_folder, get_mr_data_loader, ct_parallel_project_2d_batch
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
    # parser.add_argument('--config', type=str, default='./configs/mr_recon_i_3D.yaml', help='path to the config file.')
    parser.add_argument('--config', type=str, default='./configs/mr_recon_BraTS_3D.yaml', help='path to the config file.')
    parser.add_argument('--output_path', type=str, default='.', help="outputs path")
    parser.add_argument('--dim', type=str, default='3D', help="dimension of the data")
    parser.add_argument('--multi_contrast', type=bool, default=True, help="whether use T1 to assist")
    parser.add_argument('--contrast', type=str, default='T2', help='contrast to be reconstructed')
    parser.add_argument('--alpha', type=float, default=0.3, help='coefficient to balance between T1 and query contrast')
    parser.add_argument('--un_rate', type=int, default=8, help='undersampling rate, e.g., 4 or 8.')
    parser.add_argument('--eps', type=float, default=1e-8, help='eps used in the training loss.')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size.')
    parser.add_argument('--steps', type=int, default=1, help='steps of coarse to fine.')
    parser.add_argument('--lr', type=float, default=0.00001, help='learning rate')
    parser.add_argument('--files', nargs='+', type=str, default=[])
    parser.add_argument('--ep_thres', type=int, default=3, help='threshold of coarse to fine.')

    # Load experiment setting
    opts = parser.parse_args()
    config = get_config(opts.config)
    max_iter = config['max_iter']
    steps = 1.0 / opts.steps
    steps = math.floor(steps * 100) / 100

    config['net']['multi_contrast'] = opts.multi_contrast

    cudnn.benchmark = True
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    file_list = os.listdir(config['data_folder']) if len(opts.files) == 0 else opts.files
    for curr_image_file in file_list:
        # Setup output folder

        output_folder = os.path.splitext(os.path.basename(opts.config))[0]

        model_name = '{}_{}_{}_{}_{}_a{}_un{}_{}_lr{:.2g}' \
            .format(curr_image_file, opts.contrast, opts.dim, config['dataset_name'], 'multi' if opts.multi_contrast else '', \
                    opts.alpha, opts.un_rate, config['loss'], opts.lr)
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
            optim = torch.optim.AdamW(model.parameters(), lr=opts.lr, betas=(config['beta1'], config['beta2']),
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
                                         batch_size=opts.batch_size, dataset_name=config['dataset_name'])

        test_data_loader = get_mr_data_loader(opts.dim, config['data_folder'] + curr_image_file, train=False,
                                         batch_size=opts.batch_size, dataset_name=config['dataset_name'])

        grid, (t1_k, t2_k, t2f_k), (t1_img, t2_img, t2f_img) = data_loader.dataset.__getitem__(0)


        for un_rate in [opts.un_rate]:

            # for early stopping
            best_loss = 10000000000
            best_t1_loss = 10000000000
            ep = 0
            best_t1_losses = []

            # Current coarse level
            r = steps
            dist_sq = (grid[..., 1] * grid[..., 1] + grid[..., 2] * grid[..., 2])[None, ..., None]
            tmp_mask = torch.zeros_like(grid[None, ..., :1]).to(device)
            tmp_mask[dist_sq <= 2 * r] = 1

            # Train model
            for iterations in range(max_iter):
                optim.zero_grad()
                model.train()

                t1_losses = []
                train_losses = []
                train_psnrs = []
                train_ssims = []

                for it, data in enumerate(data_loader):
                    # image shape [bs, h, w, 2]
                    # grid shape [bs, h, w, 2]
                    grid, (t1_k, t2_k, t2f_k), (t1_img, t2_img, t2f_img) = data
                    vd_shape = t1_k.shape[1:-1]

                    mask = torch.tensor(get_vd_mask(vd_shape, un_rate, square_area=5)[np.newaxis, ..., np.newaxis]).to(device)

                    grid = grid.to(device)
                    t1_k = t1_k.to(device)
                    t1_img = t1_img.to(device)

                    if opts.contrast == 'T2':
                        query_k = t2_k.to(device)
                    elif opts.contrast in ['T2Flair', 'FLAIR']:
                        query_k = t2f_k.to(device)
                    else:
                        raise NotImplementedError


                    train_embedding = encoder.embedding(grid)  # [B, D, H, W, 512]
                    train_output = model(train_embedding)  # [B, D, H, W, 2/4]


                    if opts.multi_contrast:
                        t1_output_k = i_to_k(train_output[..., :2])
                        query_output_k = i_to_k(train_output[..., 2:])
                        t1_loss = 0.5 * loss_fn(t1_output_k, t1_k)
                        query_loss = 0.5 * loss_fn(query_output_k * mask * tmp_mask, query_k * mask * tmp_mask) / (mask * tmp_mask).mean()
                        train_loss = opts.alpha * t1_loss + (1 - opts.alpha) * query_loss
                    else:
                        query_output_k = i_to_k(train_output)
                        train_loss = 0.5 * loss_fn(query_output_k * mask, query_k * mask)

                    if iterations % config['log_iter'] == 0:
                        if opts.multi_contrast:
                            pred = train_output[..., 2:]
                        else:
                            pred = train_output
                        query_i = k_to_i(query_k)
                        train_ssim = get_ssim(pred, query_i, 1)
                        train_psnr = get_psnr(pred, query_i, 1)

                        train_ssims.append(train_ssim)
                        train_psnrs.append(train_psnr)
                        t1_losses.append(t1_loss.item())
                        train_losses.append(train_loss.item())

                        print(
                            "[Iteration {}/{} | Batch {}/{}] Train loss: {:.4g} | Train psnr: {:.4g}, Train ssim: {:.4g}".format(
                                iterations + 1, max_iter, it + 1, len(data_loader), train_loss, train_psnr, train_ssim))

                    train_loss.backward()
                    optim.step()

                if iterations % config['log_iter'] == 0:
                    train_writer.add_scalar('train_loss_T1', np.mean(t1_losses), iterations + 1)
                    train_writer.add_scalar('train_loss', np.mean(train_losses), iterations + 1)
                    train_writer.add_scalar('train_psnr', np.mean(train_psnrs), iterations + 1)
                    train_writer.add_scalar('train_ssim', np.mean(train_ssims), iterations + 1)


                # Compute testing psnr
                if iterations == 0 or (iterations + 1) % config['val_iter'] == 0:
                    model.eval()
                    test_outputs = []
                    t1_imgs = []
                    query_ks = []

                    with torch.no_grad():
                        for it, data in enumerate(test_data_loader):
                            grid, (t1_k, t2_k, t2f_k), (t1_img, t2_img, t2f_img) = data
                            grid = grid.to(device)

                            t1_k = t1_k.to(device)
                            t1_img = t1_img.to(device)

                            if opts.contrast == 'T2':
                                query_k = t2_k.to(device)
                            elif opts.contrast in ['T2Flair', 'FLAIR']:
                                query_k = t2f_k.to(device)
                            else:
                                raise NotImplementedError

                            test_embedding = encoder.embedding(grid)
                            test_output = model(test_embedding)  # [B, H, W, 2/4]
                            test_outputs.append(test_output.clone())
                            t1_imgs.append(t1_img.clone())
                            query_ks.append(query_k.clone())

                        test_outputs = torch.concat(test_outputs)
                        t1_imgs = torch.concat(t1_imgs)
                        query_ks = torch.concat(query_ks)

                        if opts.multi_contrast:
                            pred = test_outputs[..., 2:]
                            # pred_low = test_output_low[..., 2:]
                            # pred_comb = combined_test_output[..., 2:]
                            pred_T1 = test_outputs[..., :2]

                            T1_ssim = get_ssim(pred_T1, t1_imgs, 1)
                            T1_psnr = get_psnr(pred_T1, t1_imgs, 1)
                        else:
                            pred = test_outputs
                        query_is = k_to_i(query_ks)
                        test_ssim = get_ssim(DC(pred, query_is, mask, 1), query_is, 1)
                        test_psnr = get_psnr(DC(pred, query_is, mask, 1), query_is, 1)


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
                        dced = DC(pred, query_is, mask, 1)
                        np.save(output_directory + f'/{curr_image_file}_{opts.steps}_{r:.2f}_T1_i.npy',
                                pred_T1.detach().cpu().numpy())
                        np.save(output_directory + f'/{curr_image_file}_gt_{opts.contrast}_i.npy', query_is.cpu().numpy())
                        np.save(output_directory + f'/{curr_image_file}_gt_T1_i.npy', t1_imgs.cpu().numpy())
                        np.save(output_directory + f'/{curr_image_file}_zf_{opts.contrast}_i.npy', k_to_i((i_to_k(query_is) * mask)).cpu().numpy())
                        torch.save({'net': model.state_dict(),
                                    'enc': encoder.B,
                                    'opt': optim.state_dict(),
                                    }, model_name)
                    else:
                        ep += 1
                        if ep > opts.ep_thres:
                            best_t1_losses.append(best_t1_loss)
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
                                np.savetxt(output_directory + f'/{curr_image_file}_{opts.steps}_{(r-steps):.2f}_t1_losses.npy', [i.detach().cpu().numpy() for i in best_t1_losses])
                                print('Final early stopped.')
                                break
                    train_writer.add_scalar('best_loss', best_loss, iterations + 1)



                if (iterations + 1) % config['log_iter'] == 0 and (iterations + 1) > (max_iter - 50):
                    model_name = os.path.join(checkpoint_directory, 'model_%06d.pt' % (iterations + 1))
                    torch.save({'net': model.state_dict(),
                                'enc': encoder.B,
                                'opt': optim.state_dict(),
                                }, model_name)

