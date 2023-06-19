# check reconstruction error map
# --------------------------------------------------------
import argparse
import datetime
import numpy as np
import os
import pdb
import random
from pathlib import Path
from functools import partial
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt

from spirl import models_mae, mae_misc
from spirl.rainbow_utils import str2bool, ENV_NAME_TO_GYM_NAME
from spirl.draw_utils import cv2_draw_grid, mark_img_loss_gray, display_instances, \
                           ax_plot, ax_hist, ax_line_plot, draw_mask
from spirl.pretrain_dataset import AtariNormalDataset, AtariOfflineDataset
from spirl.models_mae import MaskedAutoencoderViT
import spirl.lr_sched as lr_sched


def get_signal(arr, delt_x, thres=1e-3):
    """
    thres: do not calculate ret if the original value smaller than thres
    """
    ret = np.zeros_like(arr)
    for t in range(len(arr) - 1):
        if arr[t] < thres:
            ret[t] = 0
        else:
            # pdb.set_trace()
            ret[t] = (arr[t+1]-arr[t])/(np.sqrt(1+(arr[t+1]/delt_x)**2)*np.sqrt(1+(arr[t]/delt_x)**2))
    ret[len(arr) - 1] = ret[len(arr) - 2]
    argmax_t = np.argmax(ret)
    return ret, argmax_t


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    # Model parameters
    # NOTE: in timm, for ImageNet dataset:
    #       in this page (https://pytorch.org/vision/stable/models.html), it says that
    #       "All pre-trained models expect input images normalized in the same way, 
    #       i.e. mini-batches of 3-channel RGB images of shape (3 x H x W),
    #       where H and W are expected to be at least 224. 
    #       The images have to be loaded in to a range of [0, 1] 
    #       and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]
    parser.add_argument('--data_type', default='atarizoo', choices=['random', 'atarizoo'],
                        help='type for data source to evaluate mae.')
    parser.add_argument("--mae_pretrained_path", type=str,
                        help="path to load pretrained mae; if None, won't load ckpt for ViT feature extractor")
    # -------- params related to env and keypoints --------
    parser.add_argument('--num_eval_data', default=100, type=int, help="total number of training data")
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--output_dir', default='./log/MAE_pretrain', type=str, help='path where to save log & ckpts, empty for no saving')
    parser.add_argument('--quick_log', default=False, type=str2bool, help='use it for lhpo to shorten log path! True if want to save log in the current path directly')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                        help='device to use for training / testing, for torch.device(args.device)')
    parser.add_argument('--num_workers', default=10, type=int,
                        help=' how many subprocesses to use for data loading. 0 means that the data will be loaded in the main process.')
    parser.add_argument('--pin_mem', default=True, type=str2bool,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')

    parser.add_argument('--show_type', default='ratio', type=str,
                        help='flag for special ways to load ckpt',
                        choices=['ratio', 'thres', 'weight', 'all', 'stat', 'bkg', 'dynamic_thres_deltx',
                        'dynamic_thres_k', 'dynamic_thres_45'])
    parser.add_argument('--min_mask_ratio', default=0.0, type=float,
                        help='to restrict the number of selected patches when using rec_thres')
    parser.add_argument('--log_eps', default=1e-10)

    return parser


@torch.no_grad()
def evaluate(data_loader, model, device, img_save_path, image_size, args):
    img_save_path = os.path.join(img_save_path)
    os.makedirs(img_save_path, exist_ok=True)

    metric_logger = mae_misc.MetricLogger(delimiter="  ")
    header = 'Test:'
    # switch to evaluation mode
    model.eval()

    img_cnt = 0
    for samples in metric_logger.log_every(data_loader, 10, header):
        # eval with each patches
        img_ls = samples
        img_ls = img_ls.to(device, non_blocking=True) if img_ls is not None else img_ls
        kp_pos_ls = None
        samples = (img_ls, kp_pos_ls)
        
        if args.show_type == 'ratio':
            num_row = 2
            num_column = 7
            weight_column = 5
            height_row = 5
        elif args.show_type == 'thres':
            num_row = 5
            num_column = 6
            weight_column = 5
            height_row = 5
        elif args.show_type == 'weight':
            num_row = 5
            num_column = 6
            weight_column = 5
            height_row = 5
        elif args.show_type == 'all':
            num_row = 3
            num_column = 8
            weight_column = 5
            height_row = 5
        elif args.show_type == 'stat':  # statistics about error map
            num_row = 3
            num_column = 6
            weight_column = 20
            height_row = 5
        elif args.show_type == 'dynamic_thres_deltx':
            num_row = 7
            num_column = 5
            weight_column = 5
            height_row = 5
        elif args.show_type == 'dynamic_thres_k':
            num_row = 5
            num_column = 6
            weight_column = 10
            height_row = 5
        elif args.show_type == 'dynamic_thres_45':
            num_row = 5
            num_column = 4
            weight_column = 10
            height_row = 5
        elif args.show_type == 'bkg':  # check how can the background can be reproduced
            num_row = 2
            num_column = 4
            weight_column = 5
            height_row = 5
        else:
            raise NotImplementedError

        fig, axes = plt.subplots(num_row, num_column, figsize=(num_column*weight_column, num_row*height_row))  # figsize: (weight, height)

        with torch.cuda.amp.autocast():
            # reconstruct with only cls token
            loss_mask_all, pred_mask_all, mask = model(
                (samples[0], None), mask_ratio=1, loss_all=True)
            pred_mask_all = model.unpatchify(pred_mask_all)

            surr_loss_idx_mat = np.zeros(model.patch_embed.num_patches)
            for idx in range(model.patch_embed.num_patches):
                # for each single patch, keep its surrounding patchs and calculate loss for the center patch
                surr_loss_all, surr_loss_idx, surr_pred, _ = model(
                    (samples[0], None), surr_mask_idx=idx, loss_all=True)
                surr_loss_idx_mat[idx] = min(surr_loss_idx.item(), 1) # NOTE: truncate data to be less 1 (just a safe truncation, almost no case reachs 1)
            surr_loss_idx_mat = surr_loss_idx_mat.reshape((model.num_patch_per_row, model.num_patch_per_row))

        num_samples = samples[0].shape[0]
        assert num_samples == 1 # to get correct loss (normaly the loss will be averaged over a whoel batch, thus here we constraint the batch_size is 1)
        img_cnt += 1
        img_save_dir = img_save_path

        plot_cnt = 0  # have ploted 0 image
        ori_img = np.einsum('chw->hwc', samples[0][0].cpu().numpy())
        ax_plot(image=ori_img, ax=axes[plot_cnt//num_column][plot_cnt%num_column],
                title=f'original')
        plot_cnt += 1

        pred_img = pred_mask_all[0].cpu().numpy()
        img_c, img_h, img_w = pred_img.shape
        pred_img = MinMaxScaler(feature_range=(0, 1)) \
                        .fit_transform(pred_img.reshape(-1, 1)) \
                        .reshape((img_c, img_h, img_w))
        recon_mask_all = np.einsum('chw->hwc', pred_img).astype('float32')
        ax_plot(image=recon_mask_all, ax=axes[plot_cnt//num_column][plot_cnt%num_column],
                title=f'rec with only [cls]')
        plot_cnt += 1

        if args.show_type == 'bkg':

            with torch.cuda.amp.autocast():
                pred_zero, pred_zero_pos = model.pred_encoder()
                pred_zero = model.unpatchify(pred_zero)
                pred_zero_pos = model.unpatchify(pred_zero_pos)

                pred_decoder, pred_decoder_no_pos, pred_decoder_no_mask = model.pred_decoder()
                pred_decoder = model.unpatchify(pred_decoder)
                pred_decoder_no_pos = model.unpatchify(pred_decoder_no_pos)
                pred_decoder_no_mask = model.unpatchify(pred_decoder_no_mask)

            pred_img = pred_zero[0].cpu().numpy()
            img_c, img_h, img_w = pred_img.shape
            pred_img = MinMaxScaler(feature_range=(0, 1)) \
                            .fit_transform(pred_img.reshape(-1, 1)) \
                            .reshape((img_c, img_h, img_w))
            recon_mask_all = np.einsum('chw->hwc', pred_img).astype('float32')
            ax_plot(image=recon_mask_all, ax=axes[plot_cnt//num_column][plot_cnt%num_column],
                    title=f'rec with zero embedding')
            plot_cnt += 1

            pred_img = pred_zero_pos[0].cpu().numpy()
            # img_c, img_h, img_w = pred_img.shape
            pred_img = MinMaxScaler(feature_range=(0, 1)) \
                            .fit_transform(pred_img.reshape(-1, 1)) \
                            .reshape((img_c, img_h, img_w))
            recon_mask_all = np.einsum('chw->hwc', pred_img).astype('float32')
            ax_plot(image=recon_mask_all, ax=axes[plot_cnt//num_column][plot_cnt%num_column],
                    title=f'rec with zero+pos embedding')
            plot_cnt += 1

            pred_img = pred_decoder[0].cpu().numpy()
            # img_c, img_h, img_w = pred_img.shape
            pred_img = MinMaxScaler(feature_range=(0, 1)) \
                            .fit_transform(pred_img.reshape(-1, 1)) \
                            .reshape((img_c, img_h, img_w))
            recon_mask_all = np.einsum('chw->hwc', pred_img).astype('float32')
            ax_plot(image=recon_mask_all, ax=axes[plot_cnt//num_column][plot_cnt%num_column],
                    title=f'rec with decoder [mask] + pos')
            plot_cnt += 1

            pred_img = pred_decoder_no_pos[0].cpu().numpy()
            # img_c, img_h, img_w = pred_img.shape
            pred_img = MinMaxScaler(feature_range=(0, 1)) \
                            .fit_transform(pred_img.reshape(-1, 1)) \
                            .reshape((img_c, img_h, img_w))
            recon_mask_all = np.einsum('chw->hwc', pred_img).astype('float32')
            ax_plot(image=recon_mask_all, ax=axes[plot_cnt//num_column][plot_cnt%num_column],
                    title=f'rec with decoder no pos')
            plot_cnt += 1

            pred_img = pred_decoder_no_mask[0].cpu().numpy()
            # img_c, img_h, img_w = pred_img.shape
            pred_img = MinMaxScaler(feature_range=(0, 1)) \
                            .fit_transform(pred_img.reshape(-1, 1)) \
                            .reshape((img_c, img_h, img_w))
            recon_mask_all = np.einsum('chw->hwc', pred_img).astype('float32')
            ax_plot(image=recon_mask_all, ax=axes[plot_cnt//num_column][plot_cnt%num_column],
                    title=f'rec with decoder no [mask]')
            plot_cnt += 1

        minmax_surr_loss_idx_mat = MinMaxScaler(feature_range=(0, 1)) \
                            .fit_transform(surr_loss_idx_mat.reshape(-1,1)) \
                            .reshape((model.num_patch_per_row, model.num_patch_per_row))
        # axes[plot_cnt//num_column][plot_cnt%num_column].set_yscale('log')
        # ax_hist(x=minmax_surr_loss_idx_mat.reshape((-1,)), ax=axes[plot_cnt//num_column][plot_cnt%num_column],
        #         title=f'hist: MinMaxScaled rec error map', n_bins=1000)
        # plot_cnt += 1

        t_mat = nn.functional.interpolate(torch.tensor(minmax_surr_loss_idx_mat).unsqueeze(0).unsqueeze(0),
                        scale_factor=(model.patch_size, model.patch_size),
                        mode="nearest")[0][0].cpu().numpy().astype('float32')
        ax_plot(image=t_mat, ax=axes[plot_cnt//num_column][plot_cnt%num_column],
                title=f'MinMaxScaled rec error map\nsum:{minmax_surr_loss_idx_mat.sum():.3e}, min:{minmax_surr_loss_idx_mat.min():.3e}, max: {minmax_surr_loss_idx_mat.max():.3e}')
        plot_cnt += 1

        if args.show_type == 'stat':
            err_list = surr_loss_idx_mat.reshape((-1,))
            axes[plot_cnt//num_column][plot_cnt%num_column].set_yscale('log')
            ax_hist(x=err_list, ax=axes[plot_cnt//num_column][plot_cnt%num_column],
                title=f'hist: original rec error map (0<x<=1)', n_bins=1000)
            plot_cnt += 1
            vals = np.sort(err_list.reshape(-1), axis=-1)  # ascending order
            ax_line_plot(vals, axes[plot_cnt//num_column][plot_cnt%num_column],
                         title=f'orignal rec map, ascending order', marker='x')
            plot_cnt += 1
            cumval = np.cumsum(vals, axis=-1)
            ax_line_plot(cumval, axes[plot_cnt//num_column][plot_cnt%num_column],
                         title=f'cumsum orignal rec map, ascending order', marker='x')
            plot_cnt += 1
            assert np.all(err_list>=0)
            vals /= abs(np.sum(vals, axis=-1))  # similar to attn.softmax(dim=-1)
            ax_line_plot(vals, axes[plot_cnt//num_column][plot_cnt%num_column],
                         title=f'L1 normed orignal rec map, ascending order', marker='x')
            plot_cnt += 1
            cumval = np.cumsum(vals, axis=-1)
            ax_line_plot(cumval, axes[plot_cnt//num_column][plot_cnt%num_column],
                         title=f'cumsum L1 normed orignal rec map, ascending order', marker='x')
            plot_cnt += 1
            val_diff = np.array([0]+[vals[idx+1]-vals[idx] for idx in range(vals.size-1)])
            ax_line_plot(val_diff, axes[plot_cnt//num_column][plot_cnt%num_column],
                         title=f'diff: ascending sorted L1 normed orignal rec map', marker='x')
            plot_cnt += 1
            val_div = np.array([0]+[vals[idx+1]/vals[idx] for idx in range(vals.size-1)])
            ax_line_plot(val_div, axes[plot_cnt//num_column][plot_cnt%num_column],
                         title=f'div: ascending sorted L1 normed orignal rec map', marker='x')
            plot_cnt += 1
            # ax_line_plot(np.log(val_diff), axes[plot_cnt//num_column][plot_cnt%num_column],
            #              title=f'log diff: ascending sorted L1 normed orignal rec map', marker='x')
            # plot_cnt += 1

            for delt_x in [1e-4, 1e-3, 3e-3, 5e-3, 1e-2]:
                ret, argmax_t = get_signal(vals, delt_x=delt_x)
                ax_line_plot(ret, axes[plot_cnt//num_column][plot_cnt%num_column],
                            title=f'signal with delt_x={delt_x}', marker='x')
                plot_cnt += 1

            # err_list_log2 = np.log2(surr_loss_idx_mat).reshape((-1,))
            # ax_hist(x=err_list_log2, ax=axes[plot_cnt//num_column][plot_cnt%num_column],
            #     title=f'hist: log2 original rec error map(0<x<=1)', n_bins=1000, bin_range=(-19, -1))
            # plot_cnt += 1
            # vals = np.sort(err_list_log2.reshape(-1), axis=-1)  # ascending order
            # ax_line_plot(vals, axes[plot_cnt//num_column][plot_cnt%num_column],
            #              title=f'log2 orignal rec map, ascending order')
            # plot_cnt += 1
            # cumval = np.cumsum(vals, axis=-1)
            # ax_line_plot(cumval, axes[plot_cnt//num_column][plot_cnt%num_column],
            #              title=f'cumsum log2 orignal rec map, ascending order')
            # plot_cnt += 1
            # assert np.all(err_list_log2<=0)
            # vals /= abs(np.sum(vals, axis=-1))  # similar to attn.softmax(dim=-1)
            # ax_line_plot(vals, axes[plot_cnt//num_column][plot_cnt%num_column],
            #              title=f'L1 normed log2 orignal rec map, ascending order')
            # plot_cnt += 1
            # cumval = np.cumsum(vals, axis=-1)
            # ax_line_plot(cumval, axes[plot_cnt//num_column][plot_cnt%num_column],
            #              title=f'cumsum L1 normed log2 orignal rec map, ascending order')
            # plot_cnt += 1

            # err_list_log = np.log(surr_loss_idx_mat).reshape((-1,))
            # ax_hist(x=err_list_log, ax=axes[plot_cnt//num_column][plot_cnt%num_column],
            #     title=f'hist: log original rec error map(0<x<=1)', n_bins=1000, bin_range=(-12, -1))
            # plot_cnt += 1
            # vals = np.sort(err_list_log.reshape(-1), axis=-1)  # ascending order
            # ax_line_plot(vals, axes[plot_cnt//num_column][plot_cnt%num_column],
            #              title=f'log orignal rec map, ascending order')
            # plot_cnt += 1
            # cumval = np.cumsum(vals, axis=-1)
            # ax_line_plot(cumval, axes[plot_cnt//num_column][plot_cnt%num_column],
            #              title=f'cumsum log orignal rec map, ascending order')
            # plot_cnt += 1
            # assert np.all(err_list_log<=0)
            # vals /= abs(np.sum(vals, axis=-1))  # similar to attn.softmax(dim=-1)
            # ax_line_plot(vals, axes[plot_cnt//num_column][plot_cnt%num_column],
            #              title=f'L1 normed log orignal rec map, ascending order')
            # plot_cnt += 1
            # cumval = np.cumsum(vals, axis=-1)
            # ax_line_plot(cumval, axes[plot_cnt//num_column][plot_cnt%num_column],
            #              title=f'cumsum L1 normed log orignal rec map, ascending order')
            # plot_cnt += 1
            # val_diff = np.array([0]+[vals[idx+1]-vals[idx] for idx in range(vals.size-1)])
            # ax_line_plot(val_diff, axes[plot_cnt//num_column][plot_cnt%num_column],
            #              title=f'diff: ascending sorted L1 normed log orignal rec map')
            # plot_cnt += 1

            # err_list_log10 = np.log10(surr_loss_idx_mat).reshape((-1,))
            # ax_hist(x=err_list_log10, ax=axes[plot_cnt//num_column][plot_cnt%num_column],
            #     title=f'hist: log10 original rec error map(0<x<=1)', n_bins=1000, bin_range=(-6, 0))
            # plot_cnt += 1
            # vals = np.sort(err_list_log10.reshape(-1), axis=-1)  # ascending order
            # ax_line_plot(vals, axes[plot_cnt//num_column][plot_cnt%num_column],
            #              title=f'log10 orignal rec map, ascending order')
            # plot_cnt += 1
            # cumval = np.cumsum(vals, axis=-1)
            # ax_line_plot(cumval, axes[plot_cnt//num_column][plot_cnt%num_column],
            #              title=f'cumsum log10 orignal rec map, ascending order')
            # plot_cnt += 1
            # assert np.all(err_list_log10<=0)
            # vals /= abs(np.sum(vals, axis=-1))  # similar to attn.softmax(dim=-1)
            # ax_line_plot(vals, axes[plot_cnt//num_column][plot_cnt%num_column],
            #              title=f'L1 normed log10 orignal rec map, ascending order')
            # plot_cnt += 1
            # cumval = np.cumsum(vals, axis=-1)
            # ax_line_plot(cumval, axes[plot_cnt//num_column][plot_cnt%num_column],
            #              title=f'cumsum L1 normed log10 orignal rec map, ascending order')
            # plot_cnt += 1


        # if args.show_type == 'hist':
        #     surr_loss_idx_log_mat = MinMaxScaler(feature_range=(0, 1)) \
        #                     .fit_transform(np.log2((surr_loss_idx_mat+args.log_eps).reshape(-1,1))) \
        #                     .reshape((model.num_patch_per_row, model.num_patch_per_row))
        #     ax_hist(x=surr_loss_idx_log_mat.reshape((-1,)), ax=axes[plot_cnt//num_column][plot_cnt%num_column],
        #             title=f'hist: MinMaxScaled log2 error map')
        #     plot_cnt += 1
        #     surr_loss_idx_log_mat = MinMaxScaler(feature_range=(0, 1)) \
        #                     .fit_transform(np.log10((surr_loss_idx_mat+args.log_eps).reshape(-1,1))) \
        #                     .reshape((model.num_patch_per_row, model.num_patch_per_row))
        #     ax_hist(x=surr_loss_idx_log_mat.reshape((-1,)), ax=axes[plot_cnt//num_column][plot_cnt%num_column],
        #             title=f'hist: MinMaxScaled log10 error map')
        #     plot_cnt += 1

        ori_img = (ori_img * 255).astype(int)
        if args.show_type == 'ratio':
            order = np.argsort(surr_loss_idx_mat.reshape(-1)) # ascending order
            for mask_ratio in [0.95, 0.90, 0.85, 0.80, 0.75, 0.7, 0.65, 0.6]:
                len_keep = int(surr_loss_idx_mat.size * (1-mask_ratio))
                keep_id = order[-len_keep:]
                draw_mask(keep_id=keep_id, model=model, mask_shape=surr_loss_idx_mat.shape,
                        ori_img=ori_img, ax=axes[plot_cnt//num_column][plot_cnt%num_column],
                        title=f'mask_ratio: {mask_ratio}, len_keep: {len_keep}')
                plot_cnt += 1
        elif args.show_type == 'thres':
            vals = np.sort(minmax_surr_loss_idx_mat.reshape(-1), axis=-1)
            order = np.argsort(minmax_surr_loss_idx_mat.reshape(-1)) # ascending order
            max_len_keep = int(model.patch_embed.num_patches * (1 - args.min_mask_ratio))
            # for id_thres, rec_thres in enumerate(np.arange(0.07, 0.28, 0.01)):
            for rec_thres in np.arange(0.055, 0.155, 0.005):  # for seaquest
                len_keep = min(len(np.where(vals > rec_thres)[-1]), max_len_keep)
                keep_id = order[-len_keep:]
                draw_mask(keep_id=keep_id, model=model, mask_shape=surr_loss_idx_mat.shape,
                          ori_img=ori_img, ax=axes[plot_cnt//num_column][plot_cnt%num_column],
                          title=f'rec_thres: {rec_thres:.3f}, len_keep: {len_keep}')
                plot_cnt += 1
        elif args.show_type == 'weight':
            max_len_keep = int(model.patch_embed.num_patches * (1 - args.min_mask_ratio))
            idx_mat = surr_loss_idx_mat
            thres_list = np.arange(0.90, 0.991, 0.01)
            vals = np.sort(idx_mat.reshape(-1), axis=-1)
            idx = np.argsort(idx_mat.reshape(-1)) # ascending order
            vals /= np.sum(vals, axis=-1)  # similar to attn.softmax(dim=-1)
            cumval = np.cumsum(vals, axis=-1)
            for threshold in thres_list:
                th_attn = cumval > (1 - threshold)
                len_keep = min(max(1, int(np.sum(th_attn))), max_len_keep)
                keep_id = idx[-len_keep:]
                draw_mask(keep_id=keep_id, model=model, mask_shape=idx_mat.shape,
                        ori_img=ori_img, ax=axes[plot_cnt//num_column][plot_cnt%num_column],
                        title=f'{mat}, weight: {threshold:.3f}, len_keep: {len_keep}')
                plot_cnt += 1
        elif args.show_type == 'dynamic_thres_deltx':
            err_list = surr_loss_idx_mat.reshape((-1,))
            order = np.argsort(err_list.reshape(-1)) # ascending order
            vals = np.sort(err_list.reshape(-1), axis=-1)  # ascending order
            vals /= abs(np.sum(vals, axis=-1))  # similar to attn.softmax(dim=-1)
            cumval = np.cumsum(vals, axis=-1)
            ax_line_plot(vals, axes[plot_cnt//num_column][plot_cnt%num_column],
                         title=f'L1 normed orignal rec map, ascending order', marker='x')
            plot_cnt += 1
            ax_line_plot(cumval, axes[plot_cnt//num_column][plot_cnt%num_column],
                         title=f'cumsum L1 normed orignal rec map, ascending order', marker='x')
            plot_cnt += 1

            arr = vals
            delt_x_ls = [1e-4, 1e-3, 3e-3, 5e-3, 1e-2]
            for delt_x in delt_x_ls:
                ret, argmax_t = get_signal(arr, delt_x=delt_x)
                ax_line_plot(ret, axes[plot_cnt//num_column][plot_cnt%num_column],
                            title=f'cumval: signal with delt_x={delt_x}', marker='x',
                            label='w/o ma')
                len_keep = model.patch_embed.num_patches - argmax_t  # at least keep 2 patches
                keep_id = order[-len_keep:]
                draw_mask(keep_id=keep_id, model=model, mask_shape=surr_loss_idx_mat.shape,
                        ori_img=ori_img, ax=axes[(plot_cnt//num_column)+1][plot_cnt%num_column],
                        title=f'cumval: $delt$x = {delt_x}, len_keep: {len_keep}')

                for idx, ma_window in enumerate([3,5,7,9]):
                    ret_ma = np.convolve(ret, np.ones(ma_window), 'same') / ma_window
                    argmax_ma_t = np.argmax(ret_ma)

                    ax_line_plot(ret_ma, axes[(plot_cnt//num_column)][plot_cnt%num_column],
                                title=None, marker='x', label=f'ma={ma_window}')
                    len_keep = model.patch_embed.num_patches - argmax_ma_t  # at least keep 2 patches
                    keep_id = order[-len_keep:]
                    draw_mask(keep_id=keep_id, model=model, mask_shape=surr_loss_idx_mat.shape,
                            ori_img=ori_img, ax=axes[(plot_cnt//num_column)+2+idx][plot_cnt%num_column],
                            title=f'cumval-ma={ma_window}: $delt$x = {delt_x}, len_keep: {len_keep}')
                
                axes[(plot_cnt//num_column)][plot_cnt%num_column].legend()
                plot_cnt += 1
        elif args.show_type == 'dynamic_thres_k':
            err_list = surr_loss_idx_mat.reshape((-1,))
            order = np.argsort(err_list.reshape(-1)) # ascending order
            vals = np.sort(err_list.reshape(-1), axis=-1)  # ascending order
            vals /= abs(np.sum(vals, axis=-1))  # similar to attn.softmax(dim=-1)
            cumval = np.cumsum(vals, axis=-1)
            ax_line_plot(vals, axes[plot_cnt//num_column][plot_cnt%num_column],
                         title=f'L1 normed orignal rec map, ascending order', marker='x')
            plot_cnt += 1
            ax_line_plot(cumval, axes[plot_cnt//num_column][plot_cnt%num_column],
                         title=f'cumsum L1 normed orignal rec map, ascending order', marker='x')
            plot_cnt += 1
            val_div = np.array([0]+[vals[idx+1]/vals[idx] for idx in range(vals.size-1)])
            ax_line_plot(val_div, axes[plot_cnt//num_column][plot_cnt%num_column],
                         title=f'div: ascending sorted L1 normed orignal rec map', marker='x')
            plot_cnt += 1

            # div_sort = np.argsort(val_div, axis=-1)

            # NOTE: len_extra & rank are not good solutions
            # len_keep = int(surr_loss_idx_mat.size - div_sort[-1])
            # for len_extra in range(0, 11, 2):
            #     keep_id = order[-len_keep-len_extra:]
            #     draw_mask(keep_id=keep_id, model=model, mask_shape=surr_loss_idx_mat.shape,
            #             ori_img=ori_img, ax=axes[plot_cnt//num_column][plot_cnt%num_column],
            #             title=f'div_max + extra {len_extra}, len_keep: {len(keep_id)}')
            #     plot_cnt += 1

            # for rank in range(1, 3):
            #     len_keep = int(surr_loss_idx_mat.size - div_sort[-rank-1])
            #     keep_id = order[-len_keep:]
            #     draw_mask(keep_id=keep_id, model=model, mask_shape=surr_loss_idx_mat.shape,
            #             ori_img=ori_img, ax=axes[plot_cnt//num_column][plot_cnt%num_column],
            #             title=f'div argmax rank {rank+1}, len_keep: {len(keep_id)}')
            #     plot_cnt += 1

            for seg in range(10, 34, 2):
                # TODO: here can optimize code: only calculate the results for patch [72-144]
                val_div = np.array([0]*seg+[(cumval[idx+seg]-cumval[idx])/(cumval[idx]-cumval[idx-seg]) for idx in range(seg, vals.size-seg)]+[0]*seg)
                
                ax_line_plot(cumval, axes[plot_cnt//num_column][plot_cnt%num_column],
                            title=f'cumsum L1 normed & div-seg-{seg}', marker='x')
                ax_line_plot(val_div, axes[plot_cnt//num_column][plot_cnt%num_column],
                            title=f'cumsum L1 normed & div-seg-{seg}', marker='.', is_twin=True, color='r')
                div_max = np.argmax(val_div, axis=-1)
                axes[plot_cnt//num_column][plot_cnt%num_column].axvline(x=div_max, color='g')
                plot_cnt += 1

                # NOTE: bad performance with the following modification
                # for i in range(seg-1):
                #     val_div[vals.size-seg+i] = (cumval[-1]-cumval[vals.size-seg+i])/(cumval[vals.size-seg+i]-cumval[vals.size-2*seg+2*i+1])
                div_max = np.argmax(val_div, axis=-1)
                len_keep = int(surr_loss_idx_mat.size - div_max)
                keep_id = order[-len_keep:]
                draw_mask(keep_id=keep_id, model=model, mask_shape=surr_loss_idx_mat.shape,
                        ori_img=ori_img, ax=axes[plot_cnt//num_column][plot_cnt%num_column],
                        title=f'div-seg-{seg} argmax, len_keep: {len(keep_id)}')
                plot_cnt += 1

        elif args.show_type == 'dynamic_thres_45':
            err_list = surr_loss_idx_mat.reshape((-1,))
            order = np.argsort(err_list.reshape(-1)) # ascending order
            vals = np.sort(err_list.reshape(-1), axis=-1)  # ascending order
            vals /= abs(np.sum(vals, axis=-1))  # similar to attn.softmax(dim=-1)
            cumval = np.cumsum(vals, axis=-1)
            ax_line_plot(vals, axes[plot_cnt//num_column][plot_cnt%num_column],
                         title=f'L1 normed orignal rec map, ascending order', marker='x')
            plot_cnt += 1
            ax_line_plot(cumval, axes[plot_cnt//num_column][plot_cnt%num_column],
                         title=f'cumsum L1 normed orignal rec map, ascending order', marker='x')
            for seg in [2, 4]:
                half_seg = seg//2
                cumval_diff = np.array([0]*half_seg+
                                       [(cumval[idx+half_seg]-cumval[idx-half_seg]) \
                                        for idx in range(half_seg, vals.size-half_seg)])
                for scale in [1, 2, 3, 4]:
                    ax_line_plot(cumval, axes[plot_cnt//num_column][plot_cnt%num_column],
                                title=f'cumsum L1 normed & diff-seg-{seg}-scale-{scale}', marker='x')
                    ax_line_plot(cumval_diff, axes[plot_cnt//num_column][plot_cnt%num_column],
                                title=f'cumsum L1 normed & diff-seg-{seg}-scale-{scale}', marker='.', is_twin=True, color='r')
                    selected_idx = np.argmin(np.abs(cumval_diff-(1.0*scale)/vals.size), axis=-1)
                    axes[plot_cnt//num_column][plot_cnt%num_column].axvline(x=selected_idx, color='g')
                    plot_cnt += 1

                    len_keep = int(surr_loss_idx_mat.size - selected_idx)
                    keep_id = order[-len_keep:]
                    draw_mask(keep_id=keep_id, model=model, mask_shape=surr_loss_idx_mat.shape,
                            ori_img=ori_img, ax=axes[plot_cnt//num_column][plot_cnt%num_column],
                            title=f'diff-seg-{seg}-scale-{scale}, len_keep: {len(keep_id)}')
                    plot_cnt += 1
            plot_cnt += 1


        elif args.show_type == 'all':
            err_list = surr_loss_idx_mat.reshape((-1,))
            axes[plot_cnt//num_column][plot_cnt%num_column].set_yscale('log')
            ax_hist(x=err_list, ax=axes[plot_cnt//num_column][plot_cnt%num_column],
                title=f'hist: original rec error map (0<x<=1)', n_bins=1000)
            plot_cnt += 1
            vals = np.sort(err_list.reshape(-1), axis=-1)  # ascending order
            ax_line_plot(vals, axes[plot_cnt//num_column][plot_cnt%num_column],
                         title=f'orignal rec map, ascending order')
            plot_cnt += 1
            cumval = np.cumsum(vals, axis=-1)
            ax_line_plot(cumval, axes[plot_cnt//num_column][plot_cnt%num_column],
                         title=f'cumsum orignal rec map, ascending order')
            plot_cnt += 1
            assert np.all(err_list>=0)
            vals /= abs(np.sum(vals, axis=-1))  # similar to attn.softmax(dim=-1)
            ax_line_plot(vals, axes[plot_cnt//num_column][plot_cnt%num_column],
                         title=f'L1 normed orignal rec map, ascending order')
            plot_cnt += 1
            cumval = np.cumsum(vals, axis=-1)
            ax_line_plot(cumval, axes[plot_cnt//num_column][plot_cnt%num_column],
                         title=f'cumsum L1 normed orignal rec map, ascending order')
            plot_cnt += 1

            order = np.argsort(surr_loss_idx_mat.reshape(-1)) # ascending order
            # mask_ratio
            for mask_ratio in [0.95, 0.90, 0.85, 0.80, 0.75, 0.7, 0.65, 0.6]:
                len_keep = int(surr_loss_idx_mat.size * (1-mask_ratio))
                keep_id = order[-len_keep:]
                draw_mask(keep_id=keep_id, model=model, mask_shape=surr_loss_idx_mat.shape,
                        ori_img=ori_img, ax=axes[plot_cnt//num_column][plot_cnt%num_column],
                        title=f'mask_ratio: {mask_ratio}')
                plot_cnt += 1

            # weight
            thres_list = np.arange(0.89, 0.961, 0.01)
            err_list = surr_loss_idx_mat.reshape((-1,))
            vals = np.sort(err_list.reshape(-1), axis=-1)  # ascending order
            idx = np.argsort(err_list.reshape(-1)) # ascending order
            vals /= np.sum(vals, axis=-1)  # similar to attn.softmax(dim=-1)
            cumval = np.cumsum(vals, axis=-1)
            max_len_keep = int(model.patch_embed.num_patches * (1 - args.min_mask_ratio))
            for threshold in thres_list:
                th_attn = cumval > (1 - threshold)
                len_keep = min(max(1, int(np.sum(th_attn))), max_len_keep)
                keep_id = idx[-len_keep:]
                draw_mask(keep_id=keep_id, model=model, mask_shape=surr_loss_idx_mat.shape,
                        ori_img=ori_img, ax=axes[plot_cnt//num_column][plot_cnt%num_column],
                        title=f'weight: {threshold:.3f}, len_keep: {len_keep}')
                plot_cnt += 1

            # thres  # TODO: currently we skip this method, it's non-trival to select a suitable threshold
            # for mat in ['original', 'log']:
            #     if mat == 'original':
            #         idx_mat = surr_loss_idx_mat
            #         thres_list = np.arange(0.085, 0.125, 0.005)
            #     elif mat == 'log':
            #         idx_mat = surr_loss_idx_log_mat
            #         thres_list = np.arange(0.725, 0.91, 0.025)
            #     else:
            #         raise NotImplementedError

            #     vals = np.sort(idx_mat.reshape(-1), axis=-1)
            #     order = np.argsort(idx_mat.reshape(-1)) # ascending order
            #     max_len_keep = int(model.patch_embed.num_patches * (1 - args.min_mask_ratio))
            #     # for id_thres, rec_thres in enumerate(np.arange(0.07, 0.28, 0.01)):
            #     for rec_thres in thres_list:  # for seaquest
            #         len_keep = min(len(np.where(vals > rec_thres)[-1]), max_len_keep)
            #         keep_id = order[-len_keep:]
            #         draw_mask(keep_id=keep_id, model=model, mask_shape=surr_loss_idx_mat.shape,
            #                 ori_img=ori_img, ax=axes[plot_cnt//num_column][plot_cnt%num_column],
            #                 title=f'{mat}, rec_thres: {rec_thres:.3f}, len_keep: {len_keep}')
            #         plot_cnt += 1
        elif args.show_type in ['stat', 'bkg']:
            pass
        else:
            raise NotImplementedError

        fig.tight_layout()
        fig.savefig(fname=os.path.join(img_save_path, f"frame{img_cnt}.png"))
        # pdb.set_trace()


def main(args):
    # # if gpu is not available, use cpu
    if not torch.cuda.is_available():
        args.device = 'cpu'
    device = torch.device(args.device)

    # fix the seed for reproducibility
    # seed = args.seed + misc.get_rank()
    # NOTE: setting different seeds for exp
    args.seed = args.seed if args.seed is not None else random.randint(0, 1e6)
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    if args.quick_log:
        args.output_dir = './'
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        args.output_dir = f'{args.output_dir}/{timestamp}'
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    img_save_path = os.path.join(args.output_dir, 'eval_reconstruct')
    os.makedirs(img_save_path, exist_ok=True)
    
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    cudnn.benchmark = True

    ckpt = torch.load(args.mae_pretrained_path, map_location='cpu')  # NOTE: the map_location choice may cost a lot of time
    # define the model
    model = MaskedAutoencoderViT(
        patch_size=ckpt['args'].patch_size, embed_dim=ckpt['args'].embed_dim, depth=ckpt['args'].depth,
        num_heads=ckpt['args'].num_heads, decoder_embed_dim=ckpt['args'].decoder_embed_dim,
        decoder_depth=ckpt['args'].decoder_depth, decoder_num_heads=ckpt['args'].decoder_num_heads,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), in_chans=ckpt['args'].in_chans,
        img_size=ckpt['args'].image_size, norm_pix_loss=ckpt['args'].norm_pix_loss, debug=False,
    )
    model.load_trained_mae(ckpt['model'])
    for p in model.parameters():
        p.requires_grad = False
    model.to(device)

    model_without_ddp = model  # without distributed data parallel
    mae_misc.model_summary(model_without_ddp)

    if args.data_type == 'random':
        dataset_eval = AtariNormalDataset(
            env_name=ckpt['args'].env_name, total_data=args.num_eval_data,
            img_size=ckpt['args'].image_size, in_chans=ckpt['args'].in_chans, seed=args.seed,
            transform=None, save_obs=None, transpose=True)
    elif args.data_type == 'atarizoo':
        dataset_eval = AtariOfflineDataset(
            env_name=ckpt['args'].env_name, img_size=ckpt['args'].image_size, in_chans=ckpt['args'].in_chans,
            split='test', total_data=args.num_eval_data,
        )
    else:
        raise NotImplementedError

    sampler_eval = torch.utils.data.SequentialSampler(dataset_eval)

    data_loader_eval = torch.utils.data.DataLoader(
        dataset_eval, sampler=sampler_eval,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    print(f'dataset_eval: {dataset_eval}')

    evaluate(data_loader_eval, model, device, img_save_path, image_size=ckpt['args'].image_size, args=args)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
