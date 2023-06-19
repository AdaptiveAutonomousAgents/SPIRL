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
    parser.add_argument('--data_type', default='random', choices=['random', 'atarizoo'],
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

    parser.add_argument('--show_type', type=str, help='flag for special ways to load ckpt',
                        choices=['paper_overall_45'])
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
        
        if args.show_type == 'paper_overall_45':
            # num_row = 1
            # num_column = 5
            weight_column = 5
            height_row = 5
        else:
            raise NotImplementedError

        # fig, axes = plt.subplots(num_row, num_column, figsize=(num_column*weight_column, num_row*height_row))  # figsize: (weight, height)

        with torch.cuda.amp.autocast():
            surr_pred_map = torch.zeros((1, model.num_patch_per_row**2, model.patch_size**2*3))
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
                x_pos = idx//model.num_patch_per_row
                y_pos = idx%model.num_patch_per_row
                surr_pred_map[0][idx] = surr_pred[0][idx]
            surr_loss_idx_mat = surr_loss_idx_mat.reshape((model.num_patch_per_row, model.num_patch_per_row))
            surr_pred_map = model.unpatchify(surr_pred_map)[0]

        num_samples = samples[0].shape[0]
        assert num_samples == 1 # to get correct loss (normaly the loss will be averaged over a whoel batch, thus here we constraint the batch_size is 1)
        img_cnt += 1
        img_save_dir = img_save_path

        plot_cnt = 0  # have ploted 0 image
        
        if args.show_type == 'paper_overall_45':
            os.makedirs(os.path.join(img_save_path, f'{img_cnt:3d}'), exist_ok=True)
            # original imaeg
            ori_img = np.einsum('chw->hwc', samples[0][0].cpu().numpy())
            img_h, img_w, img_c = ori_img.shape

            plt.figure()
            ax = plt.gca()
            ax_plot(image=ori_img, ax=ax)
            plt.margins(0.0)
            plt.savefig(os.path.join(img_save_path, f'{img_cnt:3d}', 'original.png'),
                        bbox_inches='tight', pad_inches=0)

            # image predicted by surrounding patches
             # minmax scale over axis=0
            surr_pred_map = np.einsum('chw->hwc', surr_pred_map.cpu().numpy())
            surr_pred_map = MinMaxScaler(feature_range=(0, 1)) \
                            .fit_transform(surr_pred_map.reshape(-1, img_c)) \
                            .reshape((img_h, img_w, img_c))
            plt.figure()
            ax = plt.gca()
            ax_plot(image=surr_pred_map, ax=ax)
            plt.savefig(os.path.join(img_save_path, f'{img_cnt:3d}', 'predicted.png'),
                        bbox_inches='tight', pad_inches=0)

            # reconstruction error map
            minmax_surr_loss_idx_mat = MinMaxScaler(feature_range=(0, 1)) \
                                .fit_transform(surr_loss_idx_mat.reshape(-1,1)) \
                                .reshape((model.num_patch_per_row, model.num_patch_per_row))

            t_mat = nn.functional.interpolate(torch.tensor(minmax_surr_loss_idx_mat).unsqueeze(0).unsqueeze(0),
                            scale_factor=(model.patch_size, model.patch_size),
                            mode="nearest")[0][0].cpu().numpy().astype('float32')

            plt.figure()
            ax = plt.gca()
            ax_plot(image=t_mat, ax=ax)
            plt.savefig(os.path.join(img_save_path, f'{img_cnt:3d}', 'error_map.png'),
                        bbox_inches='tight', pad_inches=0)

            # cumsum and key patches selected by thres45
            err_list = surr_loss_idx_mat.reshape((-1,))
            order = np.argsort(err_list.reshape(-1)) # ascending order
            vals = np.sort(err_list.reshape(-1), axis=-1)  # ascending order
            vals /= abs(np.sum(vals, axis=-1))  # similar to attn.softmax(dim=-1)
            cumval = np.cumsum(vals, axis=-1)
            seg = 2
            half_seg = seg//2
            cumval_diff = np.array([0]*half_seg+
                                    [(cumval[idx+half_seg]-cumval[idx-half_seg]) \
                                    for idx in range(half_seg, vals.size-half_seg)])
            scale = 1
            plt.figure(figsize=(6,6))
            ax = plt.gca()
            ax_line_plot(cumval, ax, lw=2.5)
            # ax_line_plot(cumval_diff, axes[plot_cnt%num_column],
            #             title=f'cumsum L1 normed & diff-seg-{seg}-scale-{scale}', marker='.', is_twin=True, color='r')
            selected_idx = np.argmin(np.abs(cumval_diff-(1.0*scale)/vals.size), axis=-1)
            ax.axvline(x=selected_idx, color='r', linestyle='dashed', lw=2.5)
            plt.yticks(size=15,weight='bold')
            plt.xticks(size=15,weight='bold')
            plt.savefig(os.path.join(img_save_path, f'{img_cnt:3d}', 'cumsum.png'),
                        bbox_inches='tight', pad_inches=0)

            len_keep = int(surr_loss_idx_mat.size - selected_idx)
            keep_id = order[-len_keep:]
            plt.figure()
            ax = plt.gca()
            draw_mask(keep_id=keep_id, model=model, mask_shape=surr_loss_idx_mat.shape,
                    ori_img=(ori_img * 255).astype(int), ax=ax)
            plt.savefig(os.path.join(img_save_path, f'{img_cnt:3d}', f'dynamic45_{len_keep}.png'),
                        bbox_inches='tight', pad_inches=0)

            # reverse cumsum and key patches selected by thres45
            order = order[::-1] # descending order
            vals = vals[::-1]  # ascending order
            cumval = np.cumsum(vals, axis=-1)
            # cumsum and key patches selected by thres45
            plt.figure(figsize=(6,6))
            ax = plt.gca()
            ax_line_plot(cumval, ax, lw=2.5)
            # ax_line_plot(cumval_diff, axes[plot_cnt%num_column],
            #             title=f'cumsum L1 normed & diff-seg-{seg}-scale-{scale}', marker='.', is_twin=True, color='r')
            ax.axvline(x=len_keep, color='r', linestyle='dashed', lw=2.5)
            plt.yticks(size=15,weight='bold')
            plt.xticks(size=15,weight='bold')
            plt.savefig(os.path.join(img_save_path, f'{img_cnt:3d}', 'rev_cumsum.png'),
                        bbox_inches='tight', pad_inches=0)

            # mask ratio selected key patches
            order = np.argsort(surr_loss_idx_mat.reshape(-1)) # ascending order
            for mask_ratio in [0.90, 0.85, 0.80, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5]:
                len_keep = int(surr_loss_idx_mat.size * (1-mask_ratio))
                keep_id = order[-len_keep:]
                plt.figure()
                ax = plt.gca()
                draw_mask(keep_id=keep_id, model=model, mask_shape=surr_loss_idx_mat.shape,
                        ori_img=(ori_img * 255).astype(int), ax=ax)
                plt.savefig(os.path.join(img_save_path, f'{img_cnt:3d}',
                                         f'mask_ratio_{mask_ratio}_len_keep_{len_keep}.png'),
                        bbox_inches='tight', pad_inches=0)
        
            # background check
            with torch.cuda.amp.autocast():
                pred_decoder, pred_decoder_no_pos, pred_decoder_no_mask = model.pred_decoder()
                pred_decoder = model.unpatchify(pred_decoder)
                pred_decoder_no_pos = model.unpatchify(pred_decoder_no_pos)
                pred_decoder_no_mask = model.unpatchify(pred_decoder_no_mask)

            pred_img = pred_decoder[0].cpu().numpy()
            recon_mask_all = np.einsum('chw->hwc', pred_img).astype('float32')
            recon_mask_all = MinMaxScaler(feature_range=(0, 1)) \
                            .fit_transform(recon_mask_all.reshape(-1, img_c)) \
                            .reshape((img_h, img_w, img_c))
            plt.figure()
            ax = plt.gca()
            ax_plot(image=recon_mask_all, ax=ax)
            plt.savefig(os.path.join(img_save_path, f'{img_cnt:3d}',
                                     f'pred_decoder_mask_and_pos.png'),
                        bbox_inches='tight', pad_inches=0)

            pred_img = pred_decoder_no_pos[0].cpu().numpy()
            recon_mask_all = np.einsum('chw->hwc', pred_img).astype('float32')
            recon_mask_all = MinMaxScaler(feature_range=(0, 1)) \
                            .fit_transform(recon_mask_all.reshape(-1, img_c)) \
                            .reshape((img_h, img_w, img_c))
            plt.figure()
            ax = plt.gca()
            ax_plot(image=recon_mask_all, ax=ax)
            plt.savefig(os.path.join(img_save_path, f'{img_cnt:3d}',
                                     f'pred_decoder_no_pos.png'),
                        bbox_inches='tight', pad_inches=0)

            pred_img = pred_decoder_no_mask[0].cpu().numpy()
            recon_mask_all = np.einsum('chw->hwc', pred_img).astype('float32')
            recon_mask_all = MinMaxScaler(feature_range=(0, 1)) \
                            .fit_transform(recon_mask_all.reshape(-1, img_c)) \
                            .reshape((img_h, img_w, img_c))
            plt.figure()
            ax = plt.gca()
            ax_plot(image=recon_mask_all, ax=ax)
            plt.savefig(os.path.join(img_save_path, f'{img_cnt:3d}',
                                     f'pred_decoder_no_mask.png'),
                        bbox_inches='tight', pad_inches=0)
        else:
            raise NotImplementedError

        # fig.tight_layout()
        # fig.savefig(fname=os.path.join(img_save_path, f"frame{img_cnt}.png"))
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
