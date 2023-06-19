# Based on: mae official implementation
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import time
import pdb
import random
import math
import cv2
from pathlib import Path
from functools import partial
from typing import Iterable
from sklearn.preprocessing import MinMaxScaler
from spirl.rainbow_utils import count_parameters
from spirl.vit_policy import VisualAttention

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.utils import save_image

import timm
# assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

from spirl.mae_misc import NativeScalerWithGradNormCount as NativeScaler

from spirl import models_mae, mae_misc
from spirl.rainbow_utils import str2bool, ARG2ALE_GAME_NAME_MAP
from spirl.draw_utils import cv2_draw_grid, mark_img_loss_gray
from spirl.pretrain_dataset import AtariNormalDataset, AtariOfflineDataset, AtariKeypointDataset
from spirl.models_mae import MaskedAutoencoderViT
import spirl.lr_sched as lr_sched


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_base_patch16', type=str,
                        choices=['mae_vit_base_patch16', 'mae_vit_large_patch16', 'mae_vit_huge_patch14', 'custom'],
                        help='Name of model to train')

    # NOTE: in timm, for ImageNet dataset:
    #       in this page (https://pytorch.org/vision/stable/models.html), it says that
    #       "All pre-trained models expect input images normalized in the same way, 
    #       i.e. mini-batches of 3-channel RGB images of shape (3 x H x W),
    #       where H and W are expected to be at least 224. 
    #       The images have to be loaded in to a range of [0, 1] 
    #       and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]
    parser.add_argument('--image_size', default=96, type=int,  # NOTE: img_size=224 is consistent with the default img_size in timm.models.layer.patch_embed.PatchEmbed 
                        help='images input size to ViT')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', default=False, type=str2bool,
                        help='Use (per-patch) normalized pixels as targets for computing loss')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--wd_layer_norm', default=True, type=str2bool, help="If True, also add weight_decay for layer_norm")

    parser.add_argument('--lr', type=float, default=None,
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3,
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0.,
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='epochs to warmup LR')

    # -------- params related to env and keypoints --------
    parser.add_argument("--env_name", default="seaquest", type=str, help="environment ID",
        choices=['msPacman', 'frostbite', 'seaquest', 'battleZone'])
    parser.add_argument('--num_training_data', default=4000, type=int, help="total number of training data")
    parser.add_argument('--num_eval_data', default=200, type=int, help="total number of training data")
    parser.add_argument("--kp_ext_loss_weight", default=0.0, type=float, help="if use_reverse_kp is True, weight more loss for patches that contains keypoints")
    parser.add_argument("--kp_ext_loss_multi", default=False, type=str2bool,
        help="if different keypoints shown in the same patch, the extra loss weight for that patch will be number_keypoints_in_that_patch * kp_ext_loss_weight  ")
    
    parser.add_argument("--use_kp", default=False, type=str2bool, help="will prrovide kp info in dataset")
    parser.add_argument("--use_reverse_kp", default=False, type=str2bool, help="if use_kp is True, mask patches that include keypoints")
    parser.add_argument("--kp_around", default=True, type=str2bool, help="if use_kp, then kp_around is True means also mark surrounding patches of keypoint patches")
    parser.add_argument("--loss_all", default=True, type=str2bool, help="backward loss for both masked and unmaksed patches")
    parser.add_argument("--use_aug", default=False, type=str2bool, help="use data augmentation before training")
    parser.add_argument("--debug", default=False, type=str2bool, help="set True to show more logs")
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--save_obs', default=False, type=str2bool, help="if True, save collected training data in this path")
    parser.add_argument('--log_tb', default=True, type=str2bool, help="log tensorboard if True, otherwise no log saved")
    parser.add_argument('--eval_epoch', default=10, type=int, help="interval of epochs to save model checkpoints")
    parser.add_argument('--eval_kps', default=False, type=str2bool, help="if True, evaluation based on keypoint patches")
    parser.add_argument('--save_checkpoint', default=True, type=str2bool, help="save checkpoint every eval_epoch")
    parser.add_argument('--save_exclude_decoder', default=True, type=str2bool, help="don't save decoder part in checkpoints")
    parser.add_argument('--save_last_checkpoint_only', default=False, type=str2bool, help="save checkpoint only at the last epoch")
    parser.add_argument('--output_dir', default='./log/MAE_pretrain', type=str, help='path where to save log & ckpts, empty for no saving')
    parser.add_argument('--quick_log', default=False, type=str2bool, help='use it for lhpo to shorten log path! True if want to save log in the current path directly')
    # NOTE: to save training time, only save best model every save_checkpoint_epoch!
    parser.add_argument('--visual_eval', default=True, type=str2bool, help="save visualization on validation dataset using the best model during traing")
    parser.add_argument('--data_type', default='random', choices=['random', 'atarizoo'],
                        help='type for data source to train and evaluate mae.')
    # --------------- params related to mae ---------------
    parser.add_argument('--in_chans', default=3, type=int)
    parser.add_argument('--patch_size', default=8, type=int)
    parser.add_argument('--embed_dim', default=64, type=int)
    parser.add_argument('--depth', default=3, type=int)
    parser.add_argument('--num_heads', default=4, type=int)
    parser.add_argument('--decoder_embed_dim', default=128, type=int)
    parser.add_argument('--decoder_depth', default=3, type=int)
    parser.add_argument('--decoder_num_heads', default=8, type=int)
    # -----------------------------------------------------
    
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing, for torch.device(args.device)')
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int,
                        help='start epoch')
    parser.add_argument('--num_workers', default=8, type=int,
                        help=' how many subprocesses to use for data loading. 0 means that the data will be loaded in the main process.')
    parser.add_argument('--pin_mem', default=True, type=str2bool,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')

    # # distributed training parameters
    parser.add_argument('--distributed', default=False, type=str2bool)

    # ------------- generate statistic for pre-training dataset
    parser.add_argument('--statistic_salient', default=False, type=str2bool,
                        help="Generate statistic for ptr-training dataset.")
    parser.add_argument("--mae_pretrained_path", type=str,
                        help="path to load pretrained mae; if None, won't load ckpt for ViT feature extractor")
    parser.add_argument("--rec_dynamic_45_seg", type=int, default=2,  # default should be 2 (= consecutive points)
                        help='if rec_dynamic_45_seg is not None, for vitRec, select \
                        key patches based on dynamic threshold (45 degree, segments)')
    parser.add_argument("--rec_dynamic_45_xscale", type=int, default=1,  # default should be 1
                        help='if rec_dynamic_45_seg is not None, delt x =xscale/#patches')


    return parser

@torch.no_grad()
def evaluate(data_loader, model, device, epoch, img_save_path):
    img_save_path = os.path.join(img_save_path, str(epoch))
    os.makedirs(img_save_path, exist_ok=True)

    metric_logger = mae_misc.MetricLogger(delimiter="  ")
    header = 'Test:'
    # switch to evaluation mode
    model.eval()

    img_cnt = 0
    for samples in metric_logger.log_every(data_loader, 10, header):
        if len(samples) == 2:  # evaluate with kps
            (img_ls, kp_pos_ls) = samples
            img_ls = img_ls.to(device, non_blocking=True) if img_ls is not None else img_ls
            kp_pos_ls = kp_pos_ls.to(device, non_blocking=True) if kp_pos_ls is not None else kp_pos_ls
            samples = (img_ls, kp_pos_ls)
 
            # compute output
            with torch.cuda.amp.autocast():
                # # reconstruct use same ratio with training, and do not care about keypoints
                # loss_same_train_nokp, pred_same_train_nokp, mask = model((samples[0], None), mask_ratio=args.mask_ratio)
                # pred_same_train_nokp = model.unpatchify(pred_same_train_nokp)

                # reconstruct use same ratio with training, and select keypoints (kp_around will determines if provide surrounding patches)
                # len_keep at least equals to num_keypoints (including surroundings)
                loss_same_train_kp, pred_same_train_kp, mask = model(samples, mask_ratio=args.mask_ratio, loss_all=args.loss_all)
                pred_same_train_kp = model.unpatchify(pred_same_train_kp)

                # reconstruct with only cls token
                loss_mask_all, pred_mask_all, mask = model((samples[0], None), mask_ratio=1, loss_all=args.loss_all)
                pred_mask_all = model.unpatchify(pred_mask_all)

            for recon_same_train_kp, recon_mask_all, ori_img in zip(pred_same_train_kp, pred_mask_all, samples[0]):
                save_image(torch.cat((ori_img, recon_same_train_kp, recon_mask_all), 2), os.path.join(img_save_path, f'{img_cnt}.png'))
                img_cnt += 1

            metric_logger.update(loss_same_train_kp=loss_same_train_kp.item())
            metric_logger.update(loss_mask_all=loss_mask_all.item())
        
        else:  # eval with each patches
            img_ls = samples
            img_ls = img_ls.to(device, non_blocking=True) if img_ls is not None else img_ls
            kp_pos_ls = None
            samples = (img_ls, kp_pos_ls)

            # compute output
            with torch.cuda.amp.autocast():
                # reconstruct use same ratio with training, and do not care about keypoints
                loss_same_train_nokp, pred_same_train_nokp, mask = model(
                    (samples[0], None),
                    mask_ratio=args.mask_ratio if not (args.use_kp and args.mask_ratio>1) else 0.75,
                    loss_all=args.loss_all)
                pred_same_train_nokp = model.unpatchify(pred_same_train_nokp)

                # reconstruct with only cls token
                loss_mask_all, pred_mask_all, mask = model(
                    (samples[0], None), mask_ratio=1, loss_all=args.loss_all)
                pred_mask_all = model.unpatchify(pred_mask_all)

                # reconstruct with only one patch and cls token
                single_loss_all_ls = []
                single_loss_idx_ls = []
                single_pred_ls = []

                surr_loss_all_ls = []
                surr_loss_idx_ls = []
                surr_pred_ls = []
                single_loss_idx_mat = np.zeros(model.patch_embed.num_patches)
                single_loss_all_mat = np.zeros(model.patch_embed.num_patches)
                surr_loss_idx_mat = np.zeros(model.patch_embed.num_patches)
                surr_loss_all_mat = np.zeros(model.patch_embed.num_patches)
                for idx in range(model.patch_embed.num_patches):
                    # keep each single patch, calculate loss for that single patch
                    single_loss_all, single_loss_idx, single_pred, _ = model(
                        (samples[0], None), unmask_idx=idx, loss_all=args.loss_all)
                    single_pred = model.unpatchify(single_pred)
                    single_loss_all_mat[idx] = single_loss_all.item()
                    single_loss_idx_mat[idx] = single_loss_idx.item()
                    
                    single_loss_all_ls.append(single_loss_all.item())
                    single_loss_idx_ls.append(single_loss_idx.item())
                    single_pred_ls.append(single_pred)

                    # for each single patch, keep its surrounding patchs and calculate loss for the center patch
                    surr_loss_all, surr_loss_idx, surr_pred, _ = model(
                        (samples[0], None), surr_mask_idx=idx, loss_all=args.loss_all)
                    surr_pred = model.unpatchify(surr_pred)
                    surr_loss_all_mat[idx] = surr_loss_all.item()
                    surr_loss_idx_mat[idx] = surr_loss_idx.item()
                    
                    surr_loss_all_ls.append(surr_loss_all.item())
                    surr_loss_idx_ls.append(surr_loss_idx.item())
                    surr_pred_ls.append(surr_pred)

            num_samples = samples[0].shape[0]
            assert num_samples == 1 # to get correct loss (normaly the loss will be averaged over a whoel batch, thus here we constraint the batch_size is 1)
            img_cnt += 1
            img_save_dir = os.path.join(img_save_path, f'{img_cnt}')
            os.makedirs(img_save_dir, exist_ok=True)

            ori_img = samples[0][0]
            recon_same_train = pred_same_train_nokp[0]
            recon_mask_all = pred_mask_all[0]
            
            save_image(torch.cat((ori_img, recon_same_train, recon_mask_all), 2), os.path.join(img_save_dir, \
                f'ori_SameTrain{round(loss_same_train_nokp.item(),3)}_ClsOnly{round(loss_mask_all.item(),3)}.png'))

            # black is 0, white is 255; opencv image is HWC
            # make a grayscale image from 1 channel to 3 channels, just need to repeat the data into 3 channels
            single_loss_all_mat = MinMaxScaler(feature_range=(0, 255)).fit_transform(single_loss_all_mat.reshape(-1,1)).reshape((model.num_patch_per_row, model.num_patch_per_row, 1))
            single_loss_idx_mat = MinMaxScaler(feature_range=(0, 255)).fit_transform(single_loss_idx_mat.reshape(-1,1)).reshape((model.num_patch_per_row, model.num_patch_per_row, 1))
            surr_loss_all_mat = MinMaxScaler(feature_range=(0, 255)).fit_transform(surr_loss_all_mat.reshape(-1,1)).reshape((model.num_patch_per_row, model.num_patch_per_row, 1))
            surr_loss_idx_mat = MinMaxScaler(feature_range=(0, 255)).fit_transform(surr_loss_idx_mat.reshape(-1,1)).reshape((model.num_patch_per_row, model.num_patch_per_row, 1))
            cv2_ori_image = (ori_img*255).cpu().numpy().transpose(1, 2, 0)  # torch image: CHW, cv2 image: HWC
            gray_ori_image_25 = mark_img_loss_gray(cv2_ori_image, surr_loss_idx_mat, mark_ratio=0.25, grid_size=args.patch_size)
            gray_ori_image_20 = mark_img_loss_gray(cv2_ori_image, surr_loss_idx_mat, mark_ratio=0.20, grid_size=args.patch_size)
            gray_ori_image_15 = mark_img_loss_gray(cv2_ori_image, surr_loss_idx_mat, mark_ratio=0.15, grid_size=args.patch_size)
            gray_ori_image_10 = mark_img_loss_gray(cv2_ori_image, surr_loss_idx_mat, mark_ratio=0.10, grid_size=args.patch_size)
            gray_ori_image_5 = mark_img_loss_gray(cv2_ori_image, surr_loss_idx_mat, mark_ratio=0.05, grid_size=args.patch_size)

            # if 1 channel, cv2 resize will drop it; if 3 channel, it will be kept
            single_loss_all_mat_enlarge = np.repeat(cv2.resize(single_loss_all_mat, (args.image_size, args.image_size), interpolation=cv2.INTER_NEAREST)[..., None], repeats=3, axis=-1)
            single_loss_idx_mat_enlarge = np.repeat(cv2.resize(single_loss_idx_mat, (args.image_size, args.image_size), interpolation=cv2.INTER_NEAREST)[..., None], repeats=3, axis=-1)
            surr_loss_all_mat_enlarge = np.repeat(cv2.resize(surr_loss_all_mat, (args.image_size, args.image_size), interpolation=cv2.INTER_NEAREST)[..., None], repeats=3, axis=-1)
            surr_loss_idx_mat_enlarge = np.repeat(cv2.resize(surr_loss_idx_mat, (args.image_size, args.image_size), interpolation=cv2.INTER_NEAREST)[..., None], repeats=3, axis=-1)

            cv2.imwrite(os.path.join(img_save_dir, f'single_all_idx_surr_all_idx.png'), np.concatenate(
                (np.concatenate((cv2_ori_image, single_loss_all_mat_enlarge, single_loss_idx_mat_enlarge,
                 surr_loss_all_mat_enlarge, surr_loss_idx_mat_enlarge), axis=1),
                 np.concatenate((gray_ori_image_25, gray_ori_image_20, gray_ori_image_15,
                 gray_ori_image_10, gray_ori_image_5), axis=1)
                 ),
                 axis=0))

            # cv2.imwrite(os.path.join(img_save_dir, f'grid_single_all_idx_surr_all_idx.png'), np.concatenate(
            #     (cv2_draw_grid(cv2_ori_image, d_row=args.patch_size),
            #      cv2_draw_grid(gray_ori_image, d_row=args.patch_size),
            #      cv2_draw_grid(single_loss_all_mat_enlarge, d_row=args.patch_size), 
            #      cv2_draw_grid(single_loss_idx_mat_enlarge, d_row=args.patch_size), 
            #      cv2_draw_grid(surr_loss_all_mat_enlarge, d_row=args.patch_size), 
            #      cv2_draw_grid(surr_loss_idx_mat_enlarge, d_row=args.patch_size)), 
            #      axis=1))

            # for id_patch in range(model.patch_embed.num_patches):
            for id_patch in set([np.random.randint(0, model.patch_embed.num_patches) for _ in range(3)]):
                single_loss_all = single_loss_all_ls[id_patch]
                single_loss_patch = single_loss_idx_ls[id_patch]
                single_pred = single_pred_ls[id_patch][0]
                surr_loss_all = surr_loss_all_ls[id_patch]
                surr_loss_patch = surr_loss_idx_ls[id_patch]
                surr_pred = surr_pred_ls[id_patch][0]
                save_image(torch.cat((ori_img, single_pred, surr_pred), 2), os.path.join(img_save_dir, \
                    (f'ori_keep_row{id_patch//model.num_patch_per_row}_column{id_patch%model.num_patch_per_row}_'
                     f'singlelossAll{round(single_loss_all,3)}_singlelossPatch{round(single_loss_patch,3)}_'
                     f'surrlossAll{round(surr_loss_all,3)}_surrlossPatch{round(surr_loss_patch,3)}.png')))

            metric_logger.update(loss_same_train_nokp=loss_same_train_nokp.item())
            metric_logger.update(loss_mask_all=loss_mask_all.item())

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def statistic_salient(data_loader, args):
    feature_extractor_ckpt = torch.load(args.mae_pretrained_path, map_location='cpu')  # NOTE: the map_location choice may cost a lot of time
    features_extractor_kwargs = {}
    for key in ['patch_size', 'embed_dim', 'depth', 'num_heads', 'in_chans',
                'decoder_embed_dim', 'decoder_depth', 'decoder_num_heads',
                'norm_pix_loss']:
        features_extractor_kwargs[key] = eval(f"feature_extractor_ckpt['args'].{key}") # use pretrained model's setting
    features_extractor_kwargs['kp'] = 'rec'
    features_extractor_kwargs['add_pos_type'] = 'none'
    features_extractor_kwargs['img_size'] = args.image_size
    features_extractor_kwargs['feature'] = 'x'
    features_extractor_kwargs['mask_ratio'] = 0
    features_extractor_kwargs['rec_dynamic_45_seg'] = args.rec_dynamic_45_seg
    features_extractor_kwargs['rec_dynamic_45_xscale'] = args.rec_dynamic_45_xscale

    features_extractor_kwargs['rec_idx_bc'] = 128
    features_extractor_kwargs['min_mask_ratio'] = 0.0
    device = args.device
    print(f'****device is {device}')
    assert feature_extractor_ckpt['args'].env_name.lower().replace("_", "") \
            == args.env_name.lower().replace("_", "")

    features_extractor = VisualAttention(**features_extractor_kwargs)
    features_extractor.load_trained_mae(feature_extractor_ckpt['model'])
    features_extractor.to(device)

    len_ids = []
    for samples in data_loader:
        img = samples
        img = img.to(device, non_blocking=True)
        assert len(img.shape) == 4, f"{img.shape}"  # B, C, H, W
        x = features_extractor.patch_embed(img)  # B, L, D
        x = x + features_extractor.pos_embed[:, 1:, :]  # shape (num_env, num_patches, embed_dim)
        B, L, D = x.shape
        with torch.no_grad():
            ids_keep = features_extractor.rec_masking(x=x[:], img=img, loss_type='L2').to(x.device)
        len_ids.append(ids_keep.shape[-1])
        print(f'## salient_check: id_img:{len(len_ids)}, len:{len_ids[-1]}')
    np.save(f'./{args.env_name}_salient_ids', len_ids)
    print(f"******* Statistic for salient patches: max:{max(len_ids)}, min:{min(len_ids)}, avg:{(np.mean(len_ids))}")


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = mae_misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', mae_misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    for data_iter_step, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if isinstance(samples, list):
            assert args.use_kp and len(samples) == 2
            (img_ls, kp_pos_ls) = samples
        else:
            assert not args.use_kp
            img_ls = samples
            kp_pos_ls = None

        if args.debug and args.use_kp:
            test_idx = model.show_patched_image(images=img_ls, kp_pos_ls=kp_pos_ls)
        # all tensor: img_ls: (num_batch, C, H, W), keypoints_ls & kp_pos_ls: (num_batch, num_keypoints, 2)
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        # pdb.set_trace()
        # NOTE: if not args.use_kp, keypoints_ls and kp_pos_ls will be None
        img_ls = img_ls.to(device, non_blocking=True) if img_ls is not None else img_ls
        # keypoints_ls = keypoints_ls.to(device, non_blocking=True) if keypoints_ls is not None else keypoints_ls
        kp_pos_ls = kp_pos_ls.to(device, non_blocking=True) if kp_pos_ls is not None else kp_pos_ls
        
        samples = (img_ls, kp_pos_ls)

        with torch.cuda.amp.autocast():
            if not(args.debug and args.use_kp):
                test_idx = None
            loss, _, _ = model(samples, mask_ratio=args.mask_ratio, test_idx=test_idx, loss_all=args.loss_all,
                                kp_reverse=args.use_reverse_kp, kp_ext_loss_weight=args.kp_ext_loss_weight,
                                kp_ext_loss_multi=args.kp_ext_loss_multi)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = mae_misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    # the global_avg is the statistic about the whole batch
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def main(args):
    assert not args.distributed, "Haven't implement distributed version"
    if args.use_reverse_kp:
        assert args.use_kp and not args.use_aug
    assert not (args.use_kp and args.use_aug), "need to write augmentation functions by myself"
    if args.distributed:
        mae_misc.init_distributed_mode(args)
    if args.use_kp and args.use_reverse_kp:
        assert not args.loss_all
    if args.use_kp:
        assert not args.use_aug

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
        args.output_dir = f'{args.output_dir}/{args.env_name}/{timestamp}'
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    # if global_rank == 0 and args.log_dir is not None:
    #     os.makedirs(args.log_dir, exist_ok=True)
    #     log_writer = SummaryWriter(log_dir=args.log_dir)
    if args.log_tb:
        log_dir = f'{args.output_dir}/tfboard'
        os.makedirs(log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=log_dir)
        print('log_dir: {}'.format(log_writer.log_dir))
    else:
        log_writer = None
    
    if args.save_checkpoint:
        ckpt_path = os.path.join(args.output_dir, 'ckpts')
        os.makedirs(ckpt_path, exist_ok=True)
    if args.visual_eval:
        img_save_path = os.path.join(args.output_dir, 'eval_reconstruct')
        os.makedirs(img_save_path, exist_ok=True)
    
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    if args.save_obs:
        args.save_obs = f'{args.output_dir}/raw_obs'
        Path(args.save_obs).mkdir(parents=True, exist_ok=True)
    else:
        args.save_obs = None

    cudnn.benchmark = True

    # define the model
    if args.model != 'custom':
        model = models_mae.__dict__[args.model](
            img_size=args.image_size, norm_pix_loss=args.norm_pix_loss, debug=args.debug)
        # to save consistent arguments, reload it according to model
        args.in_chans = model.in_chans
        args.patch_size = model.patch_size
        args.embed_dim = model.embed_dim
        args.depth = model.depth
        args.num_heads = model.num_heads
        args.decoder_embed_dim = model.decoder_embed_dim
        args.decoder_depth = model.decoder_depth
        args.decoder_num_heads = model.decoder_num_heads
    else:
        model = MaskedAutoencoderViT(
            patch_size=args.patch_size, embed_dim=args.embed_dim, depth=args.depth, num_heads=args.num_heads,
            decoder_embed_dim=args.decoder_embed_dim, decoder_depth=args.decoder_depth, decoder_num_heads=args.decoder_num_heads,
            mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
            img_size=args.image_size, norm_pix_loss=args.norm_pix_loss, debug=args.debug,
            in_chans=args.in_chans
        )

    model.to(device)
    print(f'Total param: {count_parameters(model)}')
    print(f'Encoder param: patch_embed:{count_parameters(model.patch_embed.proj)}+blocks:{count_parameters(model.blocks)}+norm:{count_parameters(model.norm)} = {count_parameters(model.blocks)+count_parameters(model.patch_embed.proj)+count_parameters(model.norm)}')
    print(f'Decoder param: patch_embed:{count_parameters(model.decoder_embed)}+blocks:{count_parameters(model.decoder_blocks)}+norm:{count_parameters(model.decoder_norm)}+pred:{count_parameters(model.decoder_pred)} = {count_parameters(model.decoder_embed)+count_parameters(model.decoder_blocks)+count_parameters(model.decoder_norm)+count_parameters(model.decoder_pred)}')

    model_without_ddp = model  # without distributed data parallel
    mae_misc.model_summary(model_without_ddp)

    param_path = os.path.join(args.output_dir, 'args.txt')
    with open(param_path, 'w') as f:
        print("{}".format(args).replace(', ', ',\n'), file=f)

    # simple augmentation
    print(f'Collecting {args.num_training_data} training frames')
    if args.use_aug and not args.use_kp:
        # TODO: more kinds of augmentation?
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(args.image_size, scale=(0.2, 1.0),
                                         interpolation=transforms.InterpolationMode.BICUBIC),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),  # transfore a uint8 ndarray ot PIL image from (H, W, C) in [0, 255] to (C, H, W) in [0.0, 1.0]
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        dataset_train = AtariNormalDataset(
            env_name=args.env_name, total_data=args.num_training_data,
            img_size=args.image_size, in_chans=args.in_chans, seed=args.seed,
            transform=transform_train, save_obs=args.save_obs, transpose=False)
    elif not args.use_aug and args.use_kp:
        raise NotImplementedError("haven't fix env wrapper in AtariKeypointDataset")
        transform_train = transforms.Compose([
            transforms.Resize(size=(args.image_size, args.image_size),
                              interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),  # transfore a uint8 ndarray ot PIL image from (H, W, C) in [0, 255] to (C, H, W) in [0.0, 1.0]
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        dataset_train = AtariKeypointDataset(
            env_name=args.env_name, total_data=args.num_training_data,
            image_size=args.image_size, patch_size=model.patch_size,
            transform=transform_train, around=args.kp_around, save_obs=args.save_obs)
    elif not args.use_aug and not args.use_kp:  # NOTE: final version
        # transform_train = transforms.Compose([
        #     # NOTE: has resized image in env_wrapper
        #     # transforms.Resize(size=(args.image_size, args.image_size),
        #     #                   interpolation=transforms.InterpolationMode.BICUBIC),
        #     transforms.ToTensor(),  # transform a uint8 ndarray ot PIL image from (H, W, C) in [0, 255] to (C, H, W) in [0.0, 1.0]
        #     # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # ])
        if args.data_type == 'random':
            dataset_train = AtariNormalDataset(
                env_name=args.env_name, total_data=args.num_training_data,
                img_size=args.image_size, in_chans=args.in_chans, seed=args.seed,
                transform=None, save_obs=args.save_obs, transpose=True)
        elif args.data_type == 'atarizoo':
            dataset_train = AtariOfflineDataset(
                env_name=args.env_name, img_size=args.image_size, in_chans=args.in_chans,
                split='train'
            )
        else:
            raise NotImplementedError
    else:
        # TODO: didn't consider combining keypoint patches with augmentation for now
        raise NotImplementedError()

    sampler_train = torch.utils.data.RandomSampler(dataset_train, 
                    replacement=True, num_samples=args.num_training_data)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    # dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
    print(f'dataset_train: {dataset_train}')
    # transform_eval = transforms.Compose([
    #         # NOTE: resize image should in env_wrapper
    #         # transforms.Resize(size=(args.image_size, args.image_size),
    #         #                     interpolation=transforms.InterpolationMode.BICUBIC),
    #         transforms.ToTensor(),  # transfore a uint8 ndarray ot PIL image from (H, W, C) in [0, 255] to (C, H, W) in [0.0, 1.0]
    #         # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #     ])
    if args.eval_kps:
        raise NotImplementedError("haven't fix env_wrapper in this Dataset")
        dataset_eval = AtariKeypointDataset(
            env_name=args.env_name, total_data=args.num_eval_data,
            image_size=args.image_size, patch_size=model.patch_size,
            transform=transform_eval, around=args.kp_around, save_obs=None)
    else:
        if args.data_type == 'random':
            dataset_eval = AtariNormalDataset(
                env_name=args.env_name, total_data=args.num_eval_data,
                img_size=args.image_size, in_chans=args.in_chans, seed=args.seed,
                transform=None, save_obs=None, transpose=True)
        elif args.data_type == 'atarizoo':
            dataset_eval = AtariOfflineDataset(
                env_name=args.env_name, img_size=args.image_size, in_chans=args.in_chans,
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

    # eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    eff_batch_size = args.batch_size * args.accum_iter
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    # if args.distributed:
    #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
    #     model_without_ddp = model.module
    
    # following timm: set wd as 0 for bias and norm layers
    # NOTE: norm layers' param: weight and bias are both len(param.shape) == 1
    param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    if args.resume:
        mae_misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if args.statistic_salient:
        data_loader_train_stastic = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=1,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )
        statistic_salient(data_loader_train_stastic, args)
        return

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    min_loss = math.inf
    min_loss_epoch = None
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        if args.eval_epoch > 0 and epoch % args.eval_epoch == 0 or epoch + 1 == args.epochs:
            # TODO: if so, we can try to remember the best model then save it after training
            # TODO: can also check if better reconstruction can lead to better doenstream task
            if train_stats['loss'] < min_loss:  # save a best model
                # TODO: a better way to save the best model? 
                min_loss = train_stats['loss']
                min_loss_epoch = epoch
            if (args.save_checkpoint and not args.save_last_checkpoint_only) or \
                (args.save_checkpoint and args.save_last_checkpoint_only and epoch + 1 == args.epochs):
                mae_misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, exclude_decoder=args.save_exclude_decoder)
            if args.visual_eval:
                test_stats = evaluate(data_loader_eval, model, device, epoch, img_save_path)
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}

        if args.output_dir and mae_misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    print(f'min_loss_epoch: {min_loss_epoch}, min_loss: {min_loss}')

    # if args.visual_eval:
    #     checkpoint_best = torch.load(os.path.join(ckpt_path, 'checkpoint-%s.pth'), map_location='cpu')
    #     model_without_ddp.load_state_dict(checkpoint['model'])
    #     model_without_ddp.eval()



if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.env_name in ARG2ALE_GAME_NAME_MAP:
        args.env_name = ARG2ALE_GAME_NAME_MAP[args.env_name]
    main(args)
