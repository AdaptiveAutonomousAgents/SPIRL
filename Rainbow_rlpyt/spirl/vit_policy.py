import gym
import pdb
import numpy as np
import torch as th
from torch import nn
import copy
import cv2
import os
from typing import Dict, List, Tuple, Type, Union, Optional, Any
from collections import OrderedDict
from functools import partial
from itertools import product, zip_longest
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.distributions import CategoricalDistribution

from timm.models.vision_transformer import PatchEmbed, Block, trunc_normal_

from .pos_embed import interpolate_pos_embed, get_2d_sincos_pos_embed
from .init import lecun_normal_
from .lr_decay import param_groups_lrd
from .draw_utils import draw_mask, ax_plot, ax_line_plot


# key: num_patch_per_row, val: patch position of corresponding type
PATCH_CORNER = {
    12: [0, 11, 132, 143],
}

"""
# code to generate edge idx
N = 12  # num_patch_per_row
ff = ""
for i in range(1, N-1):
    ff += f'{i}, '
for i in range(1, N-1):
    ff += f'{i*N}, '
    ff += f'{i*N + N - 1}, '
for i in range(1, N-1):
    ff += f'{N*(N-1) + i}, '
print(ff)
"""
PATCH_EDGE = {
    12: [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, \
        12, 23, 24, 35, 36, 47, 48, 59, \
        60, 71, 72, 83, 84, 95, 96, 107, \
        108, 119, 120, 131, 133, 134, 135, \
        136, 137, 138, 139, 140, 141, 142
    ],
}

"""
# code to generate inside idx
ff = ""
for i in range(1, N-1):
    for j in range(1, N-1):
        ff += f'{i*N+j}, '
print(ff)
"""
PATCH_INSIDE = {
    12: [
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, \
        25, 26, 27, 28, 29, 30, 31, 32, 33, 34, \
        37, 38, 39, 40, 41, 42, 43, 44, 45, 46, \
        49, 50, 51, 52, 53, 54, 55, 56, 57, 58, \
        61, 62, 63, 64, 65, 66, 67, 68, 69, 70, \
        73, 74, 75, 76, 77, 78, 79, 80, 81, 82, \
        85, 86, 87, 88, 89, 90, 91, 92, 93, 94, \
        97, 98, 99, 100, 101, 102, 103, 104, 105, 106, \
        109, 110, 111, 112, 113, 114, 115, 116, 117, 118, \
        121, 122, 123, 124, 125, 126, 127, 128, 129, 130,
    ],
}

def get_surr_idx(N):
    """
    Get surrounding kp positions for each idx in an N*N matrix.
    """
    P = N * N  # total number of patches
    ret_ids_restore = np.ones((P, P), dtype=int) * (P - 1)
    ret_surr_idx_ls = {}
    # -1: in the input to decoder, the last one must be mask_token
    #    (since you'll get 8 surrounging patches at most, then after
    #    the first 8 position will difinitly be mask_token)
    for surr_mask_idx in range(P):
        row, column = surr_mask_idx // N, surr_mask_idx % N
        surr_idx_ls = []
        for d_row, d_column in product([-1, 0, 1], [-1, 0, 1]):
            if d_row == 0 and d_column == 0:  # exclude itself
                continue
            next_row = row + d_row
            next_column = column + d_column
            if next_row < 0 or next_row >= N or \
                next_column < 0 or next_column >= N:
                continue
            surr_idx =  next_row * N + next_column
            ret_ids_restore[surr_mask_idx][surr_idx] = len(surr_idx_ls)  # NOTE: must put it before appending surr_idx! to keep the id position correct
            surr_idx_ls.append(surr_idx)
        ret_surr_idx_ls[surr_mask_idx] = surr_idx_ls
    return ret_ids_restore, ret_surr_idx_ls  # an P*P array, and a dict

def split_arr_by_bc(arr, itv):
    """
    Split an 1-d array into several 1-d array segments, according to given interval.
    """
    num_comp = len(arr) // itv  # the number of completed segments
    num_mod = len(arr) % itv  # extra elements
    ret = []
    for i in range(num_comp):
        ret.append(arr[i*itv: (i+1)*itv])
    if num_mod > 0:
        ret.append(arr[num_comp*itv:])
    return ret

def get_latest_run_id(log_path: str) -> int:
    """
    Returns the latest run number for the given log name and log path,
    by finding the greatest number in the directories.

    :param log_path: path to log folder
    :return: latest run number
    """
    max_run_id = 0
    for path in glob.glob(os.path.join(log_path, "*_[0-9]*")):
        file_name = os.path.basename(path)
        ext = file_name.split("_")[-1].split(".")[0]
        if ext.isdigit() and int(ext) > max_run_id:
            max_run_id = int(ext)
    return max_run_id


def mark_img_loss_gray(ori_img, loss_mat, mark_ratio, grid_size, keep_id=None):
    '''
    ori_img: an RGB image in shape HWC
    loss_mat: each loss corresponds to a grid
    mark_ratio: mark the top mark_ratio grid as gray
    grid_size: each grid is in shape grid_size*grid_size
    '''
    h, w, c = ori_img.shape
    assert c == 3
    assert h % grid_size == 0
    assert w % grid_size == 0
    gray_ori_image = np.repeat(cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)[..., None], repeats=3, axis=-1)
    return_img = copy.deepcopy(ori_img)

    loss_h, loss_w = loss_mat.shape[0], loss_mat.shape[1]
    assert h // grid_size == loss_h
    assert w // grid_size == loss_w

    if keep_id is None:
        loss = loss_mat.reshape(-1)
        order = np.argsort(loss) # ascending order
        len_keep = int(loss_mat.size * mark_ratio)
        keep_id = order[-len_keep:]

    for id in keep_id:
        row = id // loss_h
        col = id % loss_w
        return_img[row*grid_size:(row+1)*grid_size, col*grid_size:(col+1)*grid_size, :] = \
            gray_ori_image[row*grid_size:(row+1)*grid_size, col*grid_size:(col+1)*grid_size, :]

    return return_img


class VisualAttention(nn.Module):
    """ Vision Transformer with support for global average pooling
        based on timm.models.vision_transformer.VisionTransformer
        NOTE: not exactly a ViT, since remove the head part, input images, output latent features
              leave other part for MlpExtractor
    """
    def __init__(self, img_size, patch_size, in_chans, embed_dim, depth, feature,
                 mask_ratio, add_pos_type,
                 num_heads, mlp_ratio=4., qkv_bias=True, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=None,
                 decoder_embed_dim=None, decoder_depth=None, decoder_num_heads=None,
                 check_kp_path=None, kp='rec', rec_idx_bc=128, min_mask_ratio=None,
                 grad_pos=False, grad_cls=False, norm_pix_loss=True, rec_thres=None,
                 rec_weight=None, rec_dynamic_deltx=None, rec_dynamic_ma=None,
                 rec_dynamic_k_seg=None, mask_ratio_random_bound=None,
                 rec_dynamic_45_seg=None, rec_dynamic_45_xscale=None,
                ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            pi_embed_dim (int): to shorten pi feature concatenation length, map feature from embed_dim to pi_embed_dim for latent_feature_pi
            # representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer (nn.Module): normalization layer
            feature (str): set feature type if it's not None
            mask_ratio (float): ratio to mask patches
            # add_pos (str): add position information before or after norm
            add_pos_type (str): type of added position information
            rec_thres (float): for vitRec, if not None, will select key patches based on thres err instead of mask_ratio
            rec_weight (float): for vitRec, if not None, will select key patches based on weight err instead of mask_ratio
            rec_idx_bc (int): for vitRec, the batch size when calculate reconstruction error map
            mask_ratio_random_bound (float): mask_ratio with extra random patches up to mask_ratio_random_bound
        """
        super().__init__()

        # self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        assert distilled == False  # TODO: for now do not consider distill
        self.distilled = distilled
        # assert representation_size == -1 # TODO: do we really need it?
        self.num_tokens = 2 if distilled else 1

        self.norm_pix_loss = norm_pix_loss
        
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.in_chans = in_chans
        self.patch_size = patch_size
        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.num_patch_per_row = self.patch_embed.img_size[1] // self.patch_embed.patch_size[1]
        if min_mask_ratio is not None:
            assert mask_ratio_random_bound == None
            self.max_len_keep = int(num_patches * (1 - min_mask_ratio))
            self.min_len_mask = num_patches - self.max_len_keep
        if mask_ratio_random_bound is not None:
            assert min_mask_ratio == None
            self.max_len_keep = int(num_patches * (1 - mask_ratio_random_bound))

        self.cls_token = nn.Parameter(th.zeros(1, 1, embed_dim), requires_grad=grad_cls)
        self.dist_token = nn.Parameter(th.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(th.zeros(1, num_patches + self.num_tokens, embed_dim), requires_grad=grad_pos)  # TODO: need grad? why mae do not use grad for it?
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in th.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)  # NOTE: check norm-layer, it must be tuned after pretraining

        self.mask_ratio = mask_ratio
        self.mask_ratio_random_bound = mask_ratio_random_bound
        self.rec_thres = rec_thres
        self.rec_weight = rec_weight
        self.rec_dynamic_deltx = rec_dynamic_deltx
        self.rec_dynamic_ma = rec_dynamic_ma
        self.rec_dynamic_k_seg = rec_dynamic_k_seg
        self.rec_dynamic_45_seg = rec_dynamic_45_seg
        self.rec_dynamic_45_xscale = rec_dynamic_45_xscale

        self.ram_kp = (kp == 'ram')  # use Atari-RAM
        self.rec_kp = (kp == 'rec')  # use reconstruct error 
        if kp not in ['ram', 'rec']:
            raise NotImplementedError

        # Representation layer
        self.embed_dim = embed_dim
        self.pre_logits = nn.Identity()

        if feature not in [
            'cls', 'all', 'x', 'concat', 'xCls',
            'normAvg', 'avg', 'normAvgCls', 'avgCls',
            'normMax', 'max', 'normMaxCls', 'maxCls',
            'normSum', 'sum', 'normSumCls', 'sumCls',
            'avgMaxCls', 'avgMax', 'sumMaxCls', 'sumMax'
        ]:
            raise NotImplementedError
        self.feat_cls = (feature == 'cls')
        self.feat_all = (feature == 'all' or feature == 'xCls')
        self.feat_x = (feature == 'x')
        self.feat_concat = (feature == 'concat')

        # different pooling
        self.feat_norm_avg = (feature == 'normAvg')
        self.feat_avg = (feature == 'avg')
        self.feat_norm_avg_cls = (feature == 'normAvgCls')
        self.feat_avg_cls = (feature == 'avgCls')

        self.feat_norm_max = (feature == 'normMax')
        self.feat_max = (feature == 'max')
        self.feat_norm_max_cls = (feature == 'normMaxCls')
        self.feat_max_cls = (feature == 'maxCls')

        self.feat_norm_sum = (feature == 'normSum')
        self.feat_sum = (feature == 'sum')
        self.feat_norm_sum_cls = (feature == 'normSumCls')
        self.feat_sum_cls = (feature == 'sumCls')

        # TODO: if previous norm_sum/norm_avg works better than no norm version, we could try to add norm for the following
        #       then we need to consider if we should share norm layer for sum/max/avg
        self.feat_avg_max_cls = (feature == 'avgMaxCls')
        self.feat_avg_max = (feature == 'avgMax')
        self.feat_sum_max_cls = (feature == 'sumMaxCls')
        self.feat_sum_max = (feature == 'sumMax')

        if add_pos_type not in ['addSS', 'catXY', 'catXYMap', 'addSScatXY', 'addSScatXYMap', 'none']:
            raise NotImplementedError
        self.pos_addSS = (add_pos_type == 'addSS')
        self.pos_catXY = (add_pos_type == 'catXY')
        self.pos_catXYMap = (add_pos_type == 'catXYMap')
        self.pos_addSScatXY = (add_pos_type == 'addSScatXY')
        self.pos_addSScatXYMap = (add_pos_type == 'addSScatXYMap')
        self.pos_none = (add_pos_type == 'none')

        if self.rec_kp:
            self.decoder_embed_dim = decoder_embed_dim
            self.decoder_depth = decoder_depth
            self.decoder_num_heads = decoder_num_heads
            # MAE decoder specifics
            self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

            self.mask_token = nn.Parameter(th.zeros(1, 1, decoder_embed_dim))

            self.decoder_pos_embed = nn.Parameter(th.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

            self.decoder_blocks = nn.ModuleList([
                Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
                for i in range(decoder_depth)])

            self.decoder_norm = norm_layer(decoder_embed_dim)
            self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch

            self._rec_idx_bc = rec_idx_bc  # batch size for calculating reconstruction error map, TODO: change it by args
            self._rec_idx_group = []
            self._rec_idx_group.extend(split_arr_by_bc(PATCH_CORNER[self.num_patch_per_row], self._rec_idx_bc))
            self._rec_idx_group.extend(split_arr_by_bc(PATCH_EDGE[self.num_patch_per_row], self._rec_idx_bc))
            self._rec_idx_group.extend(split_arr_by_bc(PATCH_INSIDE[self.num_patch_per_row], self._rec_idx_bc))

            self.ret_ids_restore, self.ret_surr_idx_ls = get_surr_idx(self.num_patch_per_row)

        self.check_kp_path = check_kp_path
        if self.check_kp_path is not None:
            os.makedirs(self.check_kp_path, exist_ok=True)
            self.img_cnt = 0

        self.init_weights()            

    def init_weights(self):
        # refer to mae's initialization
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(th.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        th.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        if self.dist_token is not None:
            trunc_normal_(self.dist_token, std=.02)
        trunc_normal_(self.cls_token, std=.02)

        if self.rec_kp:
            decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
            self.decoder_pos_embed.data.copy_(th.from_numpy(decoder_pos_embed).float().unsqueeze(0))
            trunc_normal_(self.mask_token, std=.02)

        self.apply(_init_vit_weights)
        self.freeze_vit()

    def freeze_vit(self):
        # list that don't need to freeze (but for pos_embed it still may be requires_grad=True, depend on argparse)
        # vit_grad_TBD_name = ["cls_token", "norm.bias", "norm.weight", 
        #                      "vf_pool_norm.bias", "vf_pool_norm.weight", 
        #                      "pos_norm.weight", "pos_norm.bias",
        #                      "pos_embed"]
        # prefix = "features_extractor.vit_model."
        # vit_grad_TBD_name = [prefix+grad_name for grad_name in vit_grad_TBD_name]
        # TODO: consider fine-tune norm layer in future code
        #       currently, since we use vit inside env_wrapper, assume we freeze all layers
        vit_grad_TBD_name = []
        for n, p in self.named_parameters():
            if not (n in vit_grad_TBD_name):
                p.requires_grad = False

    def load_trained_mae(self, checkpoint_model: dict):
        interpolate_pos_embed(self, checkpoint_model)  # TODO: previously, mae's pos_embed is not trained
        msg = self.load_state_dict(checkpoint_model, strict=False)
        print(f"{'*'*10}checkpoint load information for pretrained mae model{'*'*10}")
        print(msg)
        # missing_keys = []
        # if self.vit_model.vf_global_pool or self.vit_model.pi_global_pool:
        #     missing_keys.extend(['fc_norm.weight', 'fc_norm.bias'])
        # if self.vit_model.proj_embed_dim > 0 and not self.vit_model.distilled:
        #     missing_keys.extend(['pre_logits.fc.weight', 'pre_logits.fc.bias'])

        # assert set(msg.missing_keys) == set(missing_keys)

    def no_weight_decay(self):  # used when construct optimizer
        return {'pos_embed', 'cls_token', 'dist_token'}

    def kp_masking(self, x, kp_pos_ls):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length (num_patches), dim
        len_keep = min(kp_pos_ls.shape[1], L)

        noise = th.rand(N, L, device=x.device)  # noise in [0, 1]
        flat_kp_pos_ls = kp_pos_ls[..., 1] * self.num_patch_per_row  + kp_pos_ls[..., 0]  # (num_batch, num_keypoints)
        noise.scatter_(dim=1, index=flat_kp_pos_ls.long(), src=th.zeros(flat_kp_pos_ls.shape, dtype=noise.dtype, device=noise.device))
        # sort noise for each sample
        ids_shuffle = th.argsort(noise, dim=1)  # ascend: smaller ones are kept, larger ones are removed

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]  # shape (num_batch, len_keep)
        x_masked = th.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))  # shape (num_batch, len_keep, dim)

        # x_masked: (num_batch, len_keep, dim)
        return x_masked, ids_keep

    def forward_encoder_idx(self, x, surr_mask_idx):
        B, P, D = x.shape
        M = len(surr_mask_idx)
        ids_keep_ls = []
        for idx in surr_mask_idx:
            ids_keep_ls.append(self.ret_surr_idx_ls[idx])
        ids_keep = th.tensor(ids_keep_ls, device=x.device).unsqueeze(-1).expand(-1, -1, D)  # (B*len(surr_mask_idx), num_surr, D)
        x = th.gather(x.expand(M, -1, -1), dim=1, index=ids_keep)  # shape (num_batch*, len_keep, dim)
        ids_restore = th.tensor(self.ret_ids_restore[surr_mask_idx], device=x.device, dtype=int)
        
        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]  # shape (1, 1, embed_dim)
        cls_tokens = cls_token.expand(B * M, -1, -1)  # shape (num_batches * M, 1, embed_dim)
        x = th.cat((cls_tokens, x), dim=1)  # shape (num_batches * M, 1 + len_keep, embed_dim)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)  # shape (num_batches, 1 + len_keep, embed_dim)

        # x: (num_batches * M, 1 + len_keep, embed_dim)
        # ids_restore: (num_batch * M, num_patches)
        return x, ids_restore

    def forward_decoder_idx(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)  # shape: (num_batches * M, 1 + len_keep, decoder_embed_dim)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.expand(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], -1)
        x_ = th.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token, shape: (num_batches * M, num_patches, decoder_embed_dim)
        x_ = th.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[2]))  # unshuffle
        x = th.cat([x[:, :1, :], x_], dim=1)  # append cls token, shape: (num_batches, 1 + num_patches, decoder_embed_dim)

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)  # shape: (num_batches, 1 + num_patches, decoder_embed_dim)

        # predictor projection
        x = self.decoder_pred(x)  # shape: (num_batches, 1 + num_patches, patch_size ** 2 * in_chanels)

        # remove cls token
        x = x[:, 1:, :]  # shape: (num_batches, num_patches, patch_size ** 2 * in_chanels)

        return x

    def forward_err_idx(self, imgs, pred, surr_mask_idx, loss_type="L2"):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)  # shape: (num_batches, h * w, p**2 * 3)
        if self.norm_pix_loss:
            # --norm_pix_loss as the target for better representation learning. 
            # To train a baseline model (e.g., for visualization), use pixel-based construction and turn off --norm_pix_loss.
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        if loss_type == 'L2':
            err = (pred - target) ** 2  # (num_batch, num_patch, p**2 * 3)
            err = err.mean(dim=-1)  # [num_batch, num_patch], mean loss per patch
        elif loss_type == 'L2trim4':
            # pdb.set_trace()
            err = (pred - target) ** 2  # (num_batch, num_patch, p**2 * 3)
            err_vals, order = th.sort(err, dim=-1)
            err = err_vals[:,:,:-4].mean(dim=-1)
        # err_idx = err[:, surr_mask_idx]  # (num_batch)
        err_idx = th.gather(err, index=th.tensor(surr_mask_idx, dtype=int, device=err.device).view(-1, 1), dim=-1)

        return err_idx.view(-1)

    @th.no_grad()
    def rec_masking(self, x, img, loss_type="L2"):
        """
        x: patch embeddings
        img: original images
        """
        # # embed patches
        # x = self.patch_embed(img)  # shape (num_batches, num_patches, embed_dim)
        # # add pos embed w/o cls token
        # x = x + self.pos_embed[:, 1:, :]  # shape (num_batches, num_patches, embed_dim)
        B, L, D = x.shape
        assert B == 1 # NOTE: for now, we parallelize for one single frame at one time
        err_mat = th.zeros((B, self.patch_embed.num_patches))

        for idx in self._rec_idx_group:
            # keep each single patch, calculate reconstruction error for that single patch
            latent, ids_restore = self.forward_encoder_idx(x, surr_mask_idx=idx)
            pred = self.forward_decoder_idx(latent, ids_restore)  # [N, L, p*p*3]  (num_batches*M, num_patches, patch_size ** 2 * in_chanels)
            err_idx = self.forward_err_idx(img, pred, surr_mask_idx=idx, loss_type=loss_type)
            err_mat[..., idx] = err_idx.cpu()  # NOTE: no gradient here!

        # err_mat_2 = th.zeros((B, self.patch_embed.num_patches))
        # for idx in range(self.patch_embed.num_patches):
        #     # keep each single patch, calculate reconstruction error for that single patch
        #     latent, ids_restore = self.forward_encoder_idx(x, surr_mask_idx=[idx])
        #     pred = self.forward_decoder_idx(latent, ids_restore)  # [N, L, p*p*3]  (num_batches*M, num_patches, patch_size ** 2 * in_chanels)
        #     err_idx = self.forward_err_idx(img, pred, surr_mask_idx=[idx])
        #     err_mat_2[..., idx] = err_idx.cpu()  # NOTE: no gradient here!
        # assert th.all(th.abs(err_mat-err_mat_2)<1e-5).item()
        # TODO: in fact, if use mask_ratio, we don't need MinMaxScale here, order of numbers are still the same
        err_mat_max = th.max(err_mat, dim=-1)[0][..., None]  # (B, 1)
        err_mat_min = th.min(err_mat, dim=-1)[0][..., None]  # (B, 1)
        err_mat_std = (err_mat - err_mat_min) / (err_mat_max - err_mat_min)
        # err_mat_scaled = err_mat_std * (err_mat_max - err_mat_min)  # (B, P)
        err_mat_scaled = err_mat_std * 255  # (B, P)

        if self.rec_thres is not None:
            vals, order = th.sort(err_mat_std, dim=-1)
            len_keep = max(1, min(len(th.where(vals > self.rec_thres)[-1]), self.max_len_keep))  # at least choose one patch
            ids_keep = order[:, -len_keep:]  # select patches with larger error
        elif self.rec_weight is not None:
            vals, order = th.sort(err_mat, dim=-1) # ascending order
            assert th.all(err_mat>=0).item()  # TODO: remove this when laucnch experiments
            vals /= th.sum(vals, dim=-1)  # L1 normalized
            cumval = th.cumsum(vals, dim=-1)
            th_attn = cumval > (1 - self.rec_weight)
            len_keep = min(max(1, int(th.sum(th_attn))), self.max_len_keep)
            ids_keep = order[:, -len_keep:]  # select patches with larger error, note that here B==1
        elif self.rec_dynamic_deltx is not None:
            vals, order = th.sort(err_mat.reshape((-1,)), dim=-1) # ascending order
            vals /= abs(th.sum(vals, dim=-1))  # similar to attn.softmax(dim=-1)
            ret = th.zeros_like(vals)
            delt_x = self.rec_dynamic_deltx
            for t in range(vals.shape[-1] - 1):
                if vals[t] < 1e-3:
                    ret[t] = 0
                else:
                    ret[t] = (vals[t+1]-vals[t])/(np.sqrt(1+(vals[t+1]/delt_x)**2)*np.sqrt(1+(vals[t]/delt_x)**2))
            ret[vals.shape[-1] - 1] = ret[vals.shape[-1] - 2]
            ma_window = self.rec_dynamic_ma
            ret_ma = th.nn.AvgPool1d(kernel_size=ma_window, stride=1, padding=ma_window//2)(ret.reshape((1, -1)))
            argmax_ma_t = th.argmax(ret_ma, dim=-1)
            # ret_ma = np.convolve(ret, np.ones(ma_window), 'same') / ma_window
            # argmax_ma_t = np.argmax(ret_ma)
            len_keep = min(max(1, self.patch_embed.num_patches-argmax_ma_t[0].item()), self.max_len_keep)  # at least keep 2 patches
            ids_keep = order[None, -len_keep:]  # select patches with larger error
        elif self.rec_dynamic_k_seg is not None:
            vals, order = th.sort(err_mat.reshape((-1,)), dim=-1) # ascending order
            vals /= abs(th.sum(vals, dim=-1))  # similar to attn.softmax(dim=-1)
            cumval = th.cumsum(vals, dim=-1)
            seg = self.rec_dynamic_k_seg
            val_div = th.zeros_like(vals)
            for idx in range(self.min_len_mask, self.patch_embed.num_patches - seg):
                val_div[idx] = (cumval[idx+seg]-cumval[idx])/(cumval[idx]-cumval[idx-seg])
            div_max = th.argmax(val_div, axis=-1).item()
            len_keep = min(max(1, self.patch_embed.num_patches - div_max), self.max_len_keep)
            ids_keep = order[None, -len_keep:]
        elif self.rec_dynamic_45_seg is not None:
            vals, order = th.sort(err_mat.reshape((-1,)), dim=-1) # ascending order
            vals /= abs(th.sum(vals, dim=-1))  # similar to attn.softmax(dim=-1)
            cumval = th.cumsum(vals, dim=-1)
            half_seg = self.rec_dynamic_45_seg//2
            cumval_diff = th.ones_like(vals)
            for idx in range(self.min_len_mask, self.patch_embed.num_patches - half_seg):
                cumval_diff[idx] = cumval[idx+half_seg] - cumval[idx-half_seg]
            selected_idx = th.argmin(
                th.abs(cumval_diff-(self.rec_dynamic_45_xscale*1.0)/self.patch_embed.num_patches), axis=-1).item()
            len_keep = min(max(1, self.patch_embed.num_patches - selected_idx), self.max_len_keep)
            ids_keep = order[None, -len_keep:]
        else:  # mask_ratio
            order = th.argsort(err_mat_scaled, dim=-1)  # ascending order
            len_keep = int(L * (1 - self.mask_ratio))
            ids_keep = order[:, -len_keep:]  # select patches with larger error

            if self.mask_ratio_random_bound is not None:
                assert B == ids_keep.shape[0]
                ids_keep_extend = th.zeros((B, self.max_len_keep),
                                            dtype=ids_keep.dtype,
                                            device=ids_keep.device)
                for idx_b in range(B):
                    patch_extend = ids_keep[idx_b,:].tolist()
                    for id_patch in ids_keep[idx_b,:]:
                        patch_extend.extend(self.ret_surr_idx_ls[id_patch.item()])
                    patch_extend = th.unique(th.tensor(patch_extend), sorted=False)
                    if len(patch_extend) >= self.max_len_keep:
                        # NOTE: try to avoid this, use a small mask_ratio and large mask_ratio_random_bound
                        patch_extend = patch_extend[-self.max_len_keep:]
                    else:
                        random_extend_idx = np.random.permutation(
                            range(self.patch_embed.num_patches - len(patch_extend)))\
                            [:self.max_len_keep - len(patch_extend)]
                        random_extend = th.gather(th.tensor(list(set(range(self.patch_embed.num_patches))
                                                       -set(patch_extend.tolist()))),
                                                  dim=-1,
                                                  index=th.tensor(random_extend_idx))
                        patch_extend = th.tensor(list(patch_extend),
                                                 dtype=ids_keep.dtype,
                                                 device=ids_keep.device)
                        patch_extend = th.concat([patch_extend, random_extend])
                    ids_keep_extend[idx_b, :] = patch_extend[:]
                ids_keep = ids_keep_extend[:]

        if self.check_kp_path is not None:
            if self.rec_thres is not None:
                title = f'thres: {self.rec_thres:.3f}, '
                fig, axes = plt.subplots(1, 2, figsize=(2*5, 1*5))
            elif self.rec_weight is not None:
                title = f'weight: {self.rec_weight:.3f}, '
                fig, axes = plt.subplots(1, 2, figsize=(2*5, 1*5))
            elif self.rec_dynamic_deltx is not None:
                title = f'dynamic_deltx: {self.rec_dynamic_deltx:.6f}, '
                fig, axes = plt.subplots(1, 3, figsize=(2*5, 1*5))
            elif self.rec_dynamic_k_seg is not None:
                title = f'dynamic_k_seg: {self.rec_dynamic_k_seg}, '
                fig, axes = plt.subplots(1, 3, figsize=(2*5, 1*5))
            elif self.rec_dynamic_45_seg is not None:
                title = f'dynamic_45_seg: {self.rec_dynamic_45_seg}\n xscale: {self.rec_dynamic_45_xscale}, '
                fig, axes = plt.subplots(1, 3, figsize=(2*5, 1*5))
            else:
                title = f'mask_ratio: {self.mask_ratio:.3f}, '
                fig, axes = plt.subplots(1, 2, figsize=(2*5, 1*5))

            draw_mask(keep_id=ids_keep[0], model=self,
                      mask_shape=(self.num_patch_per_row, self.num_patch_per_row),
                      ori_img=(img*255).cpu().numpy().transpose(0, 2, 3, 1)[0], ax=axes[0],
                      title=title+f'len_keep: {len_keep}')
            t_mat = nn.functional.interpolate(
                        err_mat_std.reshape(self.num_patch_per_row, self.num_patch_per_row).unsqueeze(0).unsqueeze(0),
                        scale_factor=(self.patch_size, self.patch_size),
                        mode="nearest")[0][0].cpu().numpy().astype('float32')
            ax_plot(image=t_mat, ax=axes[1],
                    title=f'MinMaxScaled rec error map\nsum:{err_mat_std.sum():.3e}, min:{err_mat_std.min():.3e}, max: {err_mat_std.max():.3e}')
            if self.rec_dynamic_deltx is not None:
                ax_line_plot(ret_ma[0], axes[2],
                             title=f'signal with ma={ma_window}', marker='x', label=f'ma={ma_window}')
            elif self.rec_dynamic_k_seg is not None:
                ax_line_plot(cumval, axes[2],
                            title=f'cumsum L1 normed & div-seg-{seg}', marker='x')
                ax_line_plot(val_div, axes[2],
                            title=f'cumsum L1 normed & div-seg-{seg}', marker='.', is_twin=True, color='r')
                div_max = np.argmax(val_div, axis=-1)
                axes[2].axvline(x=div_max, color='g')
            elif self.rec_dynamic_45_seg is not None:
                ax_line_plot(cumval, axes[2],
                            title=f'cumsum L1 normed & 45-seg-{self.rec_dynamic_45_seg}\n scale-{self.rec_dynamic_45_xscale}',
                            marker='x')
                ax_line_plot(cumval_diff, axes[2],
                             title=f'cumsum L1 normed & 45-seg-{self.rec_dynamic_45_seg}\n scale-{self.rec_dynamic_45_xscale}',
                             marker='.', is_twin=True, color='r')
                selected_idx = np.argmin(
                    np.abs(cumval_diff-1.0/self.patch_embed.num_patches), axis=-1)
                axes[2].axvline(x=selected_idx, color='g')
            fig.tight_layout()
            save_file = os.path.join(self.check_kp_path, f"rec_{self.img_cnt}_{len_keep}.png")
            self.img_cnt += 1
            fig.savefig(fname=save_file)

        return ids_keep

    def forward_one_frame(self, img, kp_pos=None, loss_type="L2"):
        assert len(img.shape) == 4, f"{img.shape}"  # B, C, H, W
        x = self.patch_embed(img)  # B, L, D
        x = x + self.pos_embed[:, 1:, :]  # shape (num_env, num_patches, embed_dim)
        B, L, D = x.shape
        if self.rec_kp:
            with th.no_grad():
                ids_keep = self.rec_masking(x=x[:], img=img, loss_type=loss_type).to(x.device)
            x = th.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))  # shape (num_batch, len_keep, dim)
        elif self.ram_kp:
            # x only keep unmasked patch embedding
            # ids_keep: shape(B, len_keep)
            x, ids_keep = self.kp_masking(x=x[:], kp_pos_ls=kp_pos)

            if self.check_kp_path is not None:
                print(f"{img.min().item()}_{img.max().item()}")
                grid_size = self.patch_embed.patch_size[0]
                pos = 0 # should only have one image, in shape (1, 3, h, w)
                cv2_ori_image = (img*255).cpu().numpy().transpose(0, 2, 3, 1)  # torch image: BCHW --> cv2 image: BHWC
                return_img = copy.deepcopy(cv2_ori_image[pos])
                gray_ori_image = np.repeat(cv2.cvtColor(return_img, cv2.COLOR_BGR2GRAY)[..., None], repeats=3, axis=-1)
                for id in ids_keep[pos]:
                    row = id.item() // self.num_patch_per_row
                    col = id.item() % self.num_patch_per_row
                    return_img[row*grid_size:(row+1)*grid_size, col*grid_size:(col+1)*grid_size, :] = \
                        gray_ori_image[row*grid_size:(row+1)*grid_size, col*grid_size:(col+1)*grid_size, :]
                save_file = os.path.join(self.check_kp_path, f"rec_{self.img_cnt}_{img.max().item()}_{img.min().item()}.png")
                self.img_cnt += 1
                cv2.imwrite(save_file, return_img)

        cls_token = self.cls_token + self.pos_embed[:, :1, :]  # shape (1, 1, embed_dim)
        cls_token = self.cls_token.expand(B, -1, -1)
        x = th.cat((cls_token, x), dim=1)  # num_env, len_keep+1, embed_dim
        x = self.pos_drop(x)

        x = self.blocks(x)  # (B, len_keep + 1, embed_dim)

        # TODO: maybe concatenating positional embedding instead of adding, since global pooling may 
        #       then also need to change MAE pretrain
        norm_x = self.norm(x)  # shape(B, len_keep + 1, embed_dim)
        if self.pos_addSS:
            expand_pos_embed = self.pos_embed.repeat(B, 1, 1)  # (B, 1+num_patches, embed_dim)
            expand_ids_keep = ids_keep[..., None].expand(-1, -1, self.embed_dim)  # (B, len_keep, embed_dim)
            norm_x = norm_x + th.cat((expand_pos_embed[:, :1, :],
                                        th.gather(input=expand_pos_embed[:, 1:, :], dim=1, index=expand_ids_keep)),
                                        dim=1)
            # norm_x = norm_x + th.cat((self.pos_embed[:, :1, :], self.pos_embed[:, ids_keep.view(-1), :]), dim=1)
        elif self.pos_catXY:
            expand_ids_keep = ids_keep[..., None]
            # for non-cls embedding, row-columu starts from 1
            ids_keep_xy = th.cat((expand_ids_keep % self.num_patch_per_row, expand_ids_keep // self.num_patch_per_row), dim=-1) + 1
            # for cls token, give it position (0, 0)  # TODO: we may not need cls token if we don't use it as Q value
            ids_keep_xy = th.cat((th.zeros_like(ids_keep_xy[:, :1, :]), ids_keep_xy), dim=-2)
            norm_x = th.cat((norm_x, ids_keep_xy/self.num_patch_per_row), dim=-1)  # after layer norm, the data follows standard normal distribution
        elif self.pos_catXYMap:
            expand_ids_keep = ids_keep[..., None]
            # for non-cls embedding, row-columu starts from 1
            ids_keep_xy = th.cat((expand_ids_keep % self.num_patch_per_row, expand_ids_keep // self.num_patch_per_row), dim=-1) + 1
            # for cls token, give it position (0, 0)  # TODO: we may not need cls token if we don't use it as Q value
            ids_keep_xy = th.cat((th.zeros_like(ids_keep_xy[:, :1, :]), ids_keep_xy), dim=-2)
            norm_x = th.cat((norm_x, 
                             (ids_keep_xy/self.num_patch_per_row-0.5)),
                            dim=-1)
        elif self.pos_addSScatXY: # sinusoid and xy
            expand_pos_embed = self.pos_embed.repeat(B, 1, 1)  # (B, 1+num_patches, embed_dim)
            expand_ids_keep = ids_keep[..., None].expand(-1, -1, self.embed_dim)  # (B, len_keep, embed_dim)
            norm_x = norm_x + th.cat((expand_pos_embed[:, :1, :],
                                        th.gather(input=expand_pos_embed[:, 1:, :], dim=1, index=expand_ids_keep)),
                                        dim=1)
            expand_ids_keep = ids_keep[..., None]
            ids_keep_xy = th.cat((expand_ids_keep % self.num_patch_per_row, expand_ids_keep // self.num_patch_per_row), dim=-1) + 1
            ids_keep_xy = th.cat((th.zeros_like(ids_keep_xy[:, :1, :]), ids_keep_xy), dim=-2)  # cls token use (0,0)
            norm_x = th.cat((norm_x, ids_keep_xy/self.num_patch_per_row), dim=-1)
        elif self.pos_addSScatXYMap:
            expand_pos_embed = self.pos_embed.repeat(B, 1, 1)  # (B, 1+num_patches, embed_dim)
            expand_ids_keep = ids_keep[..., None].expand(-1, -1, self.embed_dim)  # (B, len_keep, embed_dim)
            norm_x = norm_x + th.cat((expand_pos_embed[:, :1, :],
                                        th.gather(input=expand_pos_embed[:, 1:, :], dim=1, index=expand_ids_keep)),
                                        dim=1)
            expand_ids_keep = ids_keep[..., None]
            ids_keep_xy = th.cat((expand_ids_keep % self.num_patch_per_row, expand_ids_keep // self.num_patch_per_row), dim=-1) + 1
            ids_keep_xy = th.cat((th.zeros_like(ids_keep_xy[:, :1, :]), ids_keep_xy), dim=-2)  # cls token use (0,0)
            norm_x = th.cat((norm_x, 
                             (ids_keep_xy/self.num_patch_per_row-0.5)),
                            dim=-1)
        elif self.pos_none:
            pass
        
        if self.feat_cls:
            feat = norm_x[:, 0, :]  # shape(B, embed_dim)
        elif self.feat_all:
            feat = norm_x
        elif self.feat_x:
            feat = norm_x[:, 1:, :]
        elif self.feat_norm_avg:  # NOTE: finetune feat_pool_norm in spirl.models
            feat = norm_x[:, 1:, :].mean(dim=-2) # global pool without cls token
            # feat = self.feat_pool_norm(norm_x[:, 1:, :].mean(dim=-2)) # global pool without cls token
        elif self.feat_avg:
            feat = norm_x[:, 1:, :].mean(dim=-2) # global pool without cls token
        elif self.feat_norm_avg_cls:  # TODO: finetune feat_pool_norm?
            raise NotImplementedError
            feat = th.cat((self.feat_pool_norm(norm_x[:, 1:, :].mean(dim=-2)),
                         norm_x[:, 0, :]), dim=-1) # shape(B, embed_dim*2)
        elif self.feat_avg_cls:
            feat = th.cat((norm_x[:, 1:, :].mean(dim=-2),
                         norm_x[:, 0, :]), dim=-1) # shape(B, embed_dim*2)
        elif self.feat_norm_max:  # NOTE: finetune feat_pool_norm in spirl.models
            feat =th.max(norm_x[:, 1:, :],dim=-2)[0] # global pool without cls token
            # feat = self.feat_pool_norm(th.max(norm_x[:, 1:, :],dim=-2)[0]) # global pool without cls token
        elif self.feat_max:
            feat = th.max(norm_x[:, 1:, :],dim=-2)[0] # global pool without cls token
        elif self.feat_norm_max_cls:  # TODO: finetune feat_pool_norm?
            raise NotImplementedError
            feat = th.cat((self.feat_pool_norm(th.max(norm_x[:, 1:, :],dim=-2)[0]),
                         norm_x[:, 0, :]), dim=-1) # shape(B, embed_dim*2)
        elif self.feat_max_cls:
            feat = th.cat((th.max(norm_x[:, 1:, :],dim=-2)[0],
                         norm_x[:, 0, :]), dim=-1) # shape(B, embed_dim*2)
        elif self.feat_norm_sum:  # NOTE: finetune feat_pool_norm in spirl.models
            feat = th.sum(norm_x[:, 1:, :], dim=-2) # global pool without cls token
            # feat = self.feat_pool_norm(th.sum(norm_x[:, 1:, :], dim=-2)) # global pool without cls token
        elif self.feat_sum:
            feat = th.sum(norm_x[:, 1:, :], dim=-2) # global pool without cls token
        elif self.feat_norm_sum_cls:  # TODO: finetune feat_pool_norm?
            raise NotImplementedError
            feat = th.cat((self.feat_pool_norm(th.sum(norm_x[:, 1:, :], dim=-2)),
                         norm_x[:, 0, :]), dim=-1) # shape(B, embed_dim*2)
        elif self.feat_sum_cls:
            feat = th.cat((th.sum(norm_x[:, 1:, :], dim=-2),
                         norm_x[:, 0, :]), dim=-1) # shape(B, embed_dim*2)
        elif self.feat_avg_max_cls:
            feat = th.cat((norm_x[:, 1:, :].mean(dim=-2),
                         th.max(norm_x[:, 1:, :],dim=-2)[0],
                         norm_x[:, 0, :]), dim=-1) # shape(B, embed_dim*3)
        elif self.feat_avg_max:
            feat = th.cat((norm_x[:, 1:, :].mean(dim=-2),
                         th.max(norm_x[:, 1:, :],dim=-2)[0]), dim=-1) # shape(B, embed_dim*3)
        elif self.feat_sum_max_cls:
            feat = th.cat((th.sum(norm_x[:, 1:, :], dim=-2),
                         th.max(norm_x[:, 1:, :],dim=-2)[0],
                         norm_x[:, 0, :]), dim=-1) # shape(B, embed_dim*2)
        elif self.feat_sum_max:
            feat = th.cat((th.sum(norm_x[:, 1:, :], dim=-2),
                         th.max(norm_x[:, 1:, :],dim=-2)[0]), dim=-1) # shape(B, embed_dim*2)
        elif self.feat_concat:
            # NOTE: different patch order will get different output!
            feat = th.flatten(norm_x[:, 1:, :], start_dim=-2, end_dim=-1)  # B, len_keep * embed_dim

        return feat

    # def forward(self, x: th.Tensor, kp_pos=None):
    #     feature = self.forward_one_frame(x, kp_pos)
    #     return feature

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3]
        assert imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = th.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x


def _init_vit_weights(module: nn.Module, name: str='', head_bias: float=0.):
    # same with mae's initialization
    if isinstance(module, nn.Linear):
        th.nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)

