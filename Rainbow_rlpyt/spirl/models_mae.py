# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial
from itertools import product
import pdb
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from timm.models.vision_transformer import PatchEmbed, Mlp, DropPath

from .pos_embed import get_2d_sincos_pos_embed


"""
Modify Attention & Block same with DINO, to output the intermediate attention map.
NOTE: pay attention that the Attention and Block from timm0.3.2
"""
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape  # batchsize, #patches, embed_dim
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # qkv.shape: (3, B, #heads, #patches, embed_dim // #heads)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, #heads, #patches, #patches)
        attn = attn.softmax(dim=-1)  # normalize for each query patch seperately
        attn = self.attn_drop(attn)
        # (B, #heads, #patches, embed_dim // #heads)
        # --> (B, #patches, #heads, embed_dim // #heads)
        # --> (B, #patches, embed_dim)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 debug=False):
        super().__init__()
        self.debug = debug
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_depth = decoder_depth
        self.decoder_num_heads = decoder_num_heads

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.num_patch_per_row = self.patch_embed.img_size[1] // self.patch_embed.patch_size[1]

        # cls_token (class token) is an auxiliary dummy token to the encoder input
        # will be treated as the class token for training the classifier in linear probing and fine-tuning
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        # TODO: pos_embed is fixed, can not be save and reload!

        self.blocks = nn.ModuleList([  # attention block
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)  # layer norm over the last embed dim
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def load_trained_mae(self, checkpoint_model: dict):
        msg = self.load_state_dict(checkpoint_model, strict=False)
        print(f"{'*'*10}checkpoint load information for pretrained mae model{'*'*10}")
        print(msg)

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_encoder_last_selfattention(self, x):
        # simplified version
        x = self.patch_embed(x)  # shape (num_batches, num_patches, embed_dim)
        B = x.shape[0]
        assert B == 1
        # append cls token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # shape (num_batches, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # shape (num_batches, 1 + len_keep, embed_dim)
        # add pos embed
        x = x + self.pos_embed  # shape (num_batches, num_patches+1, embed_dim)

        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)

    def show_patched_image(self, images, kp_pos_ls):
        # This is a utility function which accepts a batch of images and its
        # corresponding patches and help visualize one image and its patches
        # side by side.
        # TODO: if each patch is small (i.e. there are many patches in one frame,
        #       we may unmask 9 patches around the keypoints instead of focus only on one)
        print('Show patched image')
        idx = np.random.choice(images.shape[0])
        images[idx] = images[idx]
        patches = self.patchify(images[idx].unsqueeze(0))[0]  # shape: (h * w, p**2 * 3)
        print(f"Index selected: {idx}.")

        plt.figure(figsize=(4, 4))
        plt.title("original image with keypoints")
        plt.imshow(torch.einsum('chw->hwc', images[idx]))
        # for points in keypoints_ls[idx]:
        #     plt.plot(points[0], points[1], 'bs')
        # # plt.axis("off")
        # plt.show()

        n = int(np.sqrt(patches.shape[0]))
        plt.figure(figsize=(4, 4))
        plt.title("patches of original images")
        for i, patch in enumerate(patches):
            ax = plt.subplot(n, n, i + 1)
            patch_img = torch.reshape(patch, (self.patch_embed.patch_size[0], self.patch_embed.patch_size[1], 3))
            plt.imshow(patch_img)
            plt.axis("off")
        for point in kp_pos_ls[idx]:
            plt.subplot(n, n, point[1] * n + point[0] + 1)
            plt.plot(0, 0, 'bs')  # annotate that patch in left up corner
            plt.axis("off")
        plt.show()

        # Return the index chosen to validate it outside the method.
        return idx

    def show_mask(self, image, kp_pos, mask, kp_reverse, loss_all):
        patches = self.patchify(image.unsqueeze(0))[0]  # shape: (h * w, p**2 * 3)
        n = int(np.sqrt(patches.shape[0]))
        plt.figure(figsize=(5, 5))
        plt.title("mask patches of original images")
        for i, patch in enumerate(patches):
            ax = plt.subplot(n, n, i + 1)
            patch_img = torch.reshape(patch, (self.patch_embed.patch_size[0], self.patch_embed.patch_size[1], 3))
            # pdb.set_trace()
            if mask[i] == 0.0:  # keep
                plt.imshow(patch_img)
            else:  # mask
                plt.imshow(patch_img.mean(axis=2), cmap='gray')
            plt.axis("off")
        plt.show()

        for point in kp_pos:
            # plt.subplot(n, n, point[1] * n + point[0] + 1)
            if kp_reverse:
                assert mask[point[1] * n + point[0]] != 0
            elif not loss_all:
                assert mask[point[1] * n + point[0]] == 0

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
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio, kp_pos_ls=None, loss_all=False,
                       kp_reverse=False, kp_ext_loss_weight=0.1, kp_ext_loss_multi=False):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        kp_pos_ls: if not None, will select masked patches according to it; otherwise will mask patches randomly
        kp_reverse: if kp_pos_ls is not None, kp_reverse=True will mask patches with keypoints and weight the loss for those patches                                  
        loss_all: if True, also backward loss on unmaskes patches
        """
        N, L, D = x.shape  # batch, length (num_patches), dim
        len_keep = int(L * (1 - mask_ratio))
        if kp_pos_ls is not None:
            if kp_reverse:  # mask key patches  NOTE: lost necessary information for reconstruction, recommend not to use.
                if mask_ratio <= 1:
                    len_keep = min(max(L - kp_pos_ls.shape[1], 0), len_keep)
                else:
                    len_keep = max(L - kp_pos_ls.shape[1], 0)
            else:
                if mask_ratio <= 1:
                    len_keep = min(max(kp_pos_ls.shape[1], len_keep), L)
                else:
                    len_keep = min(kp_pos_ls.shape[1], L)
        # print(f'len_keep: {len_keep}')
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        if kp_pos_ls is not None and len_keep < L:
            if not kp_reverse:
                if len_keep < kp_pos_ls.shape[1]:
                    print("the number of kept patches need to >= number of keypoints")
            else:
                if L - len_keep >= kp_pos_ls.shape[1]:
                    print("the number of masked patches need to >= number of keypoints")
            # obtain the id of keypoint patches
            # pos_y * (W / patch_size) + pos_x
            flat_kp_pos_ls = kp_pos_ls[..., 1] * self.num_patch_per_row  + kp_pos_ls[..., 0]  # (num_batch, num_keypoints)
            if torch.max(flat_kp_pos_ls).item() >= self.num_patch_per_row **2:
            # if self.debug:
                print(f'torch.max(flat_kp_pos_ls).item(): {torch.max(flat_kp_pos_ls).item()}')
                print(f'torch.min(flat_kp_pos_ls).item(): {torch.min(flat_kp_pos_ls).item()}')
                # print(f'keypoints_ls[torch.where(flat_kp_pos_ls==torch.max(flat_kp_pos_ls).item())]: {keypoints_ls[torch.where(flat_kp_pos_ls==torch.max(flat_kp_pos_ls).item())]}')
                # print(f'keypoints_ls[torch.where(flat_kp_pos_ls==torch.min(flat_kp_pos_ls).item())]: {keypoints_ls[torch.where(flat_kp_pos_ls==torch.min(flat_kp_pos_ls).item())]}')
                print(f'kp_pos_ls[torch.where(flat_kp_pos_ls==torch.max(flat_kp_pos_ls).item())]: {kp_pos_ls[torch.where(flat_kp_pos_ls==torch.max(flat_kp_pos_ls).item())]}')
                print(f'kp_pos_ls[torch.where(flat_kp_pos_ls==torch.min(flat_kp_pos_ls).item())]: {kp_pos_ls[torch.where(flat_kp_pos_ls==torch.min(flat_kp_pos_ls).item())]}')
                print(f'self.patch_size: {self.patch_size},  self.patch_embed.img_size: { self.patch_embed.img_size}')
                assert torch.max(flat_kp_pos_ls).item() < self.num_patch_per_row **2, f'torch.max(flat_kp_pos_ls).item()'
                assert torch.min(flat_kp_pos_ls).item() >= 0, f'{torch.min(flat_kp_pos_ls).item()}'
                assert noise.shape[0] == flat_kp_pos_ls.shape[0], f'{noise.shape}, {flat_kp_pos_ls.shape}'
                assert noise.shape[1] == self.num_patch_per_row ** 2, f'{noise.shape}, {self.num_patch_per_row}'
                assert noise.ndim == 2, f'{noise.ndim}'
                assert flat_kp_pos_ls.ndim == 2, f'{flat_kp_pos_ls.ndim}'
            if not kp_reverse:
                noise.scatter_(dim=1, index=flat_kp_pos_ls, src=torch.zeros(flat_kp_pos_ls.shape, dtype=noise.dtype, device=noise.device))
            else:
                noise.scatter_(dim=1, index=flat_kp_pos_ls, src=torch.ones(flat_kp_pos_ls.shape, dtype=noise.dtype, device=noise.device))

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]  # shape (num_batch, len_keep)
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))  # shape (num_batch, len_keep, dim)

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)  # check mask's dtype is float
        if not loss_all:  # only backward loss on masked patches
            mask[:, :len_keep] = 0
            # unshuffle to get the binary mask
            mask = torch.gather(mask, dim=1, index=ids_restore)  # keep - 0, mask - 1
        if kp_pos_ls is not None and kp_ext_loss_weight > 0.0:
            # assert torch.all(torch.gather(mask, index=flat_kp_pos_ls, dim=1)==1.0)
            mask.scatter_(dim=1, index=flat_kp_pos_ls, 
                          src=(1.0 + kp_ext_loss_weight) * torch.ones(flat_kp_pos_ls.shape, dtype=mask.dtype, device=mask.device))
            # assert torch.all(torch.gather(mask, index=flat_kp_pos_ls, dim=1)==(1.0 + kp_ext_loss_weight))
            # mask[flat_kp_pos_ls] += kp_ext_loss_weight

        # x_masked: (num_batch, len_keep, dim)
        # mask & ids_restore: (num_batch, num_patches)
        return x_masked, mask, ids_restore

    def forward_encoder(self, samples, mask_ratio, kp_reverse, kp_ext_loss_weight, kp_ext_loss_multi, loss_all,
                        unmask_idx, surr_mask_idx):
        # img_ls: (num_batch, C, H, W), keypoints_ls & kp_pos_ls: (num_batch, num_keypoints, 2)
        # embed patches
        x = self.patch_embed(samples[0])  # shape (num_batches, num_patches, embed_dim)
        B, P, D = x.shape[0], x.shape[1], x.shape[2]
        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]  # shape (num_batches, num_patches, embed_dim)

        # masking: length -> length * mask_ratio
        if unmask_idx is not None:
            x = x[:, unmask_idx: unmask_idx + 1, :]  # (num_batch, 1, embed_dim)
            mask = torch.ones([B, P], device=x.device)
            if not loss_all:
                mask[:, unmask_idx] = 0
            ids_restore = torch.roll(torch.arange(P, device=x.device), unmask_idx).view(1, -1).repeat(B, 1)  # (num_batches, num_patches)
        elif surr_mask_idx is not None:
            ids_restore = torch.ones([P], device=x.device, dtype=int) * (self.patch_embed.num_patches - 1)  # -1: in the input to decoder, the last one must be mask_token
            
            surr_idx_ls = []  # get surrounding patches' idx
            row, column = surr_mask_idx // self.num_patch_per_row, surr_mask_idx % self.num_patch_per_row
            for d_row, d_column in product([-1, 0, 1], [-1, 0, 1]):
                if d_row == 0 and d_column == 0:  # exclude itself
                    continue
                next_row = row + d_row
                next_column = column + d_column
                if next_row < 0 or next_row >= self.num_patch_per_row or\
                    next_column < 0 or next_column >= self.num_patch_per_row:
                    continue
                surr_idx =  next_row * self.num_patch_per_row + next_column
                ids_restore[surr_idx] = len(surr_idx_ls)  # NOTE: must put it before appending surr_idx! to keep the id position correct
                surr_idx_ls.append(surr_idx)

            ids_keep = torch.tensor(surr_idx_ls, device=x.device).view(1, -1, 1).repeat(B, 1, D)
            x = torch.gather(x, dim=1, index=ids_keep)  # shape (num_batch, len_keep, dim)
            
            ids_restore = ids_restore.view(1, -1).repeat(B, 1)  # (num_batches, num_patches)
            
            mask = torch.ones([B, P], device=x.device)
            if not loss_all:
                len_keep = len(surr_idx_ls)
                mask[:, :len_keep] = 0
                mask = torch.gather(mask, dim=1, index=ids_restore)
        else:
            x, mask, ids_restore = self.random_masking(
                x=x, mask_ratio=mask_ratio, kp_pos_ls=samples[1],
                kp_reverse=kp_reverse, kp_ext_loss_weight=kp_ext_loss_weight,
                kp_ext_loss_multi=kp_ext_loss_multi, loss_all=loss_all)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]  # shape (1, 1, embed_dim)
        cls_tokens = cls_token.expand(B, -1, -1)  # shape (num_batches, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # shape (num_batches, 1 + len_keep, embed_dim)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)  # shape (num_batches, 1 + len_keep, embed_dim)

        # x: (num_batches, 1 + len_keep, embed_dim)
        # mask & ids_restore: (num_batch, num_patches)
        return x, mask, ids_restore


    @torch.no_grad()
    def pred_encoder(self):
        B = 1  # only test 1 image at one time
        zero_embed = torch.zeros_like(self.cls_token)
        zero_embed = zero_embed.expand(B, -1, -1)

        # zero embedding without position embedding
        x = zero_embed
        # encode
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)  # shape (num_batches, 1 + len_keep, embed_dim)
        # encoder-decoder proj
        x = self.decoder_embed(x)  # shape: (num_batches, 1, decoder_embed_dim)
        # decode
        mask_tokens = self.mask_token.repeat(x.shape[0], self.patch_embed.num_patches, 1)
        x = torch.cat([x[:, :1, :], mask_tokens], dim=1)  # append cls token, shape: (num_batches, 1 + num_patches, decoder_embed_dim)
        # add pos embed
        x = x + self.decoder_pos_embed
        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)  # shape: (num_batches, 1 + num_patches, decoder_embed_dim)
        # predictor projection
        x = self.decoder_pred(x)  # shape: (num_batches, 1 + num_patches, patch_size ** 2 * in_chanels)
        # remove cls token
        x_zero = x[:, 1:, :]

        # zero embedding with position embedding
        x = zero_embed + self.pos_embed[:, :1, :]  # in fact, here the position embedding for cls token is 0
        # encode
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)  # shape (num_batches, 1 + len_keep, embed_dim)
        # decode
        x = self.decoder_embed(x)  # shape: (num_batches, 1, decoder_embed_dim)
        mask_tokens = self.mask_token.repeat(x.shape[0], self.patch_embed.num_patches, 1)
        x = torch.cat([x[:, :1, :], mask_tokens], dim=1)  # append cls token, shape: (num_batches, 1 + num_patches, decoder_embed_dim)
        # add pos embed
        x = x + self.decoder_pos_embed
        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)  # shape: (num_batches, 1 + num_patches, decoder_embed_dim)
        # predictor projection
        x = self.decoder_pred(x)  # shape: (num_batches, 1 + num_patches, patch_size ** 2 * in_chanels)
        # remove cls token
        x_zero_pos = x[:, 1:, :]

        return x_zero, x_zero_pos


    @torch.no_grad()
    def pred_decoder(self):
        B = 1  # only test 1 image at one time

        ## mask_tokens + pos_embedding
        # decode
        mask_tokens = self.mask_token.repeat(B, self.patch_embed.num_patches, 1)
        x = mask_tokens
        # add pos embed
        x = x + self.decoder_pos_embed[:, 1:, :]
        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)  # shape: (num_batches, 1 + num_patches, decoder_embed_dim)
        # predictor projection
        x = self.decoder_pred(x)  # shape: (num_batches, 1 + num_patches, patch_size ** 2 * in_chanels)
        # remove cls token
        x_mask_pos = x[:]

        ## only mask_tokens
        # decode
        mask_tokens = self.mask_token.repeat(B, self.patch_embed.num_patches, 1)
        x = mask_tokens
        # # add pos embed
        # x = x + self.decoder_pos_embed
        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)  # shape: (num_batches, 1 + num_patches, decoder_embed_dim)
        # predictor projection
        x = self.decoder_pred(x)  # shape: (num_batches, 1 + num_patches, patch_size ** 2 * in_chanels)
        # remove cls token
        x_no_pos = x[:]

        ## only decoder_pos_embed
        # decode
        decoder_pos_embed = self.decoder_pos_embed.repeat(B, 1, 1)
        x = decoder_pos_embed[:, 1:, :]
        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)  # shape: (num_batches, 1 + num_patches, decoder_embed_dim)
        # predictor projection
        x = self.decoder_pred(x)  # shape: (num_batches, 1 + num_patches, patch_size ** 2 * in_chanels)
        # remove cls token
        x_no_mask = x[:]

        return x_mask_pos, x_no_pos, x_no_mask


    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)  # shape: (num_batches, 1 + len_keep, decoder_embed_dim)

        # append mask tokens to sequence
        # pdb.set_trace()  # check if ids_restore if correct
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token, shape: (num_batches, num_patches, decoder_embed_dim)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token, shape: (num_batches, 1 + num_patches, decoder_embed_dim)

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

    def forward_loss(self, imgs, pred, mask, unmask_idx=None, surr_mask_idx=None, loss_type='L2'):
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
            loss = (pred - target) ** 2  # (num_batch, num_patch, p**2 * 3)
            loss = loss.mean(dim=-1)  # [num_batch, num_patch], mean loss per patch
        elif loss_type == 'L1':
            loss = torch.abs(pred - target)  # (num_batch, num_patch, p**2 * 3)
            loss = loss.mean(dim=-1)  # [num_batch, num_patch], mean loss per patch
        elif loss_type == 'L2trim1':
            err = (pred - target) ** 2  # (num_batch, num_patch, p**2 * 3)
            err_vals, order = torch.sort(err, dim=-1)
            loss = err_vals[:,:,:-1].mean(dim=-1)
        elif loss_type == 'L2trim2':
            err = (pred - target) ** 2  # (num_batch, num_patch, p**2 * 3)
            err_vals, order = torch.sort(err, dim=-1)
            loss = err_vals[:,:,:-2].mean(dim=-1)
        elif loss_type == 'L2trim4':
            err = (pred - target) ** 2  # (num_batch, num_patch, p**2 * 3)
            err_vals, order = torch.sort(err, dim=-1)
            loss = err_vals[:,:,:-4].mean(dim=-1)
        elif loss_type == 'L2trim6':
            err = (pred - target) ** 2  # (num_batch, num_patch, p**2 * 3)
            err_vals, order = torch.sort(err, dim=-1)
            loss = err_vals[:,:,:-6].mean(dim=-1)
        elif loss_type == 'L2trim8':
            err = (pred - target) ** 2  # (num_batch, num_patch, p**2 * 3)
            err_vals, order = torch.sort(err, dim=-1)
            loss = err_vals[:,:,:-8].mean(dim=-1)
        elif loss_type == 'L2trim10':
            err = (pred - target) ** 2  # (num_batch, num_patch, p**2 * 3)
            err_vals, order = torch.sort(err, dim=-1)
            loss = err_vals[:,:,:-10].mean(dim=-1)
        elif loss_type == 'L2trim12':
            err = (pred - target) ** 2  # (num_batch, num_patch, p**2 * 3)
            err_vals, order = torch.sort(err, dim=-1)
            loss = err_vals[:,:,:-12].mean(dim=-1)
        elif loss_type == 'L2trim14':
            err = (pred - target) ** 2  # (num_batch, num_patch, p**2 * 3)
            err_vals, order = torch.sort(err, dim=-1)
            loss = err_vals[:,:,:-14].mean(dim=-1)
        elif loss_type == 'L2trim16':
            err = (pred - target) ** 2  # (num_batch, num_patch, p**2 * 3)
            err_vals, order = torch.sort(err, dim=-1)
            loss = err_vals[:,:,:-16].mean(dim=-1)
        elif loss_type == 'L2trim18':
            err = (pred - target) ** 2  # (num_batch, num_patch, p**2 * 3)
            err_vals, order = torch.sort(err, dim=-1)
            loss = err_vals[:,:,:-18].mean(dim=-1)
        elif loss_type == 'L2trim20':
            err = (pred - target) ** 2  # (num_batch, num_patch, p**2 * 3)
            err_vals, order = torch.sort(err, dim=-1)
            loss = err_vals[:,:,:-20].mean(dim=-1)
        elif loss_type == 'L2trim22':
            err = (pred - target) ** 2  # (num_batch, num_patch, p**2 * 3)
            err_vals, order = torch.sort(err, dim=-1)
            loss = err_vals[:,:,:-22].mean(dim=-1)
        elif loss_type == 'L2trim24':
            err = (pred - target) ** 2  # (num_batch, num_patch, p**2 * 3)
            err_vals, order = torch.sort(err, dim=-1)
            loss = err_vals[:,:,:-24].mean(dim=-1)
        elif loss_type == 'L2trim26':
            err = (pred - target) ** 2  # (num_batch, num_patch, p**2 * 3)
            err_vals, order = torch.sort(err, dim=-1)
            loss = err_vals[:,:,:-26].mean(dim=-1)
        elif loss_type == 'L2trim28':
            err = (pred - target) ** 2  # (num_batch, num_patch, p**2 * 3)
            err_vals, order = torch.sort(err, dim=-1)
            loss = err_vals[:,:,:-28].mean(dim=-1)
        elif loss_type == 'L2trim30':
            err = (pred - target) ** 2  # (num_batch, num_patch, p**2 * 3)
            err_vals, order = torch.sort(err, dim=-1)
            loss = err_vals[:,:,:-30].mean(dim=-1)

        if unmask_idx is not None or surr_mask_idx is not None:
            loss_idx = loss[:, unmask_idx if unmask_idx is not None else surr_mask_idx]  # (num_batch)
        else:
            loss_idx = None

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss, loss_idx

    def forward(self, samples, mask_ratio=0.75, test_idx=None, kp_reverse=False, kp_ext_loss_weight=0.0,
                kp_ext_loss_multi=False, loss_all=False, unmask_idx=None, surr_mask_idx=None, loss_type='L2'):
        """
        samples[0]: are images
        # samples[1]: is None or keypoint coordinates (if args.use_kp),
        samples[2]: is None or keypoint patch positions (is args.use_kp)
        mask_ratio: is the ratio of masked patches
        kp_rverse: is True if mask patches that contains keypoints, is False if unmask patches that contains keypoints
        kp_ext_loss_weight: if kp_reverse is True, add extra weight to loss that corresponding to keypoint patches
        loss_all: if True backward loss for both masked/unmasked patches; if False only backward on masked patches
        unmask_idx: if not None, will keep only the unmask_idx-th patch, and the returned loss_idx is the loss for that patch
        surr_mask_idx: if not None, will keep only thr surrounding patches around the unmask_idx-th patch, and the returned loss_idx is the loss for that patch
        """
        assert not ((unmask_idx is not None) and (surr_mask_idx is not None))
        assert loss_all
        latent, mask, ids_restore = self.forward_encoder(
            samples, mask_ratio, kp_reverse=kp_reverse, kp_ext_loss_weight=kp_ext_loss_weight,
            kp_ext_loss_multi=kp_ext_loss_multi, loss_all=loss_all, unmask_idx=unmask_idx,
            surr_mask_idx=surr_mask_idx)
        if test_idx is not None:
            self.show_mask(image=samples[0][test_idx].cpu(),
                           kp_pos=samples[1][test_idx].cpu(), mask=mask[test_idx].cpu(),
                           kp_reverse=kp_reverse, loss_all=loss_all)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]  (num_batches, num_patches, patch_size ** 2 * in_chanels)
        loss, loss_idx = self.forward_loss(samples[0], pred, mask, unmask_idx=unmask_idx, surr_mask_idx=surr_mask_idx, loss_type=loss_type)
        if loss_idx is not None:
            return loss, loss_idx, pred, mask
        else:
            return loss, pred, mask


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
# NOTE: last 'two chars' need to be the patch size!, first 4 char must be 'mae_'
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
