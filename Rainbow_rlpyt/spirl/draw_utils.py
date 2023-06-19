import random
import colorsys
import copy
import cv2
import pdb
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def cv2_draw_grid(data, d_row, d_col=None, color=(0, 255, 0), thickness=1):
    '''
    img: cv2 image, HWC
    color: line color
    '''
    img = copy.deepcopy(data.astype(np.uint8)).copy() # here the 2nd copy is to make the space continuous
    if d_col == None:
        d_col = d_row
    h, w, c = img.shape
    if c == 1:
        img = np.repeat(img, repeats=3, axis=-1)
    assert h % d_row == 0
    assert w % d_col == 0
    rows = h // d_row
    cols = h // d_col
    # dy, dx = h / rows, w / cols

    # draw vertical lines
    for x in np.linspace(start=d_col, stop=w-d_col, num=cols-1):
        x = int(round(x))
        cv2.line(img, (x, 0), (x, h), color=color, thickness=thickness)

    # draw horizontal lines
    for y in np.linspace(start=d_row, stop=h-d_row, num=rows-1):
        y = int(round(y))
        cv2.line(img, (0, y), (w, y), color=color, thickness=thickness)

    return img


def mark_img_loss_gray(ori_img, loss_mat, mark_ratio, grid_size):
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


def random_colors(N, bright=True):
    """
    Generate random colors.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = image[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
    return image


def display_instances(image, mask, blur=False, alpha=0.5):
    if len(mask.shape) == 2:
        mask = mask[None, :, :]
        N = 1
    else:
        assert len(mask.shape) == 3
        N = mask.shape[0]
    # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]
        _mask = mask[i]
        if blur:
            _mask = cv2.blur(_mask,(10,10))
        # Mask
        masked_image = apply_mask(masked_image, _mask, color, alpha)
    return masked_image


def ax_plot(image, ax, title=None):
    if len(image.shape) == 3:
        # image shape should be like (H, W, C)
        assert image.shape[-1] in [1, 3, 4]
    elif len(image.shape) != 2:
        raise NotImplementedError

    height, width = image.shape[:2]
    margin = 0
    ax.set_ylim(height + margin, -margin)
    ax.set_xlim(-margin, width + margin)
    ax.axis('off')
    ax.set_aspect('equal')
    ax.set_box_aspect(1)
    if title is not None:
        ax.set_title(title)
    ax.imshow(image, aspect='auto')


def ax_hist(x, ax, title=None, bin_range=None, n_bins=100):
    # ax.set_aspect('equal')
    # ax.set_box_aspect(1)
    ax.grid()
    ax.set_title(f'{title}\nsum:{x.sum():.3e}, min:{x.min():.3e}, max:{x.max():.3e}')
    n, bins, _ = ax.hist(x=x, bins=n_bins, density=False, range=bin_range)


def ax_line_plot(y, ax, title=None, marker=None, label=None, is_twin=False, color='b', lw=2.5, norm_x=False):
    if is_twin:
        ax2 = ax.twinx()
        if norm_x:
            x = np.range(0, 1.00000001, y.shape[-1])
            ax2.plot(x, y, marker=marker, label=label, color=color, lw=lw)
        else:
            ax2.plot(y, marker=marker, label=label, color=color, lw=lw)
    else:
        ax.grid()
        if title is not None:
            ax.set_title(f'{title}\nsum:{y.sum():.3e}, min:{y.min():.3e}, max:{y.max():.3e}')
        if norm_x:
            x = np.arange(0, 1.00000001, 1.0/(y.shape[-1]-1))
            ax.plot(x, y, marker=marker, label=label, color=color, lw=lw)
        else:
            ax.plot(y, marker=marker, label=label, color=color, lw=lw)


@torch.no_grad()
def draw_mask(keep_id, model, mask_shape, ori_img, ax, title=None):
    mask = np.zeros(mask_shape)
    for id_patch in keep_id:
        row = id_patch // model.num_patch_per_row
        col = id_patch % model.num_patch_per_row
        mask[row][col] = 1.0
    mask = nn.functional.interpolate(torch.tensor(mask).unsqueeze(0).unsqueeze(0),
            scale_factor=(model.patch_size, model.patch_size),
            mode="nearest")[0][0].cpu().numpy()
    masked_img = display_instances(image=ori_img, mask=mask)
    ax_plot(image=masked_img, ax=ax, title=title)

