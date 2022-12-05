import torch
import numpy as np
from random import randint
from skimage.metrics import structural_similarity as ssim
from skimage.util import view_as_blocks

# full-reference metrics

def calc_psnr(out_img, gt_img):
    """
    Calculate PSNR of the output image and the label image. Support batched or single data.
    Return a list whose elements are tensors, i.e., the calculated PSNR.
    """
    out_img = out_img.cpu()
    gt_img = gt_img.cpu()
    assert out_img.shape == gt_img.shape
    if len(out_img.shape) > 3:
        batched = True
    else:
        batched = False
    if batched:
        result = []
        for b in range(out_img.shape[0]):
            b_out_img = out_img[b]
            b_gt_img = gt_img[b]
            mse = torch.mean((b_out_img - b_gt_img) ** 2)
            if mse == 0:
                result.append(torch.tensor(100.))
            else:
                result.append(20 * torch.log10(255. / torch.sqrt(mse)))
    else:
        mse = torch.mean((out_img - gt_img) ** 2)
        if mse == 0:
            return [torch.tensor(100.)]
        else:
            return [20 * torch.log10(255 / torch.sqrt(mse))]

    return result


def calc_ssim(out_img, gt_img):
    """
    Calculate SSIM. Support batched or single data.
    Return a list whose elements are tensors, i.e., the calculated SSIM.
    """
    out_img = out_img.cpu().numpy()
    gt_img = gt_img.cpu().numpy()
    assert out_img.shape == gt_img.shape
    if len(out_img.shape) > 3:
        batched = True
    else:
        batched = False
    if batched:
        result = []
        for b in range(out_img.shape[0]):
            b_out_img = out_img[b]
            b_gt_img = gt_img[b]
            mean_ssim = ssim(b_out_img, b_gt_img, channel_axis=0)
            result.append(torch.tensor(mean_ssim))
    else:
        mean_ssim = ssim(out_img, gt_img, channel_axis=0)
        return [torch.tensor(mean_ssim)]


# no-reference metrics

def calc_eme(out_img):
    """
    Calculate EME(Enhancement Measure Ealuation). EME shows the contrast of an image. Support batched or single data.
    Return a list whose elements are tensors, i.e., the calculated EME.
    """
    out_img = out_img.cpu().numpy()
    if len(out_img.shape) > 3:
        batched = True
    else:
        batched = False
    if batched:
        result = []
        for b in range(out_img.shape[0]):
            eme_list = []
            for channel in range(out_img.shape[1]):
                cur_channel = out_img[b][channel]
                block_row = cur_channel.shape[0] // 8
                block_column = cur_channel.shape[1] // 8
                blocks = view_as_blocks(cur_channel, block_shape=(block_row, block_column))
                for r in range(block_row):
                    for c in range(block_column):
                        eme_list.append(20 * np.log10(blocks[r, c].max() / blocks[r, c].min()))
            result.append(torch.tensor(np.mean(eme_list)))
        return result
    else:
        eme_list = []
        for channel in range(out_img.shape[0]):
            cur_channel = out_img[channel]
            block_row = 8
            block_column = 8
            blocks = view_as_blocks(cur_channel, block_shape=(block_row, block_column))
            for r in range(blocks.shape[0]):
                for c in range(blocks.shape[1]):
                    eme_list.append(20 * np.log10(blocks[r, c].max() / (blocks[r, c].min() + 1e-6)))
        return [torch.tensor(np.mean(eme_list))]


def calc_loe(out_img, ori_img):
    """
    Calculate LOE(Lightness Order Error). Support batched or single data.
    Return a list whose elements are tensors, i.e., the calculated LOE.
    """
    out_img = out_img.cpu().numpy()
    ori_img = ori_img.cpu().numpy()
    if len(out_img.shape) > 3:
        batched = True
    else:
        batched = False
    if batched:
        result = []
        for b in range(out_img.shape[0]):
            error = 0
            for channel in range(out_img.shape[1]):
                cur_channel = out_img[b][channel]
                ori_channel = ori_img[b][channel]
                sample_num = 500
                half_patch_size = 15
                for s in range(sample_num):
                    r = randint(0, out_img.shape[1] - 1)
                    c = randint(0, out_img.shape[2] - 1)
                    w_min, w_max = max(0, r - half_patch_size), min(out_img.shape[1] - 1, r + half_patch_size)
                    h_min, h_max = max(0, c - half_patch_size), min(out_img.shape[2] - 1, c + half_patch_size)
                    pixel_value = cur_channel[r, c]
                    ori_pixel_value = ori_channel[r, c]
                    error_map = (pixel_value > cur_channel) ^ (ori_pixel_value > ori_channel)
                    error += np.sum(error_map)
            result.append(torch.tensor(error / (sample_num * 3)))
        return result
    else:
        error = 0
        for channel in range(out_img.shape[0]):
            cur_channel = out_img[channel]
            ori_channel = ori_img[channel]
            sample_num = 500
            half_patch_size = 15
            for s in range(sample_num):
                r = randint(0, out_img.shape[1] - 1)
                c = randint(0, out_img.shape[2] - 1)
                w_min, w_max = max(0, r - half_patch_size), min(out_img.shape[1] - 1, r + half_patch_size)
                h_min, h_max = max(0, c - half_patch_size), min(out_img.shape[2] - 1, c + half_patch_size)
                pixel_value = cur_channel[r, c]
                ori_pixel_value = ori_channel[r, c]
                error_map = (pixel_value > cur_channel[w_min:w_max, h_min:h_max]) ^ (ori_pixel_value > ori_channel[w_min:w_max, h_min:h_max])
                error += np.sum(error_map)
        return [torch.tensor(error / (sample_num * 3))]
