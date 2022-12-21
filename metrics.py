import torch
import numpy as np
import pandas as pd
from random import randint
from skimage.metrics import structural_similarity as ssim
from skimage.util import view_as_blocks
import pdb

# 强制numpy在遇到warning时暂停，用于调试除零警告
from numpy import seterr
seterr(all='raise')

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
        print(out_img.shape,gt_img.shape)
        mean_ssim = ssim(out_img, gt_img, channel_axis=0)
        return [torch.tensor(mean_ssim)]


# no-reference metrics

def calc_eme(out_img):
    """
    Calculate EME(Enhancement Measure Ealuation). EME shows the contrast of an image. Support batched or single data.
    Return a list whose elements are tensors, i.e., the calculated EME.
    """
    out_img = out_img.cpu().numpy()
    
    eps = 1e-8
    BLK = 8 # block size (write in short)
    
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
                # block_row = cur_channel.shape[0] // BLK
                # block_column = cur_channel.shape[1] // BLK
                
                # view_as_blocks不接受不能够整除的情况，预先padding到能够整除
                pad_row = BLK - cur_channel.shape[0] % BLK if cur_channel.shape[0] % BLK != 0 else 0
                pad_column = BLK - cur_channel.shape[1] % BLK if cur_channel.shape[1] % BLK != 0 else 0
                cur_channel = np.pad(cur_channel,(
                    (0,pad_row),
                    (0,pad_column)
                    ),'minimum')
                
                blocks = view_as_blocks(cur_channel, block_shape=(BLK,BLK))
                # pdb.set_trace()
                # block.shape = (block_row,block_column,BLK,BLK)
                for r in range(blocks.shape[0]):
                    for c in range(blocks.shape[1]):
                        # 有可能出现全是0的情况 那对比度当然是最低的 相当于全是非0值 置0即可
                        # 有可能出现最小值是0 可能作者也是这么取eps的 因为EME指标都在很高的值
                        if blocks[r,c].max() == 0:
                            eme_value = 0
                        else:
                            eme_value = 20 * np.log10(
                                blocks[r, c].max() / (blocks[r, c].min()+eps)
                                )
                        eme_list.append(eme_value)
            result.append(torch.tensor(np.mean(eme_list)))
        return result
    else:
        eme_list = []
        for channel in range(out_img.shape[0]):
            cur_channel = out_img[channel]
            
            # view_as_blocks不接受不能够整除的情况，预先padding到能够整除
            pad_row = BLK - cur_channel.shape[0] % BLK if cur_channel.shape[0] % BLK != 0 else 0
            pad_column = BLK - cur_channel.shape[1] % BLK if cur_channel.shape[1] % BLK != 0 else 0
            cur_channel = np.pad(cur_channel,(
                (0,pad_row),
                (0,pad_column)
                ),'minimum')
                
            blocks = view_as_blocks(cur_channel, block_shape=(BLK,BLK))
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
                    r = randint(0, out_img.shape[2] - 1)
                    c = randint(0, out_img.shape[3] - 1)
                    w_min, w_max = max(0, r - half_patch_size), min(out_img.shape[2] - 1, r + half_patch_size)
                    h_min, h_max = max(0, c - half_patch_size), min(out_img.shape[3] - 1, c + half_patch_size)
                    pixel_value = cur_channel[r, c]
                    ori_pixel_value = ori_channel[r, c]
                    error_map = (pixel_value > cur_channel[w_min:w_max, h_min:h_max]) ^ (ori_pixel_value > ori_channel[w_min:w_max, h_min:h_max])
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

def get_metrics(img_name,out_img,ori_img,gt_img=None):
    """
    Generate pd.DataFrame for metrics.
    Index = img_name, columns = name_of_metrics.
    If gt_img=None, calculate no-reference metrics only.
    """
    img_count = 1
    metric_count = 4 if gt_img is not None else 2
    metrics = np.zeros((img_count,metric_count)).astype(float)
    
    metrics[:,0] = calc_eme(out_img)
    metrics[:,1] = calc_loe(out_img,ori_img)
    
    metric_names = ['EME','LOE']
    
    if gt_img is not None:
        metrics[:,2] = calc_psnr(out_img,gt_img)
        metrics[:,3] = calc_ssim(out_img,gt_img)
        metric_names.extend(['PSNR','SSIM'])
    
    metrics_df = pd.DataFrame(metrics,index=[img_name],columns=metric_names)
    
    return metrics_df