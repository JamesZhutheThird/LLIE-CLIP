import os
import sys
import cv2
import numpy as np
import torch
import argparse
import torch.utils
import torch.backends.cudnn as cudnn
from PIL import Image
from torch.autograd import Variable
from model import Finetunemodel
from metrics import *

from multi_read_data import MemoryFriendlyLoader

parser = argparse.ArgumentParser("SCI")
parser.add_argument('--data_path', type=str, default='./data/medium',
                    help='location of the data corpus')
parser.add_argument('--save_path', type=str, default='./results/medium', help='path to save results')
parser.add_argument('--model', type=str, default='./weights/medium.pt', help='model path')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--seed', type=int, default=2, help='random seed')

args = parser.parse_args()
save_path = args.save_path
os.makedirs(save_path, exist_ok=True)

TestDataset = MemoryFriendlyLoader(img_dir=args.data_path, task='test')

test_queue = torch.utils.data.DataLoader(
    TestDataset, batch_size=1,
    pin_memory=True, num_workers=0)


def save_images(image_numpy, path):
    im = Image.fromarray(np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8'))
    im.save(path, 'png')

def save_ori_images(image_numpy, path):
    im = Image.fromarray(np.clip(image_numpy, 0, 255.0).astype('uint8'))
    im.save(path, 'png')

def singleScaleRetinex(img, sigma):

    retinex = np.log10(img) - np.log10(cv2.GaussianBlur(img, (0, 0), sigma))

    return retinex

def multiScaleRetinex(img, sigma_list):
    retinex = np.zeros_like(img, dtype=float)
    for sigma in sigma_list:
        retinex += singleScaleRetinex(img, sigma)

    retinex = retinex / len(sigma_list)

    return retinex

def clahe(mri_img):
    lab = cv2.cvtColor(mri_img, cv2.COLOR_BGR2LAB)
    lab_planes = list(cv2.split(lab))

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return bgr

def main():
    with torch.no_grad():
        for _, (input, image_name) in enumerate(test_queue):
            # r = multiScaleRetinex(cv2.imread(image_name[0]), [15., 80., 200.]) * 255
            r = torch.tensor(clahe(cv2.imread(image_name[0])))
            # calculate metrics
            m1 = calc_ssim(r.permute(2, 0, 1) / 255, input[0])  # in metric calculation, axis 0 should be RGB channels
            m2 = calc_psnr(r.permute(2, 0, 1) / 255, input[0])
            m3 = calc_eme(r.permute(2, 0, 1) / 255)
            m4 = calc_loe(r.permute(2, 0, 1) / 255, input[0])

            image_name = image_name[0].split('\\')[-1].split('.')[0]

            u_name = '%s.png' % (image_name)
            print('processing {}'.format(u_name))
            u_path = save_path + '/' + u_name
            # save_images(r, u_path)
            save_ori_images(r, u_path)



if __name__ == '__main__':
    main()
