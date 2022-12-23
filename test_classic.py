import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.utils
from PIL import Image
from torch.autograd import Variable

from metrics import *
from model import Finetunemodel
from multi_read_data import MemoryFriendlyLoader

# PC有中文路径，以此解决中文路径读取问题
def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path,dtype=np.uint8),-1)
    return cv_img

def save_images(image_numpy, path):
    im = Image.fromarray(np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8'))
    im.save(path, 'png')

def save_ori_images(image_numpy, path):
    im = Image.fromarray(np.clip(image_numpy, 0, 255.0).astype('uint8'))
    im.save(path, 'png')

def singleScaleRetinex(img, sigma):
    thrd = 1
    retinex = np.log10(np.clip(img,thrd,None)) 
    - np.log10(np.clip(cv2.GaussianBlur(img, (0, 0), sigma),thrd,None))

    return retinex

def multiScaleRetinex(img, sigma_list):
    retinex = np.zeros_like(img, dtype=float)
    for sigma in sigma_list:
        retinex += singleScaleRetinex(img, sigma)

    retinex = retinex / len(sigma_list)

    retinex = (retinex - np.min(retinex)) / (np.max(retinex) - np.min(retinex))

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
        metrics_df = None
        
        for _, (input, gt, image_name) in enumerate(test_queue):
            # image_name (batch, list of str)里只有个文件名，改成路径，可能和Windows系统有关
            image_file = os.path.join(args.data_path, image_name[0])
            
            if args.mode == 'retinex':
                r = torch.Tensor(multiScaleRetinex(cv_imread(image_file), [15., 80., 200.,]) * 255)
            elif args.mode == 'clahe':
                r = torch.tensor(clahe(cv_imread(image_file)))
            else:
                raise NotImplementedError
            
            metrics_batch = get_metrics(image_name, r.permute(2, 0, 1) / 255, input[0], gt[0])
            if metrics_df is not None:
                metrics_df = pd.concat([metrics_df, metrics_batch],axis=0)
            else:
                metrics_df = metrics_batch

            image_name = image_name[0].split('\\')[-1].split('.')[0]

            u_name = '%s.png' % (image_name)
            print('processing {}'.format(u_name))
            u_path = os.path.join(save_path, u_name)
            # save_images(r, u_path)
            # 参数要求图像是ndarray
            save_ori_images(r.numpy(), u_path)
            
        metrics_df.to_csv(f"{save_path}/metrics_{args.mode}.csv")

parser = argparse.ArgumentParser("SCI")
parser.add_argument('--data',type=str,default='lol',help='name of the dataset')
parser.add_argument('--mode',type=str,default='clahe',help='method used')
# parser.add_argument('--data_path', type=str, default='./data/lol/low',
#                     help='location of the data corpus')
# parser.add_argument('--gt_path', type=str, default='./data/lol/high')
# parser.add_argument('--save_path', type=str, default='./results/lol', help='path to save results')
parser.add_argument('--model', type=str, default='./weights/medium.pt', help='model path')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--seed', type=int, default=2, help='random seed')

args = parser.parse_args()

if args.data in ['all','difficult','medium','easy']:
    args.data_path = './data/'+ args.data
    args.gt_path = None
elif args.data in ['gladnet','lol']:
    args.data_path = './data/'+ args.data + '/low'
    args.gt_path = './data/' + args.data + '/high' 
else:
    raise NotImplementedError
args.save_path = './results/'+args.data+'-'+args.mode

save_path = args.save_path
os.makedirs(save_path, exist_ok=True)

TestDataset = MemoryFriendlyLoader(img_dir=args.data_path, task='test', gt_dir=args.gt_path)

test_queue = torch.utils.data.DataLoader(
    TestDataset, batch_size=1,
    pin_memory=True, num_workers=0)


if __name__ == '__main__':
    main()
