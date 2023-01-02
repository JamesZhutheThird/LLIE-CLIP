import os
import time
from itertools import product

import cv2
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

from utils_plots import *

# 将几种方法的效果图画在同一张图上

set_name,method_list,path_list = get_names_adv1()
img_names = pick_images(path_list[0],n=3)


def plot_hist(img_names):
    save_dir = f'../Figs/{set_name}-hist'
    os.makedirs(save_dir,exist_ok=True)
    
    fig = plt.figure(figsize=(12,3))
    M = 1
    N = len(path_list)-2
    
    ori_pixels = []
    gt_pixels = []
    
    for i,img_name in enumerate(img_names):
        ori_img = cv_imread(os.path.join(path_list[0],img_name),cv2.COLOR_BGR2GRAY)
        gt_img = cv_imread(os.path.join(path_list[-1],img_name),cv2.COLOR_BGR2GRAY)
        ori_pixels.append(ori_img.reshape(-1))
        gt_pixels.append(gt_img.reshape(-1))
        
    ori_pixels = np.concatenate(ori_pixels,axis=0)
    gt_pixels = np.concatenate(gt_pixels,axis=0)
        
    for j,set_dir in enumerate(path_list[1:-1]):
        pixels = []
        print('Plotting hist of '+set_dir+'...')
        
        for i,img_name in enumerate(img_names):        
            if img_name not in os.listdir(set_dir):
                img = cv_imread(os.path.join(set_dir,ddprun_namefix(img_name)))
            else:
                img = cv_imread(os.path.join(set_dir, img_name))
            pixels.append(img.reshape(-1))
        pixels = np.concatenate(pixels,axis=0)
        
        title=method_list[j+1]
        
        hist_kwargs = {'bins':64,"alpha":0.3,"density":True}
        plt.subplot(M,N,j+1)
        plt.hist(ori_pixels,label='Origin',**hist_kwargs)
        plt.hist(pixels,label='Processed',**hist_kwargs)
        plt.hist(gt_pixels,label='Truth',**hist_kwargs)
        if j == len(path_list[1:-1]) - 1:
            plt.legend()
        
        xticks = [0,63,127,191,255]
        plt.xticks(xticks,[str(t) for t in xticks])
        plt.xlim(0,255)
        plt.ylim(0,0.025)
        plt.grid()
        plt.title(title)
        
    plt.tight_layout()
    plt.savefig(f'{save_dir}/hist_comparision_{int(time.time())}', dpi=300)
    plt.close()
    
if __name__ == '__main__':
    plot_hist(img_names)