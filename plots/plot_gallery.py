import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

from utils_plots import *

# 将几种方法的效果图画在同一张图上

# 总体结果配图参数设置
set_name,method_list,path_list = get_names_adv2()
ratio=(3,2)
img_names = fix_images2()

# 消融实验配图参数设置
# set_name,method_list,path_list = get_names_abl()
# ratio=(2,3)
# img_names = fix_images1()

M = 1
N = len(path_list)

def plot_gallery_single(img_names):
    save_dir = f'../Figs/{set_name}-gallery'
    os.makedirs(save_dir,exist_ok=True)

    M = 1
    N = len(path_list)
    for i,img_name in enumerate(img_names):
        print('Plotting gallery of '+img_name+'...')
        fig = plt.figure(figsize=(8,4))
        
        for j,set_dir in enumerate(path_list):
            img = cv_imread(os.path.join(set_dir, img_name))
            title=method_list[j]
            #行，列，索引
            plt.subplot(M,N,j+1)
            plt.imshow(img)
            plt.title(title,fontsize=20)
            plt.xticks([])
            plt.yticks([])
        plt.tight_layout()
        plt.savefig(f'{save_dir}/{img_name}', dpi=300)
        plt.close()
        
def plot_gallery_multi(img_names):
    save_dir = f'../Figs/{set_name}-gallery'
    os.makedirs(save_dir,exist_ok=True)
    M = len(img_names)
    N = len(path_list)
    fig = plt.figure(figsize=(ratio[1]*N,ratio[0]*M)) 
    
    for i,img_name in enumerate(img_names):
        print('Plotting gallery of '+img_name+'...')
        for j,set_dir in enumerate(path_list):
            if img_name not in os.listdir(set_dir):
                img = cv_imread(os.path.join(set_dir,ddprun_namefix(img_name)))
            else:
                img = cv_imread(os.path.join(set_dir, img_name))
            title=method_list[j]
            #行，列，索引
            plt.subplot(M,N,i*N+j+1)
            plt.imshow(img)
            if i == 0:
                plt.title(title,fontsize=20)
            plt.xticks([])
            plt.yticks([])
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{set_name}_{int(time.time())}', dpi=200)
    plt.close()

        
if __name__ == '__main__':
    plot_gallery_multi(img_names)
