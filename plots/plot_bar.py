import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

from utils_plots import *

set_name,method_names,path_list = get_names_ablbar()
log_files = get_log_files(path_list)
reverse_metrics = ['LOE']
save_dir = '../Figs/bars/'
os.makedirs(save_dir,exist_ok=True)

def read_stat_for_bars(log_file):
    # print(log_file)
    df = pd.read_csv(log_file,index_col=0)
    for m in reverse_metrics:
        df[m] = 1 / df[m]
    
    stats = np.mean(df.values,axis=0)
    return stats

def plot_bars():
    metrics = ['EME','LOE','PSNR','SSIM']
    colors = ['C0','C1','C2','C4']
    stats = np.zeros((len(log_files),len(metrics)))
    for i,log_file in enumerate(log_files):
        stats[i,:] = read_stat_for_bars(log_file)
    
    M = 2
    N = 2
    
    fig = plt.figure(figsize=(6,4))
    
    # x = [1,2,3,4,6,7,8,9]
    x = [0,1,2,3,4,6,7,8,9,10]
    for j in range(len(metrics)):
        plt.subplot(M,N,j+1)
        
        patches = []
        for k,v in enumerate(stats[:,j]):
            if stats[k,j] == max(stats[(k//5)*5:(k//5+1)*5,j]):
                patches.append('//')
            else:
                patches.append(None)
                
        plt.bar(x,stats[:,j],1,alpha=0.5,color=colors[j],edgecolor=colors[j],linewidth=2,hatch=patches)
        plt.title(metrics[j])
        plt.xlabel('SCI              CLIP')
        plt.xticks(x,method_names)
        plt.xlim(-1,11)
        plt.grid(linewidth=0.5)
        
    plt.tight_layout()
    plt.savefig(f'{save_dir}/bars_{int(time.time())}', dpi=300)
    plt.close()
    
if __name__ == '__main__':
    plot_bars()