import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
from utils_plots import *

plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

set_name,method_names,path_list = get_names_adv1()
method_names = method_names[1:-1]
path_list = path_list[1:-1]
log_files = get_log_files(path_list)

base_idx = len(log_files) - 1
reverse_metrics = ['LOE']
save_dir = '../Figs/spider/'
os.makedirs(save_dir,exist_ok=True)

def read_stats_for_spider(df):
    for m in reverse_metrics:
        df[m] = 1 / df[m]
        
    labels = df.columns.values
    stats = np.mean(df.values,axis=0)
    angles=np.linspace(0, 2*np.pi, len(labels), endpoint=False)
    
    return labels,stats,angles

def plot_spider():
    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)

    yticks = np.linspace(0.0,1.0,5,endpoint=True)

    # 准备基准线数据
    df_base = pd.read_csv(log_files[base_idx],index_col=0)
    labels,stats,angles = read_stats_for_spider(df_base)

    base_stats = stats
    # 角度与指标一一对应
    for _s,_x in zip(base_stats,angles):
        # 在每个角度下，半径与数值一一对应
        for _y in yticks:
            ax.annotate(f'{_s*_y:.2f}',(_x,_y))

    for idx,log_file in enumerate(log_files):
        df_metrics = pd.read_csv(log_file,index_col=0)

        print(method_names[idx],np.mean(df_metrics.values,axis=0))
        
        labels,stats,angles = read_stats_for_spider(df_metrics)
        
        if idx == base_idx:
            stats = np.ones_like(base_stats)
        else:
            stats = stats / base_stats
        
        # 画图数据准备，角度、状态值
        stats=np.concatenate((stats,[stats[0]]))
        angles=np.concatenate((angles,[angles[0]]))
        # 用 Matplotlib 画蜘蛛图

        ax.plot(angles, stats, '.-', linewidth=3,label=method_names[idx],alpha=0.7)
        # ax.fill(angles, stats, alpha=0.25)

    plt.xticks(angles[:-1],labels)
    plt.yticks(yticks,['']*len(yticks))
    plt.legend(loc = 'upper right')
    plt.tight_layout()
    plt.savefig(save_dir+f"{set_name}-spider_{int(time.time())}",dpi=300)
    plt.close()
    
if __name__ == '__main__':
    plot_spider()