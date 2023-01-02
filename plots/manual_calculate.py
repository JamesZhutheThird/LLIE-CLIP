import os
import sys
import torch
from pathlib import Path

import numpy as np
import pandas as pd
from utils_plots import *

from metrics import *

ori_dir = '../data/test/low'
test_dir = ablation_path(0,1,1)
gt_dir = '../data/test/high'

save_dir = './'

metric_df = None
filelist = os.listdir(ori_dir)
filelist.sort()
for file in filelist:
    ori = torch.Tensor(cv_imread(os.path.join(ori_dir,file)))
    test = torch.Tensor(cv_imread(os.path.join(test_dir,ddprun_namefix(file))))
    gt = torch.Tensor(cv_imread(os.path.join(gt_dir,file)))
    
    metrics_batch = get_metrics(file,test,ori,gt)
    if metrics_df is not None:
        metrics_df = pd.concat([metrics_df, metrics_batch],axis=0)
    else:
        metrics_df = metrics_batch
        
metrics_df.to_csv(save_dir+'metrics.csv')