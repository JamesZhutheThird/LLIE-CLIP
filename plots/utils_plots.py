import os

import cv2
import numpy as np


def cv_imread(file_path,code=cv2.COLOR_BGR2RGB):
    cv_img = cv2.imdecode(np.fromfile(file_path,dtype=np.uint8),-1)
    cv_img = cv2.cvtColor(cv_img,code=code)
    return cv_img
        
def ablation_path(fl,sl,icp):
    root = '../results'
    for sub1 in os.listdir(root):
        if sub1.startswith('ablation_') and sub1.endswith(f'_fl_{fl}_sl_{sl}_icp_{icp}'):
            for sub2 in os.listdir(os.path.join(root,sub1)):
                sub3 = 'images'
                for sub4 in os.listdir(os.path.join(root,sub1,sub2,sub3)):
                    if sub4.isdigit():
                        return os.path.join(root,sub1,sub2,sub3,sub4)
    return ''

def pick_images(root,n=10,ratio=(2,3)):
    images = os.listdir(root)
    # images = images.sort()
    picked_list = []
    
    while len(picked_list) < n:
        img_idx = np.random.randint(len(images))
        img_name = images[img_idx]
        if not img_name in picked_list:
            img = cv_imread(os.path.join(root,img_name))
            if img.shape[0]/ratio[0] == img.shape[1]/ratio[1]:
                picked_list.append(img_name)
                # print('Read image ' + img_name +', picked.'+f'{img.shape}')
        #     else:
        #         print('Read image ' + img_name +', dropped.'+f'{img.shape}')
        # else:
        #     print('Want to read image' + img_name+', repeated request.')
                
    return picked_list

def fix_images1():
    fixed_list = [
        'a0016-jmac_MG_0795.png',
        'a0017-050710_031618__MG_3496.png',
        'a0024-_DSC8932.png',
        'a0043-07-11-27-at-12h09m46s-_MG_7307.png',        
        'a0054-kme_097.png',
        'a0060-jmac_DSC3171.png',
        'a0065-_DSC6405.png',
        'a0076-jmac_MG_5736.png',
        'a0091-jmac_MG_4959.png'
    ]
    return fixed_list

def fix_images2():
    fixed_list = [
        'a0080-kme_544.png',
        'a0015-DSC_0081.png',
        'a0089-jn_20080509_245.png',
        'a0013-MB_20030906_001.png',
        'a0068-LS051026_day_1_arive53.png'
    ]
    
    return fixed_list
    
def ddprun_namefix(old_name):
    name,postfix = old_name.split('.')
    new_name = name+'_20' + '.'+postfix
    return new_name


def get_names_tra():
    set_name = 'lol'
    method_list = ['Original','CLAHE','MSR','Ground Truth']
    path_list = ['../data/lol/low',
                '../results/lol-clahe',        
                '../results/lol-retinex',
                '../data/lol/high',
                ]
    return set_name,method_list,path_list
    
def get_names_abl():
    set_name = 'test'

    method_list = ['Original',
                'SCI',
                'SCI-os',
                'SCI,of',
                'SCI-ss',
                'CLIP',
                'CLIP-os',
                'CLIP-of',
                'CLIP-ss',
                'Ground Truth'
                ]

    path_list = ['../data/test/low',
                ablation_path(1,1,0),
                ablation_path(0,1,0),
                ablation_path(1,0,0),
                ablation_path(1,2,0),
                ablation_path(1,1,1),
                ablation_path(0,1,1),
                ablation_path(1,0,1),
                ablation_path(1,2,1),
                '../data/test/high',
                ]
    
    return set_name,method_list,path_list

def get_names_ablbar():
    set_name = 'test'

    # method_list = [
    #             'Cnn',
    #             'Cnn, o-smooth',
    #             'Cnn, o-fidelity',
    #             'Cnn, s-smooth',
    #             'Cnn, s-fidelity',
    #             'Icp',
    #             'Icp, o-smooth',
    #             'Icp, o-fidelity',
    #             'Icp, s-smooth',
    #             'Icp, s-fidelity',
    #             # 'Ground Truth'
    #             ]

    method_list = [
        'nm',
        'os',
        'of',
        'ss',
        'sf',
    ] * 2
    
    path_list = [
                ablation_path(1,1,0),
                ablation_path(0,1,0),
                ablation_path(1,0,0),
                ablation_path(1,2,0),
                ablation_path(2,1,0),
                ablation_path(1,1,1),
                ablation_path(0,1,1),
                ablation_path(1,0,1),
                ablation_path(1,2,1),
                ablation_path(2,1,1),
                # '../data/test/high',
                ]
    
    return set_name,method_list,path_list

def get_names_adv1():
    set_name = 'test'
    method_list = [
        'Original',
        'CLAHE',
        'MSR',
        'SCI',
        'CLIP',
        'Ground Truth'
        ]
    path_list = ['../data/test/low',
                '../results/test-clahe',
                '../results/test-retinex',
                ablation_path(1,1,0),
                ablation_path(1,1,1),
                '../data/test/high',
                ]
    return set_name,method_list,path_list

def get_names_adv2():
    set_name = 'test'
    method_list = [
        'Original',
        'Traditional',
        'SCI',
        'CLIP',
        'Ground Truth'
        ]
    path_list = ['../data/test/low',
                '../results/test-clahe',
                ablation_path(1,1,0),
                ablation_path(1,1,1),
                '../data/test/high',
                ]
    return set_name,method_list,path_list

def get_log_files(path_list):
    log_files = []
    
    for path in path_list:
        found = False
        
        for file in os.listdir(path):
            if file.endswith(".csv"):
                log_files.append(os.path.join(path,file))                
                found = True
                
        if not found:
            log_files.append('')
            
        found = False
        
    return log_files