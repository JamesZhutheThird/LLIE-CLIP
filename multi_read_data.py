import numpy as np
import torch
import torch.utils.data
import random
from PIL import Image
from glob import glob
import torchvision.transforms as transforms
import os

batch_w = 600
batch_h = 400


class MemoryFriendlyLoader(torch.utils.data.Dataset):
    def __init__(self, img_dir, task, gt_dir=None):
        self.low_img_dir = img_dir
        self.high_img_dir = gt_dir
        self.task = task
        self.train_low_data_names = []
        self.test_high_data_names = []
        
        self.has_gt = (self.high_img_dir is None)
            
        for root, dirs, names in os.walk(self.low_img_dir):
            for name in names:
                self.train_low_data_names.append(os.path.join(root, name))
                if self.has_gt:
                    self.test_high_data_names.append(os.path.join(self.high_img_dir, name))

        # os.walk是无序的，必须sort
        self.train_low_data_names.sort()
        self.test_high_data_names.sort()
        self.count = len(self.train_low_data_names)

        transform_list = []
        transform_list += [transforms.ToTensor()]
        self.transform = transforms.Compose(transform_list)
        
        gt_transform_list = [transforms.ToTensor()]
        self.gt_transform = transforms.Compose(gt_transform_list)

    def load_images_transform(self, file):
        im = Image.open(file).convert('RGB')
        img_norm = self.transform(im).numpy()
        img_norm = np.transpose(img_norm, (1, 2, 0))
        return img_norm

    def __getitem__(self, index):
        
        # 相同index文件名相同，文件路径不同
        low = self.load_images_transform(self.train_low_data_names[index])
        
        if self.has_gt:
            _high = Image.open(self.test_high_data_names[index]).convert('RGB')
            high = self.gt_transform(_high)
        else:
            gt = None

        h = low.shape[0]
        w = low.shape[1]
        #
        h_offset = random.randint(0, max(0, h - batch_h - 1))
        w_offset = random.randint(0, max(0, w - batch_w - 1))
        #
        # if self.task != 'test':
        #     low = low[h_offset:h_offset + batch_h, w_offset:w_offset + batch_w]

        low = np.asarray(low, dtype=np.float32)
        low = np.transpose(low[:, :, :], (2, 0, 1))

        img_name = self.train_low_data_names[index].split('\\')[-1]
        # if self.task == 'test':
        #     # img_name = self.train_low_data_names[index].split('\\')[-1]
        #     return torch.from_numpy(low), img_name
        low = torch.from_numpy(low)
        
        return low, high, img_name

    def __len__(self):
        return self.count
