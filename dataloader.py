'''
Author: Hao
Date: 2022-06-12 16:41:50
Email: wenh19@outlook.com
LastEditors: Hao
LastEditTime: 2022-06-12 21:40:21
Description: 
Copyright (c) 2022 by Wen Hao, All Rights Reserved. 
'''
from PIL import Image
import sys, os
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomCrop
from tqdm import tqdm
from torch.utils.data import Dataset 
import torch.nn.functional as F
from torch import nn
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import clip
import glob
import torch
import numpy as np
from utils import *
import pdb

class prmlDataset(Dataset):
    ROOT_PATH = r'/mnt/cache/share_data/caisonghao.vendor/full'
    JSON_NAME = ['train_all.json','test_all.json', 'cn_en.json']
    CV_SEED = 41 # Cross validation seed, must be fixed while training
    TR_RATIO = 0.8
    def __init__(self, type, transform):
        self.rs = np.random.RandomState(self.CV_SEED)
        self.translator = parseJson(os.path.join(self.ROOT_PATH, self.JSON_NAME[2]))
        self.capList = []
        self.imgList = []
        if str(type)=='test':
            self.cn_capList = []
        self.type = type
        if str(type)=='train' or str(type)=='valid':
            jsPath = os.path.join(self.ROOT_PATH, self.JSON_NAME[0])
            self.js = parseJson(jsPath)
            self.imgPath = os.path.join(self.ROOT_PATH, 'train')
            comList = glob.glob(self.imgPath+'/*/*.jpg')### 所有图片的name
            self.l = int(self.TR_RATIO * len(comList))
            self.rs.shuffle(comList)
            if str(type)=='train':
                comList = comList[:self.l]
            if str(type)=='valid':
                comList = comList[self.l:]
                self.l = len(comList)
            for filePath in comList:
                com_idx_jpg = filePath.split('/')[-1]   ## sigle
                com, idx = com_idx_jpg.split('_')[0], int(com_idx_jpg.strip('.jpg').split('_')[1])
                #pdb.set_trace()
                cap_cn = self.js[com]['imgs_tags'][idx][com_idx_jpg]
                cap_en = self.translator[cap_cn]
                self.capList.append(cap_en)  
                self.imgList.append(filePath) 
            self.title = clip.tokenize(self.capList)
            
        elif str(type)=='test':
            jsPath = os.path.join(self.ROOT_PATH, self.JSON_NAME[1])
            self.js = parseJson(jsPath)
            self.imgPath = os.path.join(self.ROOT_PATH, 'test')
            comList = glob.glob(self.imgPath+'/*/*.jpg')
            self.l = len(comList)
            for filePath in comList:
                #pdb.set_trace()
                com_idx_jpg = filePath.split('/')[-1]
                com = com_idx_jpg.split('_')[0]
                cap_cn = self.js[com]['optional_tags']
                cap_en =  [self.translator[cn] for cn in cap_cn]
                self.capList.append(cap_en)
                self.cn_capList.append(cap_cn)  
                self.imgList.append(filePath)    
            self.title = [clip.tokenize(en) for en in self.capList]
                
                
            
        self._transform = transform
    
        
    def __len__(self):
        return len(self.imgList)

    def __getitem__(self, idx):
        image = self._transform(Image.open(self.imgList[idx]))
        title = self.title[idx]
        if str(self.type) == 'test':
            info = {'path': self.imgList[idx], 
                    'name': self.imgList[idx].split('/')[-1].strip('.jpg'),
                    'caption': self.capList[idx],
                    'ori_label':self.cn_capList[idx]}
        else:
            info = {'path': self.imgList[idx], 
                    'name': self.imgList[idx].split('/')[-1].strip('.jpg'),
                    'caption': self.capList[idx]}
        return image, title, info

    def get_json(self):
        if str(self.type) == 'test':
            return self.js
        else:
            return None

if __name__ == "__main__":
    # test
    dataset = prmlDataset(type='test', transform=None)