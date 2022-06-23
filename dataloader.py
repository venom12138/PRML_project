from PIL import Image
import cv2
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
import torch.utils.data as data
from contextlib import suppress
import torch
import socket
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

class prmlDataset(Dataset):
    ROOT_PATH = f'./data/full'
    JSON_NAME = ['train_all.json','test_all.json', 'dirty_cn_en.json']
    CV_SEED = 41 # Cross validation seed, must be fixed while training #  'cn_en.json',
    TR_RATIO = 0.8 # 划分训练集和验证集
    def __init__(self, type, transform):
        self.rs = np.random.RandomState(self.CV_SEED)
        # 转换后的text集
        self.translator = parseJson(os.path.join(self.ROOT_PATH, self.JSON_NAME[2]))
        self.tagList = []
        self.imgList = []
        self.type = str(type)

        if self.type=='test':
            self.cn_tagList = []
            jsPath = os.path.join(self.ROOT_PATH, self.JSON_NAME[1])
        else:
            self.gt_idx = []
            jsPath = os.path.join(self.ROOT_PATH, self.JSON_NAME[0])
        
        self.js = parseJson(jsPath)
        self.imgPath = os.path.join(self.ROOT_PATH, 'train')
        dataList = glob(self.imgPath+'/*/*.jpg') ### 所有图片的name
        # 划分训练集和验证集
        self.train_len = int(self.TR_RATIO * len(dataList))

        if self.type == 'train' or self.type == 'valid':
            
            self.rs.shuffle(dataList)
            if self.type=='train':
                dataList = dataList[:self.train_len]
            if self.type=='valid':
                dataList = dataList[self.train_len:]
                self.train_len = len(dataList)
            for filePath in dataList:
                # 图片的名称
                com_name = filePath.split('/')[-1]   ## single
                # 图片所属的商品类 和 图片在该类商品中的标号
                com = com_name.split('_')[0]
                idx = int(com_name.strip('.jpg').split('_')[1])
                # 获取图片的中文gt label
                tag_cn = self.js[com]['imgs_tags'][idx][com_name]
                # 获取图片的英文处理后的label
                tag_en = self.translator[tag_cn]
                # optional cap
                opt_tags_cn = self.js[com]['optional_tags']
                opt_tags_en =  [self.translator[cn] for cn in opt_tags_cn]
                # optional tags list
                self.tagList.append(opt_tags_en)
                # self.tagList.append(tag_en)  
                self.imgList.append(filePath) 
                self.gt_idx.append(opt_tags_en.index(tag_en))
  
            self.title = [clip.tokenize(en) for en in self.tagList]
            
            
        elif self.type=='test':
            
            for filePath in dataList:
                #pdb.set_trace()
                com_name = filePath.split('/')[-1]
                com = com_name.split('_')[0]
                tag_cn = self.js[com]['optional_tags']
                tag_en =  [self.translator[cn] for cn in tag_cn]
                self.tagList.append(tag_en)
                self.cn_tagList.append(tag_cn)  
                self.imgList.append(filePath)    
            self.title = [clip.tokenize(en) for en in self.tagList]         
            
        self._transform = transform
    
    def __len__(self):
        return len(self.imgList)

    def __getitem__(self, idx):
        if self._transform is not None:
            image = self._transform(Image.open(self.imgList[idx])) # Image.open(self.imgList[idx])
        else:
            self._transform = Compose([
                    _convert_to_rgb,
                    ToTensor(),
                ])
            image = self._transform(Image.open(self.imgList[idx]))
        # optional labels
        title = self.title[idx]
        if str(self.type) == 'test':
            info = {'path': self.imgList[idx], 
                    'name': self.imgList[idx].split('/')[-1].strip('.jpg'),
                    'caption': self.tagList[idx],
                    'ori_label':self.cn_tagList[idx]}
        else:
            info = {'path': self.imgList[idx], 
                    'name': self.imgList[idx].split('/')[-1].strip('.jpg'),
                    'caption': self.gt_idx[idx],
                    'text_length': len(title)}
        # print(title.shape)
        return image, title, info

    def get_json(self):
        if str(self.type) == 'test':
            return self.js
        else:
            return None

def _convert_to_rgb(image):
    return image.convert('RGB')

if __name__ == "__main__":
    # test
    train_dataset = prmlDataset(type='train', transform=None)
    # Define your own dataloader
    # train_dataloader = DataLoader(train_dataset, batch_size = 128, shuffle=False, num_workers=12, pin_memory=True,persistent_workers=True,collate_fn=my_collate_fn) 
    # for idx, batch in enumerate(train_dataloader):
    #     images, texts, info = batch
    #     # length = max(info[:]['text_length'])
    #     print(texts.shape)

    #     print('---------')
    
    
    # for i in range(10):
    #     _, title, _ = dataset[i]
    #     print(len(title))
    