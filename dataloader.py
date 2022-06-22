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
import torch.utils.data as data
from contextlib import suppress
import torch
import socket
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

class prmlDataset(Dataset):
    ROOT_PATH = f'./data/full'
    JSON_NAME = ['train_all.json','test_all.json', 'cn_en.json']
    CV_SEED = 41 # Cross validation seed, must be fixed while training
    TR_RATIO = 0.8 # 划分训练集和验证集
    def __init__(self, type, transform):
        self.rs = np.random.RandomState(self.CV_SEED)
        # 转换后的text集
        self.translator = parseJson(os.path.join(self.ROOT_PATH, self.JSON_NAME[2]))
        self.capList = []
        self.imgList = []
        
        self.type = type
        if str(type)=='test':
            self.cn_capList = []
        else:
            self.gt_idx = []
        if str(type)=='train' or str(type)=='valid':
            jsPath = os.path.join(self.ROOT_PATH, self.JSON_NAME[0])
            self.js = parseJson(jsPath)
            self.imgPath = os.path.join(self.ROOT_PATH, 'train')
            comList = glob(self.imgPath+'/*/*.jpg') ### 所有图片的name
            # 划分训练集和验证集
            self.l = int(self.TR_RATIO * len(comList))
            self.rs.shuffle(comList)
            if str(type)=='train':
                comList = comList[:self.l]
            if str(type)=='valid':
                comList = comList[self.l:]
                self.l = len(comList)
            for filePath in comList:
                # 图片的名称
                com_idx_jpg = filePath.split('/')[-1]   ## single
                # 图片所属的商品类 和 图片在该类商品中的标号
                com, idx = com_idx_jpg.split('_')[0], int(com_idx_jpg.strip('.jpg').split('_')[1])
                # 获取图片的中文gt label
                cap_cn = self.js[com]['imgs_tags'][idx][com_idx_jpg]
                # 获取图片的英文处理后的label
                cap_en = self.translator[cap_cn]
                # optional cap
                opt_tags_cn = self.js[com]['optional_tags']
                opt_tags_en =  [self.translator[cn] for cn in opt_tags_cn]
                # optional tags list
                self.capList.append(opt_tags_en)
                # self.capList.append(cap_en)  
                self.imgList.append(filePath) 
                self.gt_idx.append(opt_tags_en.index(cap_en))

            self.title = [clip.tokenize([f'a photo of {word} clothes' for word in en]) for en in self.capList]  
            # self.title2 = clip.tokenize(self.CapList)
            # print(f'title shape:{self.title[0].shape}')
            # print(f'title2 shape:{self.title2.shape}')
            
        elif str(type)=='test':
            jsPath = os.path.join(self.ROOT_PATH, self.JSON_NAME[1])
            self.js = parseJson(jsPath)
            self.imgPath = os.path.join(self.ROOT_PATH, 'test')
            comList = glob(self.imgPath+'/*/*.jpg')
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
            self.title = [clip.tokenize([f'a photo of {word} clothes' for word in en]) for en in self.capList]               
            
        self._transform = transform
    
    def __len__(self):
        return len(self.imgList)

    def __getitem__(self, idx):
        if self._transform is not None:
            image = self._transform(Image.open(self.imgList[idx]))
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
                    'caption': self.capList[idx],
                    'ori_label':self.cn_capList[idx]}
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
    