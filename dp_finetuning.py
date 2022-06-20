# Latest Update : 31 May 2022, 09:55 GMT+7

# TO ADD :
# Gradient Checkpointing
# Filter out bias from weight decay
# Decaying learning rate with cosine schedule
# Half-precision Adam statistics
# Half-precision stochastically rounded text encoder weights were used

#BATCH_SIZE must larger than 1
import multiprocessing as mp
from turtle import forward
mp.set_start_method('spawn', force=True)
import torch.utils.data as data
import torch
import clip
import socket
from dataloader import *
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from torch import nn
import numpy as np
import argparse
import pdb
from utils import *
BATCH_SIZE = 168
EPOCH = 30


#https://github.com/openai/CLIP/issues/57
def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 
        
def main(args):
    logger = create_logger(os.path.join(args.ckp_path,'log.txt'))
    device = "cuda" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.
    logger.info('current device:{}'.format(device))
    model_map = {'ViT-B16':'ViT-B/16',
                 'ViT-B32':'ViT-B/32',
                 'ViT-L14':'ViT-L/14',
                 'ViT-L14-336':'ViT-L/14@336px'}
    logger.info("Preparing Model:" + args.model)

    model, transform = clip.load(model_map[args.model],device=device,jit=False) #Must set jit=False for training
    logger.info("Successfully create CLIP model")

    # use your own data
    logger.info("Preparing Dataset")
    train_dataset = prmlDataset('train', transform)
    valid_dataset = prmlDataset('valid', transform)
    train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True, num_workers=24) #Define your own dataloader
    valid_dataloader = DataLoader(valid_dataset, batch_size = args.batch_size, num_workers=24)
    logger.info("=> Done")

    if device == "cpu":
        model.float()
    else :
        clip.model.convert_weights(model) # Actually this line is unnecessary since clip by default already on float16

    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=2.5e-7,betas=(0.9,0.999),eps=1e-6,weight_decay=0.05) #Params used from paper, the lr is smaller, more safe for fine tuning to new dataset
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epoch * len(train_dataloader))
    
    # add your own code to track the training progress.
    logger.info("=> Start Training")
    best_acc = 0
    for epoch in range(EPOCH):
        train_epoch(args,epoch,train_dataloader,model,[loss_img,loss_txt],optimizer,scheduler=scheduler,logger=logger)
        if epoch % 5 == 0 and epoch > 0:
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()
                }, os.path.join(args.ckp_path,"model_{}.pt".format(epoch)))
        pred = []
        gt = []
        with torch.no_grad():
            for batch in tqdm(valid_dataloader):
                images, texts, info = batch 
                images= images.cuda()
                texts = texts.cuda()
                logits_per_image, logits_per_text = model(images, texts)
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()
                tag = np.argmax(probs, axis=-1)
                pred.extend(list(np.array(info['caption'])[tag]))
                gt.extend(info['caption'])
            count = 0
            for i in range(len(gt)):
                if pred[i] == gt[i]: count += 1
            acc_valid = count / len(gt)
            if acc_valid > best_acc:
                best_acc = acc_valid
                torch.save({
                    'epoch':epoch,
                    'state_dict':model.state_dict()
                },os.path.join(args.ckp_path,'best_acc_model.pt'))
            logger.info('Best Acc:{},Current Acc:{}'.format(best_acc,acc_valid))
            
def resume(model, ckp):
    ckp = torch.load(ckp)
    model.load_state_dict(ckp['model_state_dict'])
    return model

def train_epoch(args,epoch,data_loader,model,criterion,optimizer,scheduler,
                logger):
    model.train()
    loss_img = criterion[0]
    loss_txt = criterion[1]
    train_loss = 0
    total_batches = 0
    pred = []
    gt = []
    for batch in tqdm(data_loader):
        optimizer.zero_grad()
        images, texts, info = batch 
        images= images.cuda()
        texts = texts.cuda()
        logits_per_image, logits_per_text = model(images,texts)
        tag = torch.argmax(logits_per_image, dim=-1)
        ground_truth = torch.arange(len(images), dtype=torch.long, device=args.device)
        total_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
        total_loss.backward()
        train_loss += total_loss.item()
        if args.device == "cpu":
            optimizer.step()
        else: 
            convert_models_to_fp32(model)
            optimizer.step()
            if scheduler:
                scheduler.step()
            clip.model.convert_weights(model)
        pred.extend(list(np.array(info['caption'])[tag.cpu().numpy()]))
        gt.extend(info['caption'])
        count = 0
        for i in range(len(info['caption'])):
            if info['caption'][tag.cpu().numpy()[i]] == info['caption'][i]: count += 1
        total_batches += 1
    train_loss /= total_batches
    count = 0
    for i in range(len(gt)):
        if pred[i] == gt[i]: count += 1
    acc_train = count / len(gt)
    logger.info("=> Epoch {}: Loss: {}, Train Acc {}".format(epoch,train_loss,acc_train))


def evaluate(model,test_dataloader,device,ckp):
    parm = torch.load(ckp)
    state_dict = parm['state_dict']
    model.load_state_dict(state_dict)
    model.eval()
    results = {}
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            images, texts, info = batch 
            images= images.to(device)
            texts = texts.to(device)
            texts = texts.squeeze(0)
            image_features = model.encode_image(images)
            text_features = model.encode_text(texts)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            values, indices = similarity[0].topk(1)
            predict_label = info['ori_label'][indices][0]
            com_idx = info['name'][0]
            results[com_idx] = predict_label
    createJs(results,'/mnt/lustre/xingsen.vendor/PRMLdataset/full/test_all.json','results3.json')



if __name__ == '__main__':
    model, transform = clip.load("ViT-L/14@336px",device='cuda',jit=False)
    clip.model.convert_weights(model)
    ckp = '/mnt/lustre/xingsen.vendor/PRML/ckp3/best_acc_model.pt'
    test_dataset = prmlDataset('test', transform)
    test_dataloader = DataLoader(test_dataset, batch_size = 1)
    evaluate(model,test_dataloader,'cuda',ckp)
    # parser = argparse.ArgumentParser(description='Args for color match')
    # parser.add_argument('--device',type=str,default='cuda')
    # parser.add_argument('--optimizer',type=str,default='AdamW')
    # parser.add_argument('--nproc_per_node', type=int, default=1)
    # parser.add_argument('--backend', type=str, default='nccl')
    # parser.add_argument('--lr_scheduler',type=str,default='cosine')
    # parser.add_argument('--batch_size',type=int,default=48)
    # parser.add_argument('--epoch',type=int,default=30)
    # parser.add_argument('--model',type=str,default='ViT-L14-336',choices=['ViT-B32','ViT-B16','ViT-L14','ViT-L14-336'])
    # parser.add_argument('--ckp_path',type=str,default='/mnt/lustre/xingsen.vendor/PRML/ckp3')
    # parser.add_argument('--master_addr', type=str, default=socket.gethostbyname(socket.gethostname()))
    # parser.add_argument('--master_port', type=int, default=31111)
    # parser.add_argument('--nnodes', type=int, default=None)
    # parser.add_argument('--node_rank', type=int, default=None)
    # args = parser.parse_args()
    # main(args)
    # torch.multiprocessing.spawn(main,args=(args,),nprocs=args.nproc_per_node)