
import torch.multiprocessing as mp

import torch.distributed as dist
import torch.utils.data as data
from contextlib import suppress
import torch
import socket
from dataloader import *
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.cuda.amp import GradScaler
import torch.optim as optim
from tqdm import tqdm
from torch import nn
import argparse
from model import create_model_and_transforms
from utils import *
import math
from loss import *

# https://github.com/openai/CLIP/issues/57
def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 
        
def main(args, exp):
    torch.cuda.set_device(0)
    
    logger = create_logger(os.path.join(exp._save_dir,'log.txt'))
    device = "cuda" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.
    logger.info('current device:{}'.format(device))
    model_map = {'ViT-B16':'ViT-B/16',
                 'ViT-B32':'ViT-B/32',
                 'ViT-L14':'ViT-L/14',
                 'RN50':'RN50'}
    logger.info("Preparing Model:" + args.model)
    
    random_seed(1)

    model, preprocess_train, preprocess_val = create_model_and_transforms(
        model_map[args.model],
        args.pretrained,
        precision=args.precision,
        device=device,
        jit=False,
        force_quick_gelu=False,
        pretrained_image=False,
    )
    
    logger.info("Successfully create CLIP model")

    # use your own data
    logger.info("Preparing Dataset")
    train_dataset = prmlDataset('train', preprocess_train)
    valid_dataset = prmlDataset('valid', preprocess_val)
    train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True, num_workers=12, pin_memory=True, persistent_workers=True, collate_fn=my_collate_fn) #Define your own dataloader
    valid_dataloader = DataLoader(valid_dataset, batch_size = args.batch_size, num_workers=12, pin_memory=True, persistent_workers=True, collate_fn=my_collate_fn)# collate_fn=my_collate_fn
    logger.info("=> Done")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr,betas=(0.9,0.999),eps=1e-6,weight_decay=0.05) #Params used from paper, the lr is smaller, more safe for fine tuning to new dataset
    total_steps = len(train_dataloader) * args.epochs

    scaler = GradScaler() if args.precision == "amp" else None
    logger.info("=> Start Training")

    best_acc = 0

    for epoch in range(args.epochs):
        train_one_epoch(
            args = args,
            epoch = epoch,
            data_loader=train_dataloader,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            logger=logger,
        )
        if epoch % 5 == 0 and epoch > 0:
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, os.path.join(exp._save_dir,f"model_{epoch}.pt"))
            
        
        acc_valid = evaluate(args,valid_dataloader,model)
        
        if acc_valid > best_acc:
            best_acc = acc_valid
            torch.save({
                'state_dict':model.state_dict()
            },os.path.join(exp._save_dir,'model_best.pt'))
        logger.info('Best acc:{},current acc:{}'.format(best_acc, acc_valid))
    
    # test
    model, transform = clip.load("ViT-B/32",device='cuda',jit=False)
    clip.model.convert_weights(model)
    ckpt = f'{exp._save_dir}/model_best.pt'
    parm = torch.load(ckpt)

    test_dataset = prmlDataset('test', transform)
    test_dataloader = DataLoader(test_dataset, batch_size = 1)
    test(model, test_dataloader, 'cuda', ckpt=ckpt)

def train_one_epoch(args, epoch, data_loader, model, optimizer, scaler, 
                logger):
    model.train()
    autocast = torch.cuda.amp.autocast if args.precision == 'amp' else suppress

    train_loss = 0
    total_batches = 0
    for idx, batch in enumerate(tqdm(data_loader)):
        adjust_learning_rate(optimizer, epoch+idx/len(data_loader), args)
        optimizer.zero_grad()
        images, texts, info = batch
        
        images= images.cuda()
        texts = texts.cuda()
        with autocast():
            image_features, text_features, logit_scale = model(images, texts)
            total_loss = SingleLoss(image_features, text_features, logit_scale, info)
        if scaler is not None:
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            optimizer.step()
        train_loss += total_loss.item()
        total_batches += 1
    train_loss /= total_batches
    
    logger.info("=> Epoch {}: Loss: {}".format(epoch,train_loss))

def evaluate(args, valid_dataloader, model):
    cumulative_loss = 0.0
    num_samples = 0
    autocast = torch.cuda.amp.autocast if args.precision == 'amp' else suppress
    model.eval()
    pred = []
    gt = []
    with torch.no_grad():
        for batch in tqdm(valid_dataloader):
            images, texts, info = batch 
            images= images.cuda()
            texts = texts.cuda()
            with autocast():
                # ??????eval
                image_features, text_features, logit_scale = model(images, texts)
                probs = []
                for i in range(image_features.shape[0]):
                    logits_per_image = logit_scale * image_features[i].unsqueeze(0)@ text_features[i].T
                    effect_logits = logits_per_image[:info[i]['text_length']].softmax(dim=-1).cpu().numpy()
                    probs.extend(np.argmax(effect_logits, axis=-1))
                    gt.append(info[i]['caption'])
                
                pred.extend(probs)
        count = 0 
        for i in range(len(gt)):
            if pred[i] == gt[i]:
                count += 1
        acc_valid = count / len(gt)
    return acc_valid

def test(model, test_dataloader, device, ckpt):
    parm = torch.load(ckpt)
    state_dict = parm['state_dict']
    state_dict = consume_prefix(state_dict)
    model.load_state_dict(state_dict)
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
    createJs(results,'./data/full/test_all.json',f'{exp._save_dir}/results.json')

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup:
        lr = args.lr * epoch / args.warmup 
    else:
        lr = args.lr * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup) / (args.epochs - args.warmup)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr

def consume_prefix(ckpt):
    new_state_dict = {}
    for key, value in ckpt.items():
        if key.startswith('module.'):
            new_prefix = key[len('module.'):]
            new_state_dict[new_prefix] = value
        else:
            new_state_dict[key] = value
    return new_state_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for prml')
    parser.add_argument('--device',type=str,default='cuda')
    parser.add_argument('--optimizer',type=str,default='AdamW')
    parser.add_argument('--batch_size',type=int,default=128)
    parser.add_argument('--epochs',type=int,default=30)
    parser.add_argument('--model',type=str,default='ViT-B32',choices=['ViT-B32','ViT-B16','ViT-L14','RN50'])
    parser.add_argument('--pretrained',type=str,default='openai')
    parser.add_argument('--lr',type=float,default=2.5e-5)
    parser.add_argument('--warmup',type=int,default=5)
    parser.add_argument(
        "--precision",
        choices=["amp", "fp16", "fp32"],
        default="fp32",
        help="Floating point precision."
    )
    
    parser.add_argument('--en_wandb', type=int, default=0, choices=[0, 1])
    args = parser.parse_args()
    args.nproc_per_node = torch.cuda.device_count()

    exp = ExpHandler(en_wandb=args.en_wandb, args=args)
    exp.save_config(args)
    main(0, args, exp)

