# Latest Update : 31 May 2022, 09:55 GMT+7

# TO ADD :
# Gradient Checkpointing
# Filter out bias from weight decay
# Decaying learning rate with cosine schedule
# Half-precision Adam statistics
# Half-precision stochastically rounded text encoder weights were used

#BATCH_SIZE must larger than 1
import multiprocessing as mp
mp.set_start_method('spawn', force=True)
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
from scheduler import cosine_lr
from utils import *
from loss import *
BATCH_SIZE = 168
EPOCH = 30

class MyClip(nn.Module):
    def __init__(self,clip_model):
        super().__init__()
        self.model = clip_model.float()
        self.logit_scale = clip_model.logit_scale
    def forward(self,images, texts):
        image_features = self.model.encode_image(images)
        text_features = self.model.encode_text(texts)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        return image_features,text_features,self.logit_scale.exp()



#https://github.com/openai/CLIP/issues/57
def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 
        
def main(local_rank, args):
    rank, world_size = init_distributed(local_rank, args)
    logger = create_logger(os.path.join(args.ckp_path,'log.txt'))
    device = "cuda" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.
    logger.info('current device:{}'.format(device))
    model_map = {'ViT-B16':'ViT-B/16',
                 'ViT-B32':'ViT-B/32',
                 'ViT-L14':'ViT-L/14',
                 'RN50':'RN50'}
    logger.info("Preparing Model:" + args.model)
    dist.barrier()

    random_seed(1)

    #model, transform = clip.load(model_map[args.model],device=device,jit=False) #Must set jit=False for training
    model, preprocess_train, preprocess_val = create_model_and_transforms(
        model_map[args.model],
        args.pretrained,
        precision=args.precision,
        device=device,
        jit=args.torchscript,
        force_quick_gelu=args.force_quick_gelu,
        pretrained_image=False,
    )
    dist.barrier()
    model = DistributedDataParallel(model,device_ids=[local_rank],find_unused_parameters=False)
    logger.info("Successfully create CLIP model")

    # use your own data
    logger.info("Preparing Dataset")
    train_dataset = prmlDataset('train', preprocess_train)
    train_sampler = data.DistributedSampler(train_dataset)
    valid_dataset = prmlDataset('valid', preprocess_val)
    train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=(train_sampler is None), num_workers=24, sampler=train_sampler, pin_memory=True, persistent_workers=True) #Define your own dataloader
    valid_dataloader = DataLoader(valid_dataset, batch_size = args.batch_size, num_workers=24, pin_memory=True, persistent_workers=True)
    logger.info("=> Done")

    optimizer = optim.AdamW(model.parameters(), lr=2.5e-7,betas=(0.9,0.999),eps=1e-6,weight_decay=0.05) #Params used from paper, the lr is smaller, more safe for fine tuning to new dataset
    total_steps = len(train_dataloader) * args.epochs
    scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)
    scaler = GradScaler() if args.precision == "amp" else None
    logger.info("=> Start Training")
    if rank == 0:
        best_rank = 1000000000000
    for epoch in range(EPOCH):
        train_sampler.set_epoch(epoch)
        train_epoch(
            args=args,
            epoch = epoch,
            data_loader=train_dataloader,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            logger=logger,
            rank=rank,
            world_size=world_size
        )
        if epoch % 5 == 0 and epoch > 0:
            if rank == 0:
                torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict()
                    }, os.path.join(args.ckp_path,"model_{epoch}.pt")) 
            dist.barrier()
        if rank == 0:
            val_metrics = evaluate(args,valid_dataloader,model)
            cur_rank = val_metrics['image_to_text_mean_rank'] + 0.1 * val_metrics['image_to_text_median_rank']
            if cur_rank < best_rank:
                best_rank = cur_rank
                torch.save({
                    'state_dict':model.state_dict()
                },os.path.join(args.ckp_path,'best_acc_model.pt'))
            logger.info('Best mean rank:{},current mean rank:{}'.format(best_rank,cur_rank))
        dist.barrier()

def train_epoch(args,epoch,data_loader,model,optimizer, scaler, scheduler,
                logger,rank,world_size):
    model.train()
    autocast = torch.cuda.amp.autocast if args.precision == 'amp' else suppress
    loss = ClipLoss(
        local_loss = False,
        gather_with_grad=False,
        cache_labels=True,
        rank=rank,
        world_size=world_size,
        use_horovod=args.horovod
    )
    train_loss = 0
    total_batches = 0
    for batch in tqdm(data_loader):
        optimizer.zero_grad()
        images, texts, info = batch 
        images= images.cuda()
        texts = texts.cuda()
        with autocast():
            image_features, text_features, logit_scale = model(images,texts)
            total_loss = loss(image_features, text_features, logit_scale)
        if scaler is not None:
            scaler.scale(total_loss).backward()
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            optimizer.step()
        train_loss += total_loss.item()
        total_batches += 1
    train_loss /= total_batches
    if rank == 0:
        logger.info("=> Epoch {}: Loss: {}".format(epoch,train_loss))

def evaluate(args,valid_dataloader,model):
    cumulative_loss = 0.0
    num_samples = 0
    autocast = torch.cuda.amp.autocast if args.precision == 'amp' else suppress
    model.eval()
    all_image_features,all_text_features = [], []
    with torch.no_grad():
        for batch in tqdm(valid_dataloader):
            images, texts, info = batch 
            images= images.cuda()
            texts = texts.cuda()
            with autocast():
                image_features, text_features, logit_scale = model.module(images, texts)
                all_image_features.append(image_features.cpu())
                all_text_features.append(text_features.cpu())
                logit_scale = logit_scale.mean()
                logits_per_image = logit_scale * image_features @ text_features.t()
                logits_per_text = logits_per_image.t()
                batch_size = images.shape[0]
                labels = torch.arange(batch_size,device=args.device).long()
                total_loss = (
                    F.cross_entropy(logits_per_image, labels)+
                    F.cross_entropy(logits_per_text, labels)
                ) / 2
            cumulative_loss += total_loss * batch_size 
            num_samples += batch_size
        val_metrics = get_metrics(
            image_features=torch.cat(all_image_features),
            text_features = torch.cat(all_text_features),
            logit_scale=logit_scale.cpu()
        )
    return val_metrics

def test(model,test_dataloader,device,ckp):
    parm = torch.load(ckp)
    state_dict = parm['model_state_dict']
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
    createJs(results,'/mnt/lustre/xingsen.vendor/PRMLdataset/full/test_all.json','results.json')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Args for color match')
    parser.add_argument('--device',type=str,default='cuda')
    parser.add_argument('--optimizer',type=str,default='AdamW')
    parser.add_argument('--nproc_per_node', type=int, default=4)
    parser.add_argument('--backend', type=str, default='nccl')
    parser.add_argument('--lr_scheduler',type=str,default='cosine')
    parser.add_argument('--batch_size',type=int,default=896)
    parser.add_argument('--epochs',type=int,default=30)
    parser.add_argument('--model',type=str,default='RN50',choices=['ViT-B32','ViT-B16','ViT-L14','RN50'])
    parser.add_argument('--pretrained',type=str,default='openai')
    parser.add_argument('--ckp_path',type=str,default='/mnt/lustre/xingsen.vendor/PRML')
    parser.add_argument('--master_addr', type=str, default=socket.gethostbyname(socket.gethostname()))
    parser.add_argument('--master_port', type=int, default=31111)
    parser.add_argument('--nnodes', type=int, default=None)
    parser.add_argument('--node_rank', type=int, default=None)
    parser.add_argument('--lr',type=float,default=2.5e-5)
    parser.add_argument('--warmup',type=int,default=5)
    parser.add_argument(
        "--precision",
        choices=["amp", "fp16", "fp32"],
        default="fp32",
        help="Floating point precision."
    )
    parser.add_argument(
        "--horovod",
        default=False,
        action="store_true",
        help="Use horovod for distributed training."
    )
    parser.add_argument(
        "--torchscript",
        default=False,
        action='store_true',
        help="torch.jit.script the model, also uses jit version of OpenAI models if pretrained=='openai'",
    )
    parser.add_argument(
        "--force-quick-gelu",
        default=False,
        action='store_true',
        help="Force use of QuickGELU activation for non-OpenAI transformer models.",
    )
    args = parser.parse_args()
    torch.multiprocessing.spawn(main,args=(args,),nprocs=args.nproc_per_node)