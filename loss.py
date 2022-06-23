import torch
import torch.distributed.nn
from torch import distributed as dist, nn as nn
from torch.nn import functional as F

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

def SingleLoss(image_features, text_features, logit_scale, info):
    # print(f'text_features:{text_features.shape}')
    # print(f'image_features:{image_features.shape}')
    # print(f'text_features:{text_features.permute(2,0,1).shape}')
    total_loss = 0
    for i in range(image_features.shape[0]):
        logits_per_image = logit_scale * image_features[i].unsqueeze(0)@ text_features[i].T
        total_loss += F.cross_entropy(logits_per_image[:info[i]['text_length']], torch.tensor([info[i]['caption']]).cuda())
    return total_loss/image_features.shape[0]

    # logits_per_image = logit_scale * image_features @ text_features.permute(2,0,1)

    # print(f'ddd:{logits_per_image.shape()}')