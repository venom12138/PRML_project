from utils import *
from scheduler import cosine_lr
from model.model import *
import torch.optim as optim

a = [(5,6,{7,8}),(5,0,{7,8})]
print(a[:][:][0])