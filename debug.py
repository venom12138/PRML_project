from utils import *
from scheduler import cosine_lr
from model.model import *
import torch.optim as optim

model = Bottleneck(2,3)
optimizer = optim.SGD(model.parameters(), lr=0.1)
total_steps = 100
scheduler = cosine_lr(optimizer, 0.3, 0, total_steps)
print(scheduler)
for i in range(100):
    optimizer.step()
    scheduler()
    print(optimizer.param_groups[0]['lr'])
