import sys
sys.path.append("..")
from models.Process import *

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

batch_size = 200      # 
classnum = 10         # [10, 100]
dataset = "mnist"     # ["mnist", "cifar10", "cifar100", "imagenet"]
model = "LeNet"       # ["LeNet", "AlexNet", "ResNet"]
loss = "ODN"          # ["xent", "mlm", "smlm", "ODN"]  
opt = "SGD"           # ["SGD", "Adam", "RMSprop"]
is_gpu = 1            # [0, 1]
epoch_num = 400       #

#####Train#####
# learner = train_valid(batch_size=batch_size, is_gpu=is_gpu, dataset=dataset, epoch_num = epoch_num,
#      model=model, loss=loss)
# learner.fit()

learner = train_test(batch_size=batch_size, is_gpu=is_gpu, dataset=dataset, epoch_num = epoch_num,
     model=model, loss=loss)
learner.fit()