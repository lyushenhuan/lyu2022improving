import sys
sys.path.append("..")
from models.Process import *
from sklearn.utils import shuffle
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

batch_size = 200      # 
classnum = 10         # [10, 100]
dataset = "cifar10"     # ["mnist", "cifar10", "cifar100", "imagenet"]
model = "AlexNet"       # ["LeNet", "AlexNet", "ResNet"]
loss = "ODN"          # ["xent", "mlm", "smlm", "ODN"]  
hinge = 1             # [0, 1]
opt = "SGD"           # ["SGD", "Adam", "RMSprop"]
is_gpu = 1            # [0, 1]
epoch_num = 400       #

#####Train#####
for fraction in [0.5, 1, 5, 10, 20, 100]:
    for loss in ["xent", "mlm", "ODN", "smlm"]:
        print("\n"+loss+"  "+str(fraction)+":")
        learner = train_test(batch_size=batch_size, is_gpu=is_gpu, dataset=dataset, epoch_num = (epoch_num*int(100/fraction)),
             model=model, loss=loss, hinge=hinge, fraction=fraction)
        learner.fit()



