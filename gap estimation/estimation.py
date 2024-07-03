import sys
sys.path.append("..")
from models.Process import *
from sklearn.utils import shuffle
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

batch_size = 200      # 
classnum = 10         # [10, 100]
dataset = "mnist"     # ["mnist", "cifar10", "cifar100", "imagenet"]
model = "LeNet"       # ["LeNet", "AlexNet", "ResNet"]
loss = "ODN"          # ["xent", "mlm", "smlm", "ODN"]  
hinge = 0             # [0, 1]
opt = "SGD"           # ["SGD", "Adam", "RMSprop"]
is_gpu = 1            # [0, 1]
epoch_num = 60       #
fraction = 5

class is_hinge():

    def __init__(self, batch_size=200, is_gpu=1, dataset="cifar10", epoch_num=200, model="AlexNet", loss="ODN", hinge=1, opt="SGD", fraction=100):
        super(is_hinge, self).__init__()
        self.batch_size = batch_size
        self.is_gpu = is_gpu
        self.dataset = dataset
        self.epoch_num = epoch_num
        self.model = model
        self.loss = loss
        self.opt = opt
        self.fraction = fraction
        self.hinge = hinge

    def fit(self):

        if self.dataset=="mnist":
            samples = 60000
            classnum = 10
        if self.dataset=="cifar10":
            samples = 50000
            classnum = 10

        #####Data transform######
        tvloader, trainloader, validloader, testloader = data_transform(dataset=self.dataset, batch_size=self.batch_size, fraction=self.fraction)
        #####Nets#####
        if self.model=="LeNet":
            net = Net(num_classes=classnum)
        if self.model=="AlexNet":
            net = Sim_AlexNet(num_classes=classnum)
        if self.is_gpu==1:
            net = net.cuda()


        #####Loss and Optimizer#####
        import torch.optim as optim

        if self.loss=="xent":
            criterion = nn.CrossEntropyLoss()
        if self.loss=="mlm":
            criterion = nn.MultiMarginLoss()
        if self.loss=="smlm":
            criterion = SoftMarginLoss(class_num=classnum)
        if self.loss=="ODN":
            if self.dataset=="mnist":
                criterion = OptimalMarginDistributionLoss(class_num=classnum, hinge=self.hinge , gamma=1.7, theta=0.3)
            if self.dataset=="cifar10":
                criterion = OptimalMarginDistributionLoss(class_num=classnum, hinge=self.hinge , gamma=1.2, theta=0.7)   # hinge=0, gamma=1.2, theta=0.7, mu=0.1, class_num=10
        if self.opt=="SGD": 
            optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        if self.opt=="Adam":
            optimizer = optim.Adam(net.parameters(), lr=0.001)
        if self.opt=="RMSprop":
            optimizer = optim.RMSprop(net.parameters())

        if dataset=="mnist":
            epoch_list = []
            train_list = []
            test_list = []
            train_rate_list = []
            test_rate_list = []
            c = 0
            for epoch in range(self.epoch_num): 
                
                for data in tvloader:
                    inputs, labels = data
                    if self.is_gpu==0:
                        inputs, labels = Variable(inputs), Variable(labels)
                    if self.is_gpu==1:
                        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                    optimizer.zero_grad()
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    c += 1

                    if c%15==0:
                        epoch_c = c/15
                        correct = 0
                        total = 0
                        ds = torch.Tensor(0)
                        for data in tvloader:
                            images, labels = data
                            if self.is_gpu==0:
                                outputs = net(Variable(images))
                            if self.is_gpu==1:
                                outputs = net(Variable(images.cuda()))
                                outputs = outputs.cpu()
                            d_m = torch.zeros(self.batch_size)
                            for i in range(self.batch_size):
                                d = torch.zeros([10])
                                for j in range(10):
                                    if j != labels[i]:
                                        d[j] = outputs.data[i, labels[i]] - outputs.data[i, j]
                                    else:
                                        d[j] = 99999
                                d_m[i] = torch.min(d)
                            ds = torch.cat((ds, d_m), 0)
                            _, predicted = torch.max(outputs.data, 1)
                            total += labels.size(0)
                            correct += (predicted == labels).sum()
                        mean = torch.mean(ds)
                        std = torch.std(ds)
                        train_rate = std/mean
                        train_acc = correct / total
                        print('%d epoch; Accuracy of the network on the train images: %f %%' % (epoch + 1, 100 * train_acc))

                        correct = 0
                        total = 0
                        for data in testloader:
                            images, labels = data
                            if self.is_gpu==0:
                                outputs = net(Variable(images))
                            if self.is_gpu==1:
                                outputs = net(Variable(images.cuda()))
                                outputs = outputs.cpu()
                            d_m = torch.zeros(self.batch_size)
                            for i in range(self.batch_size):
                                d = torch.zeros([10])
                                for j in range(10):
                                    if j != labels[i]:
                                        d[j] = outputs.data[i, labels[i]] - outputs.data[i, j]
                                    else:
                                        d[j] = 99999
                                d_m[i] = torch.min(d)
                            ds = torch.cat((ds, d_m), 0)
                            _, predicted = torch.max(outputs.data, 1)
                            total += labels.size(0)
                            correct += (predicted == labels).sum()
                        mean = torch.mean(ds)
                        std = torch.std(ds)
                        test_rate = std/mean
                        test_acc = correct / total
                        print('%d epoch; Accuracy of the network on the test images: %f %%' % (epoch + 1, 100 * test_acc))

                        epoch_list.append((epoch_c+1))
                        train_list.append(train_acc)
                        test_list.append(test_acc)
                        train_rate_list.append(train_rate)
                        test_rate_list.append(test_rate)

            if self.loss=='ODN':
                np.savetxt("./result100/{}_{}_{}_epoch_list.txt".format(self.dataset,self.hinge,self.opt),np.array(epoch_list))
                np.savetxt("./result100/{}_{}_{}_train_list.txt".format(self.dataset,self.hinge,self.opt),np.array(train_list))
                np.savetxt("./result100/{}_{}_{}_test_list.txt".format(self.dataset,self.hinge,self.opt),np.array(test_list))
                np.savetxt("./result100/{}_{}_{}_train_rate_list.txt".format(self.dataset,self.hinge,self.opt),np.array(train_rate_list))
                np.savetxt("./result100/{}_{}_{}_test_rate_list.txt".format(self.dataset,self.hinge,self.opt),np.array(test_rate_list))
            else:
                np.savetxt("./result100/{}_{}_{}_epoch_list.txt".format(self.dataset,self.loss,self.opt),np.array(epoch_list))
                np.savetxt("./result100/{}_{}_{}_train_list.txt".format(self.dataset,self.loss,self.opt),np.array(train_list))
                np.savetxt("./result100/{}_{}_{}_test_list.txt".format(self.dataset,self.loss,self.opt),np.array(test_list))

##################################################
for opt in ["SGD"]:
    print("\n"+"is_hinge="+loss+":")
    learner = is_hinge(batch_size=batch_size, is_gpu=is_gpu, dataset=dataset, epoch_num = (epoch_num),
         model=model, loss=loss, hinge=hinge, opt=opt, fraction=fraction)
    learner.fit()