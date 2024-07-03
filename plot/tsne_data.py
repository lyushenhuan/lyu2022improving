import sys
sys.path.append("..")
import torch
import torchvision
import numpy as np
import torchvision.transforms as transforms
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.utils.data as Data
from models.Nets import *
import os
from torch.autograd import Variable
from sklearn import manifold
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

pp = 100

#########################################################################
dataset_list = ['CIFAR10']#,'MNIST'
for dataset in dataset_list:
    if dataset=='CIFAR10':
        x = torch.load('/user/lvsh/exp_2018/ODN/data/CIFAR10/2600_features.pkl')
        y = torch.load('/user/lvsh/exp_2018/ODN/data/CIFAR10/2600_labels.pkl')
        classnum = 10
        transform = transforms.Compose(
            [transforms.Resize([224, 224]), transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        torch_dataset = Data.TensorDataset(data_tensor=x, target_tensor=y)
        trainloader = Data.DataLoader(
            dataset=torch_dataset,      # torch TensorDataset format
            batch_size=200,      # mini batch size
            shuffle=False,               # 要不要打乱数据 (打乱比较好)
            num_workers=4,              # 多线程来读数据
        )

        testset = torchvision.datasets.CIFAR10(root='/user/lvsh/exp_2018/ODN/data/CIFAR10', train=False,
                                               download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=200,
                                                 shuffle=False, num_workers=4)

    if dataset=='MNIST':
        x = torch.load('/user/lvsh/exp_2018/ODN/data/MNIST/300_features.pkl')
        y = torch.load('/user/lvsh/exp_2018/ODN/data/MNIST/300_labels.pkl')

        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        torch_dataset = Data.TensorDataset(data_tensor=x, target_tensor=y)
        trainloader = Data.DataLoader(
            dataset=torch_dataset,      # torch TensorDataset format
            batch_size=200,      # mini batch size
            shuffle=False,               # 要不要打乱数据 (打乱比较好)
            num_workers=4,              # 多线程来读数据
        )

        testset = torchvision.datasets.MNIST(root='/user/lvsh/exp_2018/ODN/data/MNIST', train=False,
                                       download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=200,
                                                 shuffle=False, num_workers=4)

    model_list = ['Xent', 'MLM', 'SMLM', 'ODN']
    for model in model_list:
        if dataset=='CIFAR10':
            net = AlexNet()
        if dataset=='MNIST':
            net = Net()
        net = net.cuda()
        net.load_state_dict(torch.load('./param/05frac_{}_{}_param.pkl'.format(dataset,model)))
        ###train
        # i = 0
        # for data in trainloader:
        #     images, labels = data
        #     outputs = net(Variable(images.cuda()))
        #     x = Variable(images.cuda())
        #     outputs = net(x)
        #     xxx = outputs.data
        #     if i == 0:
        #         X = xxx
        #         color = labels
        #     else:
        #         X = torch.cat((X, xxx), 0)
        #         color = torch.cat((color, labels), 0)
        #     i += 1
        
        # X = X.cpu().numpy()
        # color = color.cpu().numpy()
        # tsne = manifold.TSNE(n_components=2, init='random', random_state=0, perplexity=pp)
        # Y = tsne.fit_transform(X)
        # np.savetxt("./embed/{}_{}_train.txt".format(dataset,model),Y)
        # np.savetxt("./embed/{}_label_train.txt".format(dataset),color)

        #test
        i = 0
        for data in testloader:
            images, labels = data
            outputs = net(Variable(images.cuda()))
            x = Variable(images.cuda())
            outputs = net(x)
            xxx = outputs.data
            if i == 0:
                X = xxx
                color = labels
            else:
                X = torch.cat((X, xxx), 0)
                color = torch.cat((color, labels), 0)
            i += 1
        if dataset=='CIFAR10':
            num = 10000
        if dataset=='CIFAR10':
            num = 300
        X = X.cpu().numpy()[range(num)]
        color = color.cpu().numpy()[range(num)]
        tsne = manifold.TSNE(n_components=2, init='random', random_state=0, perplexity=pp)
        Y = tsne.fit_transform(X)
        np.savetxt("./embed/{}_{}_test.txt".format(dataset,model),Y)
        np.savetxt("./embed/{}_label_test.txt".format(dataset),color)