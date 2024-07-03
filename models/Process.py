import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from models.Nets import *
from models.Loss import *
from torch.utils.data import *
from torch.autograd import Variable
from sklearn.utils import shuffle

class Subset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

class Data_loader():
    def __init__(self, dataset_name="mnist", batch_size=200, fraction=100):
        super(Data_loader, self).__init__()
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.fraction = fraction

    def mnist_loader(self):
        transform = transforms.Compose([transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        tvset = torchvision.datasets.MNIST(root='../dataset/MNIST', train=True,
                                                download=True, transform=transform)
        
        if self.fraction<100:
            num = int(self.fraction*60000/100.0)
            sample_idx = shuffle(range(0,60000))
            sample_idx = sample_idx[0:num]
            tvset = Subset(tvset, sample_idx)
            tvloader = torch.utils.data.DataLoader(dataset = tvset, batch_size = self.batch_size,
                                                shuffle=True, num_workers=4)
        tvloader = torch.utils.data.DataLoader(tvset, batch_size=self.batch_size,
                                         shuffle=True, num_workers=4)

        for train_idx, valid_idx in k_folds(n_splits = 5, subjects = 60000):
                dataset_train = Subset(tvset, train_idx)
                dataset_valid = Subset(tvset, valid_idx)
                trainloader = torch.utils.data.DataLoader(dataset = dataset_train, batch_size = self.batch_size, shuffle=True, num_workers=4)
                validloader = torch.utils.data.DataLoader(dataset = dataset_valid, batch_size = self.batch_size, shuffle=True, num_workers=4)

        testset = torchvision.datasets.MNIST(root='../dataset/MNIST', train=False,
                                       download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size,
                                         shuffle=False, num_workers=4)
        return [tvloader, trainloader, validloader, testloader]

    def cifar10_loader(self):
        transform = transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor(),#
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


        tvset = torchvision.datasets.CIFAR10(root='../dataset/CIFAR10', train=True,
                                                download=True, transform=transform)

        if self.fraction<100:
            num = int(self.fraction*50000/100.0)
            sample_idx = shuffle(range(0,50000))
            sample_idx = sample_idx[0:num]
            tvset = Subset(tvset, sample_idx)
            tvloader = torch.utils.data.DataLoader(dataset = tvset, batch_size = self.batch_size,
                                                shuffle=True, num_workers=4)
        tvloader = torch.utils.data.DataLoader(tvset, batch_size=self.batch_size,
                                         shuffle=True, num_workers=4)

        for train_idx, valid_idx in k_folds(n_splits = 5, subjects = 50000):
                dataset_train = Subset(tvset, train_idx)
                dataset_valid = Subset(tvset, valid_idx)
                trainloader = torch.utils.data.DataLoader(dataset = dataset_train, batch_size = self.batch_size, shuffle=True, num_workers=4)
                validloader = torch.utils.data.DataLoader(dataset = dataset_valid, batch_size = self.batch_size, shuffle=True, num_workers=4)

        testset = torchvision.datasets.CIFAR10(root='../dataset/CIFAR10', train=False,
                                               download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size,
                                                 shuffle=False, num_workers=4)

        return [tvloader, trainloader, validloader, testloader]
    

def data_transform(dataset="mnist", batch_size=200, fraction=100):
    data_loader = Data_loader(dataset, batch_size, fraction)
    if dataset=="mnist":
        tvloader, trainloader, validloader, testloader = data_loader.mnist_loader()
    if dataset=="cifar10":
        tvloader, trainloader, validloader, testloader = data_loader.cifar10_loader()
    return [tvloader, trainloader, validloader, testloader]	




# Developer: Alejandro Debus
# Email: aledebus@gmail.com

def partitions(number, k):
    '''
    Distribution of the folds
    Args:
        number: number of patients
        k: folds number
    '''
    n_partitions = np.ones(k) * int(number/k)
    n_partitions[0:(number % k)] += 1
    return n_partitions

def get_indices(n_splits = 5, subjects = 50000, frames = 1):
    '''
    Indices of the set test
    Args:
        n_splits: folds number
        subjects: number of patients
        frames: length of the sequence of each patient
    '''
    l = partitions(subjects, n_splits)
    fold_sizes = l * frames
    indices = np.arange(subjects * frames).astype(int)
    current = 0
    for fold_size in fold_sizes:
        start = current
        stop =  current + fold_size
        current = stop
        yield(indices[int(start):int(stop)])

def k_folds(n_splits = 5, subjects = 50000, frames = 1):
    '''
    Generates folds for cross validation
    Args:
        n_splits: folds number
        subjects: number of patients
        frames: length of the sequence of each patient
    '''
    indices = np.arange(subjects * frames).astype(int)
    for test_idx in get_indices(n_splits, subjects, frames):
        train_idx = np.setdiff1d(indices, test_idx)
        yield train_idx, test_idx


class train_valid():

    def __init__(self, batch_size=200, is_gpu=0, dataset="mnist", epoch_num=200, model="LeNet", loss="xent", hinge=1, opt="SGD"):
        super(train_valid, self).__init__()
        self.batch_size = batch_size
        self.is_gpu = is_gpu
        self.dataset = dataset
        self.epoch_num = epoch_num
        self.model = model
        self.loss = loss
        self.opt = opt
        self.hinge = hinge

    def fit(self):


        if self.dataset=="mnist":
            samples = 60000
            classnum = 10
        if self.dataset=="cifar10":
            samples = 50000
            classnum = 10

        #####Data transform######
        tvloader, trainloader, validloader, testloader = data_transform(dataset=self.dataset, batch_size=self.batch_size)
        #####Nets#####
        if self.model=="LeNet":
            net = Net(num_classes=classnum)
        if self.model=="AlexNet":
            net = AlexNet(num_classes=classnum)
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
        if self.opt=="SGD":                                                                                # hinge=0, gamma=1.7, theta=0.3, mu=0.1, class_num=10
            optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        if self.opt=="Adam":
            optimizer = optim.Adam(net.parameters(), lr=0.001)
        if self.opt=="RMSprop":
            optimizer = optim.RMSprop(net.parameters())

        best_valid_acc = 0
        for epoch in range(self.epoch_num):
            temp_valid_acc = 0    
            for data in trainloader:
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

            correct = 0
            total = 0
            for data in validloader:
                images, labels = data
                if self.is_gpu==0:
                    outputs = net(Variable(images))
                if self.is_gpu==1:
                    outputs = net(Variable(images.cuda()))
                    outputs = outputs.cpu()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
            temp_valid_acc += correct / total
            print('%d epoch; Accuracy of the network on the valid images: %f %%' % (epoch + 1, 100 * temp_valid_acc))
            if best_valid_acc > temp_valid_acc:
                bad += 1
            else:
                bad = 0
                best_valid_acc = temp_valid_acc
            if bad >= 5:
                break

class train_test():

    def __init__(self, batch_size=200, is_gpu=0, dataset="mnist", epoch_num=200, model="LeNet", loss="xent", hinge=1, opt="SGD", fraction=100):
        super(train_test, self).__init__()
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
            net = AlexNet(num_classes=classnum)
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

        correct = 0
        total = 0
        for data in testloader:
            images, labels = data
            if self.is_gpu==0:
                outputs = net(Variable(images))
            if self.is_gpu==1:
                outputs = net(Variable(images.cuda()))
                outputs = outputs.cpu()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        test_acc = correct / total
        print('%d epoch; Accuracy of the network on the 10000 test images: %f %%' % (epoch + 1, 100 * test_acc))