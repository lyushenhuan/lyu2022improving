import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch

class OptimalMarginDistributionLoss(nn.Module):
    def __init__(self, hinge=0, gamma=1.2, theta=0.7, mu=0.1, class_num=10):
        super(OptimalMarginDistributionLoss, self).__init__()
        self.hinge = hinge
        self.gamma = gamma
        self.theta = theta
        self.mu = mu
        self.class_num = class_num

    def forward(self, outputs, labels):
        margin = margin_calc(outputs, labels, self.class_num) / 0.4
        L = (margin <= (self.gamma - self.theta)).float()*(self.hinge*torch.abs(margin - 
                self.gamma + self.theta) / 0.5 + (1-self.hinge)*(torch.pow((margin - 
                self.gamma + self.theta), 2) / (self.gamma - self.theta)**2)) + (margin >= (self.gamma + 
                self.theta)).float()*self.mu*  (1-self.hinge)*(torch.pow((margin - 
                self.gamma - self.theta), 2) / (self.gamma + self.theta)**2)
        loss = L.sum() / L.size(0)       
        return loss


def margin_calc(outputs, labels, class_num=10):         #  Margin calculate
    batchs = labels.size(0)
    f = Variable(torch.zeros([batchs]).cuda())       #is_gpu=1
    out_w = torch.zeros([batchs, class_num]).cuda()  #is_gpu=1
    out_w.copy_(outputs.data, async=True)
    for j in range(batchs):
        out_w[j, labels.data[j]] = -999
    _, m = torch.max(out_w, 1)
    for j in range(batchs):
        f[j] = outputs[j, labels.data[j]] - outputs[j, m[j]]
    return f


class SoftMarginLoss(nn.Module):
    def __init__(self, alpha = 0.8, class_num = 10):
        super(SoftMarginLoss, self).__init__()
        self.alpha = alpha
        self.class_num = class_num

    def forward(self, outputs, labels):
        batchs = labels.size(0)
        targets = torch.zeros([batchs, self.class_num]).cuda()
        for i in range(batchs):
            targets[i, labels.data[i]] = 1
        targets = Variable(targets)
        loss = self.alpha * F.multilabel_soft_margin_loss(outputs, targets) + (1-self.alpha) * F.cross_entropy(outputs, labels)
        return loss