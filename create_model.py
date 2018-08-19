from __future__ import print_function, division
import argparse
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torchvision import datasets, transforms
from torch.autograd import Variable
#import torchsample as ts
import matplotlib.pyplot as plt
import numpy as np
import random as rd
import cv2

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=20, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=13, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, kernel_size=5)
        self.conv2 = nn.Conv2d(5, 5, kernel_size=4)
        self.conv3 = nn.Conv2d(5, 10, kernel_size=3)
        self.conv4 = nn.Conv2d(10, 20, kernel_size=3)
        self.conv5 = nn.Conv2d(20, 20, kernel_size=2)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # x = F.relu(F.max_pool2d(self.conv2(x), 2))
        # x = F.relu(F.max_pool2d(self.conv2_drop(self.conv3(x)), 2))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose(
                       [
                        # transforms.RandomHorizontalFlip(),
                    #    transforms.Pad(5),
                    #    transforms.Resize((28, 28)),
                       transforms.RandomRotation(10),
                    #    ts.transforms.Rotate((rd.randint(0, 30))),
                       transforms.ToTensor(),
                    #    transforms.Normalize((0.1307,), (0.3081,))
                   ]
                   )
                   ),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                    #    transforms.Pad(8),
                    #    transforms.Resize((28, 28)),
                    #    ts.transforms.Rotate(rd.randint(0, 30)),
                       transforms.RandomRotation(10),
                       transforms.ToTensor(),
                    #    transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)

model = Net()

if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

def trai(a):
    b, c = a.shape
    for i in range(c):
        if sum(a[:, i]) > 0:
            return i

def phai(a):
    b, c = a.shape
    for i in range(c-1, 0, -1):
        if sum(a[:, i]) > 0:
            return i

def tren(a):
    b, c = a.shape
    for i in range(b):
        if sum(a[i, :]) > 0:
            return i

def duoi(a):
    b, c = a.shape
    for i in range(b-1, 0, -1):
        if sum(a[i, :]) > 0:
            return i

def crop(a):
    top = np.random.randint(0, tren(a)+1)
    bottom = np.random.randint(duoi(a), 28)
    left = np.random.randint(0, trai(a)+1)
    right = np.random.randint(phai(a), 28)
    a1 = a[top: bottom, left: right]
    a1 = cv2.resize(a1, (28, 28)) 
    return a1

def padding(a):
    anh_pad = cv2.copyMakeBorder(a, top = rd.randint(0, 24), bottom=rd.randint(0, 24), left=rd.randint(0, 24), right=rd.randint(0, 24), borderType = 0, value = [0])
    a2 = cv2.resize(anh_pad, (28, 28))            
    return a2


def train(epoch):
    anh1 = np.zeros((20, 1, 28, 28), np.uint8)
    anh2 = np.zeros((20, 1, 28, 28), np.uint8)
    anh3 = np.zeros((20, 1, 28, 28), np.uint8)
    anh4 = np.zeros((20, 1, 28, 28), np.uint8)

    anh_gop = np.zeros((80, 1, 28, 28), np.uint8)
    anh_shuff = np.zeros((80, 1, 28, 28), np.uint8)
    tar = torch.LongTensor(80)
    tar1 = torch.LongTensor(80)

    a1 = np.zeros((28, 28), np.uint8)
    a2 = np.zeros((28, 28), np.uint8)
    a3 = np.zeros((28, 28), np.uint8)

    # tar = np.zeros((60), np.uint8)
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
	
        a = data.numpy()
        b = target.numpy()
        for i in range(20):
            tar[i] = int(b[i])
            tar[20+i] = int(b[i])
            tar[40+i] = int(b[i])
            tar[60+i] = int(b[i])    
	#tap anh 1: tap anh bi random crop
        for i in range(20):
            a1 = crop(a[i, 0, :, :])
            anh1[i, 0, :, :] = a1 > 0


#tap anh 2: tap anh binh thuong
        for i in range(20):
            # a2 = a
            anh2[i, 0, :, :] = a[i, 0, :, :] > 0


#tap anh 3: tap anh random padding
        for i in range(20):
            a2 = padding(a[i, 0, :, :])        
            anh3[i, 0, :, :] = a2>0

#tap anh 4: tap anh random padding & random crop
        for i in range(20):
            a3 = crop(a[i, 0, :, :])
            a3 = padding(a3)        
            anh4[i, 0, :, :] = a3>0

        anh1 = 1*anh1
        anh2 = 1*anh2
        anh3 = 1*anh3
        anh4 = 1*anh4

        for i in range(20):
            anh_gop[i, 0, :, :] = anh1[i, 0, :, :]
            anh_gop[20+i, 0, :, :] = anh2[i, 0, :, :]
            anh_gop[40+i, 0, :, :] = anh3[i, 0, :, :]
            anh_gop[60+i, 0, :, :] = anh4[i, 0, :, :]
        
        n = 0
        for i in range(80):
            while(n < 4):
                x = rd.randint(0, 27)
                y = rd.randint(0, 27)
                if anh_gop[i, 0, x, y] == 1:
                    anh_gop[i, 0, x, y] = 0
                    n += 1

	# tron random cac loai anh sau xu ly voi nhau  
        array = []
        for i in range(80):
            array.append(i)
        rd.shuffle(array)
        for i in range(80):
            anh_shuff[i, 0, :, :] = anh_gop[array[i], 0, :, :]
            tar1[i] = tar[array[i]]

        anh_tensor = torch.Tensor(anh_shuff)
        
        data = Variable(anh_tensor)
        target = Variable(tar1)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

def test():
    anh1 = np.zeros((1000, 1, 28, 28), np.uint8)
    a2 = np.zeros((28, 28), np.uint8)
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        
        b = data.numpy()
        for i in range(1000):
            a2 = crop(b[i, 0, :, :])
            anh1[i, 0, :, :] = a2 > 0

        anh1 = 1*anh1
        c = torch.Tensor(anh1)
        data, target = Variable(c), Variable(target)
        output = model(data)

        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def test_1():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set_1: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def test_2():
    anh3 = np.zeros((1000, 1, 28, 28), np.uint8)
    a3 = np.zeros((28, 28), np.uint8)
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        a = data.numpy()
        for i in range(1000):
            a2 = padding(a[i, 0, :, :])
            anh3[i, 0, :, :] = a2>0
   
        anh3 = 1*anh3
        c = torch.Tensor(anh3)
        data, target = Variable(c), Variable(target)        
#        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set_1: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def test_3():
    anh3 = np.zeros((1000, 1, 28, 28), np.uint8)
    a3 = np.zeros((28, 28), np.uint8)
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        a = data.numpy()
        for i in range(1000):
            a2 = crop(a[i, 0, :, :])
            a2 = padding(a2)
            anh3[i, 0, :, :] = a2>0
   
        anh3 = 1*anh3
        c = torch.Tensor(anh3)
        data, target = Variable(c), Variable(target)        
#        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set_1: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


for epoch in range(args.epochs+1):      #, args.epochs + 1):
    train(epoch)
    test()
    test_1()
    test_2()
    test_3()
    torch.save(model, '/home/lmhoang45/Desktop/model_19_8.pt')

