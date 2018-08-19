from __future__ import print_function
import cv2
import numpy as np
from scipy import misc
import pylab as pl
from matplotlib import pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import os
from skimage.morphology import skeletonize
from skimage.util import invert
from PIL import Image
from torchvision.transforms import ToTensor as tensor
from torchvision.transforms import Normalize as normal

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

cap = cv2.VideoCapture(0)
kernel_1 = np.ones((1, 1),np.uint8)
kernel_3 = np.ones((3, 3),np.uint8)

def lay_anh(frame, stt):
    win = np.empty((280, 280, 3), np.uint8)
    win[:, :, :] = frame[80:360, 180:460, :]    #lay mot phan cua so window cua frame

    draft = cv2.resize(win, (280, 280)) 
    
    gray_image = cv2.cvtColor(draft, cv2.COLOR_BGR2GRAY)    #convert sang anh muc xam
    #nhi phan anh dung adaptive threshold
    thres = cv2.adaptiveThreshold(gray_image, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # z, thres = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY, cv2.THRESH_OTSU)
    nen_den = 1-thres
    loc_nhieu_2 = cv2.medianBlur(nen_den, 3)

    for i in range(15):                 #loc nhieu trong anh dung Median Blurring
        loc_nhieu_2 = cv2.medianBlur(loc_nhieu_2, 3)
    
    dilation = cv2.dilate(loc_nhieu_2, kernel_3, iterations = 6)
    # dilation_thres = np.zeros((280, 280), np.uint8)
    dilation_thres = 1*(dilation>0)
    dilation_thres = np.asarray(dilation_thres, dtype= np.uint8)
    #remove connectedComponents
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(dilation_thres, connectivity=8)
    sizes = stats[1:, -1]
    nb_components = nb_components - 1
    min_size = 1000
    img2 = np.zeros((output.shape))

    for i in range(nb_components):
	    if sizes[i] >= min_size:
		    img2[output == i + 1] = 1
    # print("Kich thuoc dilation", dilation_thres)
    # dilation_thres = cv2.adaptiveThreshold(gray_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 2)
    gray_image1 = cv2.resize(img2, (0, 0), fx = 0.1, fy = 0.1, interpolation=cv2.INTER_AREA) 
    gray_image1 = 1*(gray_image1>0)
    # gray_image1 = np.asarray(gray_image1, dtype= np.uint8)
    # print(gray_image1)
    # print("gray_image1 la: ", gray_image1)
    # gray_image1_thres = cv2.adaptiveThreshold(gray_image1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # print("Loc_nhieu la: ", loc_nhieu_2)
    stt = str(stt)
    anh_to = '/home/lmhoang45/Desktop/anh/to' + stt + '.png'
    anh_nho = '/home/lmhoang45/Desktop/anh/nho' + stt + '.png'
    # cv2.imwrite('/home/lmhoang45/Desktop/to.jpg', img2)
    # cv2.imwrite('/home/lmhoang45/Desktop/nho.jpg', gray_image1)
    cv2.imwrite(anh_to, img2)
    cv2.imwrite(anh_nho, gray_image1)

model1 = torch.load('/home/lmhoang45/Desktop/model_19_8.pt')

# anh1 = np.empty((28, 28), np.uint8)
def test_nhap(stt):
    model1.eval()
    test_loss = 0
    correct = 0

    stt = str(stt)
    ten_nho = '/home/lmhoang45/Desktop/anh/nho' + stt + '.png'
    ten_to = '/home/lmhoang45/Desktop/anh/to' + stt + '.png'
    #load image
    #anh = cv2.imread(ten_nho, 0)
    anh = cv2.imread(ten_to, 0)
    anh = cv2.resize(anh, (28, 28))
    anh_thres = 1*(anh>0)
    # print("anh_thres la: ", anh_thres)
    # anh_thres = np.asarray(anh_thres, dtype= np.uint8)
    # print("anh la: ", anh)
    # z, anh1 = cv2.threshold(anh, 0, 255, cv2.THRESH_BINARY, cv2.THRESH_OTSU)
    # anh1 = 255 - cv2.adaptiveThreshold(anh, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    b = cv2.imread(ten_to, 0)
    b_thres = 255*(b>0)
    b_thres = np.asarray(b_thres, dtype= np.uint8)
    cv2.imshow("Anh dau vao", b_thres)
#    print(type(b_thres))
    
    a = np.empty((1, 1, 28, 28), np.float)    
    a[0, 0, :, :] = anh_thres
    c = torch.Tensor(a)
    # print(c)
    d = Variable(c)
    # d = c.numpy()
    # anh1 = d[0, 0, :, :]
    # plt.imshow(anh1)
    # plt.show()
    # print(d)

    output = model1(d)
    print(output)
    np.sum(output)
    _, predicted = torch.max(output.data, 1)
    
    return predicted
    
t=0
while(True):
    
    (ret, frame_ori) = cap.read()
    frame1 = cv2.flip(frame_ori, 1)
    frame2 = cv2.flip(frame1, 1);
    cv2.rectangle(frame1 ,(180, 80),(460, 360),(255,0,0), 1)
    cv2.imshow("Hello", frame1)
    
    c = cv2.waitKey(1)
    d = c & 0xFF
    if d == ord('a'):
        t+=1
        lay_anh(frame2, t)
        test_nhap(t)
        print("Lan thu %d" %t, test_nhap(t))
        for i in range(3):
            print('\n')
    
    if d == ord('z'):
        break

    
        
