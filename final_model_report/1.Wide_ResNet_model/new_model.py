# 1. import

import os
import time

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

import sys 
sys.path.append('..')

## 참고 ) random.seed(seed)
import random
import torch
import numpy as np
from tqdm import tqdm





### 2. M_relu 정의

class M_relu(nn.Module):    
    def __init__(self, input_dim, init_active_dim):
            super().__init__()
            self.input_dim = input_dim
            self.init_active_dim = init_active_dim

            self.active = nn.Parameter(torch.cuda.FloatTensor([1]*self.init_active_dim), requires_grad=False)      

    def forward(self, x):
        if len(x.shape)==2:            
            M = self.active.view(1,-1)       
            return M * F.relu(x)
        elif len(x.shape)==4:            
            M = self.active.view(1,-1,1,1)       
            return M * F.relu(x)
        



### 3. M_relu를 적용한 BasicBlock 정의

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)

class M_BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(M_BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.M_relu1 = M_relu(planes,planes)
        self.dropout = nn.Dropout(p=dropout_rate)  
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.M_relu2 = M_relu(planes,planes)
        

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion* planes: 
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.M_relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.M_relu2(out)
        return out

    def get_2norm(self):
        norm1 = filter_ranks(self.conv1, self.bn1) #filter_ranks?? 
        norm2 = filter_ranks(self.conv2, self.bn2)
        if len(self.shortcut)!=0:
            norm2 += filter_ranks(self.shortcut[0], self.shortcut[1])            
        return [norm1, norm2]    



### 4. filter_ranks
def filter_ranks(conv, bn):
    bn_scale = bn.weight.data / torch.sqrt(bn.running_var + bn.eps)
    new_filter = bn_scale.reshape(-1,1,1,1) * conv.weight.data
    new_bias = bn.bias.data - bn_scale*bn.running_mean
    return (torch.pow(new_filter, 2)).sum((1,2,3))
 #+ torch.pow(new_bias,2)# 


### 5. Basic Block을 적용한 Wide_Resnet 모델 적용

class Wide_ResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes):
        super(Wide_ResNet, self).__init__()
        self.in_planes = 16

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(3,nStages[0]) #3, 16
        self.bn1 = nn.BatchNorm2d(nStages[0], momentum=0.9) #16
        self.M_relu = M_relu(nStages[0],nStages[0]) #16, 16 
        self.layer1 = self._wide_layer(M_BasicBlock, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(M_BasicBlock, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(M_BasicBlock, nStages[3], n, dropout_rate, stride=2)
        self.linear = nn.Linear(nStages[3]*M_BasicBlock.expansion, num_classes)
        self.register_buffer('sigma', torch.tensor([1.0]).cuda())

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.M_relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out

    
    
import smtplib
from email.mime.text import MIMEText
# 참고 : https://yeolco.tistory.com/93
# gmail_id와 password만 각자 자기꺼 입력

def send_email(email_text ='' , gmail_id='dusgh9514@snu.ac.kr', password = 'xfwzpgxgmcxbvzcd'):
    
    # 세션 생성
    s = smtplib.SMTP('smtp.gmail.com', 587)

    # TLS 보안 시작
    s.starttls()

    # 로그인 인증
    s.login(gmail_id, password)
    
    # 보낼 메시지 설정
    msg = MIMEText(email_text)
    msg['Subject'] = email_text
    
    # 메일 보내기
    s.sendmail(gmail_id, gmail_id, msg.as_string())

    # 세션 종료
    s.quit()

