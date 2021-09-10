import torch
import math
import torch.nn as nn
import torch.optim as optim
import numpy as np


class encoder(nn.Module):
    def __init__(self,M,num_hidden_units,n):
        super(encoder, self).__init__()
        self.enc1 = nn.Linear(M, num_hidden_units) # 符号化器用レイヤー
        self.act1 = torch.relu
        self.enc2 = nn.Linear(num_hidden_units, 2*n) # 符号化器用レイヤー
    def normalize(self, x): # 送信信号の正規化
        # 等電力制約
        #norm = torch.norm(x,dim=1).view(mbs, 1).expand(-1, 2) # Normalization layer
        #x = x/norm
        # 平均エネルギー制約
        mbs ,_,_ = x.shape
        norm = torch.sqrt((x.norm(dim=1)**2).sum()/mbs)
        x = x/norm
        return x


    def forward(self, m):
        s = self.enc1(m)
        s = self.act1(s)
        s = self.enc2(s)
        mbs, n_len= s.shape
        n = int(n_len/2)
        s = s.reshape(mbs,2,n)
        y = self.normalize(s) # normalization
        return y

class repeater(nn.Module):
    def __init__(self,num_hidden_units,n):
        super(repeater, self).__init__()
        self.mid1 = nn.Linear(1*n, num_hidden_units) # 符号化器用レイヤー
        self.act1 = torch.relu
        self.mid2 = nn.Linear(num_hidden_units,num_hidden_units*3)
        self.act2 = torch.relu
        self.mid3 = nn.Linear(num_hidden_units*3, 2*n) # 符号化器用レイヤー
    def normalize(self, x): # 送信信号の正規化
        # 等電力制約
        #norm = torch.norm(x,dim=1).view(mbs, 1).expand(-1, 2) # Normalization layer
        #x = x/norm
        # 平均エネルギー制約
        mbs ,_,_ = x.shape
        norm = torch.sqrt((x.norm(dim=1)**2).sum()/mbs)
        x = x/norm
        return x
    def detection(self,x):
        y = x[:,0,:]**2 + x[:,1,:]**2
        return y

    def forward(self, m):
        s = self.detection(m)
        s = self.mid1(s)
        s = self.act1(s)
        s = self.mid2(s)
        s = self.act2(s)
        s = self.mid3(s)
        mbs, n_len = s.shape
        n = int(n_len/2)
        s = s.reshape(mbs,2,n)
        y = self.normalize(s)
        return y


class decoder(nn.Module):
    def __init__(self,M,num_hidden_units,n):
        super(decoder, self).__init__()
        self.dec1 = nn.Linear(1*n, num_hidden_units) # 符号化器用レイヤー
        self.act1 = torch.relu
        self.dec2 = nn.Linear(num_hidden_units, M) # 符号化器用レイヤー
        self.softmax = torch.softmax

    def detection(self,x):
        y = x[:,0,:]**2 + x[:,0,:]**2
        return y

    def forward(self, m):
        s = self.detection(m)
        s = self.dec1(s)
        s = self.act1(s)
        s = self.dec2(s)
        y = self.softmax(s,dim=1)
        return y
