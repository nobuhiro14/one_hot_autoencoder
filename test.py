import torch
import math
import torch.nn as nn
import torch.optim as optim
import numpy as np
from argparse import ArgumentParser

from train import train_rep,valid_rep
from train_no_rep import train,valid
from train_classify import train_cl, valid_cl
## python3 test.py
class Option():
    def __init__(self):
        parser = ArgumentParser()
        parser.add_argument("-one_hot",type = int,default=4)
        parser.add_argument("-hidden", type = int,default=10)
        parser.add_argument("-channel",type = int,default=1)
        parser.add_argument("-batch",type = int,default=100)
        parser.add_argument("-gauss",type = float,default=0.1)
        parser.add_argument("-epoch",type = int,default=8000)
        parser.add_argument("-learn_rate",type = float,default=0.01)
        parser.add_argument("-mode",choices=["no_rep","rep","class0","class1","class2"])
        self.parser = parser

    def get_param(self):
        return self.parser.parse_args()


if __name__ == "__main__":
    args = Option().get_param()
    m = args.one_hot
    hidden = args.hidden
    n = args.channel
    batch = args.batch
    sigma = args.gauss
    ep = args.epoch
    lr = args.learn_rate
    if args.mode =="rep" :

        enc, rep ,dec = train_rep(m,hidden,n,batch,sigma,ep,lr)
        valid_rep(enc,rep,dec,m,batch,sigma)
    elif args.mode =="no_rep" :
        enc, dec = train(m,hidden,n,batch,sigma,ep,lr)
        valid(enc,dec,m,batch,sigma)

    elif args.mode =="class0":
        enc, rep ,dec = train_cl(m,hidden,n,batch,sigma,ep,lr,0)
        valid_cl(enc,rep,dec,m,batch,sigma,0)
    elif args.mode =="class1":
        enc, rep ,dec = train_cl(m,hidden,n,batch,sigma,ep,lr,1)
        valid_cl(enc,rep,dec,m,batch,sigma,1)
    elif args.mode =="class2":
        enc, rep ,dec = train_cl(m,hidden,n,batch,sigma,ep,lr,2)
        valid_cl(enc,rep,dec,m,batch,sigma,2)
    else :
        print(f"{args.mode} is not available")
