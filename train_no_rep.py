import torch
import math
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from model import encoder, decoder

def gen_minibatch(M,batch):
    one_hot_generator = torch.distributions.OneHotCategorical((1.0/M)*torch.ones(batch, M))
    return one_hot_generator.sample()
def train(M,hidden,n,batch,sigma,epoch,learn_rate,alpha):


    enc = encoder(M,hidden,n)
    dec = decoder(M,hidden,n)
    loss_func = nn.MSELoss()
    enc_opt= optim.Adam(enc.parameters(), lr=learn_rate)
    dec_opt = optim.Adam(dec.parameters(),lr = learn_rate)


    for i in range(epoch):
        m = gen_minibatch(M,batch)
        enc_opt.zero_grad()
        dec_opt.zero_grad()
        enc_sig = enc(m)
        shape = enc_sig.shape
        noisy = enc_sig + torch.normal(torch.zeros(shape),std=sigma*alpha)
        #noisy = noisy + torch.normal(torch.zeros(shape),std=sigma)
        m_hat = dec(noisy)
        loss = loss_func(m_hat, m)
        loss.backward()
        enc_opt.step()
        dec_opt.step()
        if i % 1000 == 0:
            print(i, loss.item())

    return enc,  dec

def valid(enc,dec,M,batch,sigma,alpha):
    m = gen_minibatch(M,batch)
    loss_func = nn.MSELoss()
    with torch.no_grad():
        m = gen_minibatch(M,batch)
        enc.zero_grad()
        dec.zero_grad()
        enc_sig = enc(m)
        shape = enc_sig.shape
        noisy1 = enc_sig + torch.normal(torch.zeros(shape),std=sigma*alpha)
        #noisy2 = noisy1 + torch.normal(torch.zeros(shape),std=sigma)
        noisy2 = noisy1
        m_hat = dec(noisy2)

    score = 0
    m_np = m.detach().numpy()
    m_hat_np = m_hat.detach().numpy()
    for ans , res in zip(m_np,m_hat_np):
      if np.where(ans ==1) == np.where(res == np.max(res)):
        score +=1
    mbs ,_ = m_hat.shape
    print(score/mbs)
    print(loss_func(m,m_hat))
    x_re = enc_sig[:, 0,:].detach().numpy()
    x_im = enc_sig[:, 1,:].detach().numpy()
    n_re = noisy1[:, 0,:].detach().numpy()
    n_im = noisy1[:, 1,:].detach().numpy()
    y_re = noisy2[:, 0,:].detach().numpy()
    y_im = noisy2[:, 1,:].detach().numpy()
    x_amp = np.unique(x_re**2 + x_im**2)
    print(f"encoder sig points: {x_amp.shape}")
    print(f"enc points :{x_amp}")
    fig = plt.figure(figsize=(5,5))
    plt.xlim([-3,3])
    plt.ylim([-3,3])
    plt.grid()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.scatter(x_re, x_im)
    fig.savefig("encoder_no_rep.png")
    fig = plt.figure(figsize=(5,5))
    plt.xlim([-3,3])
    plt.ylim([-3,3])
    plt.grid()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.scatter(n_re, n_im)
    fig.savefig("noisy1_no_rep.png")
    fig = plt.figure(figsize=(5,5))
    plt.xlim([-3,3])
    plt.ylim([-3,3])
    plt.grid()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.scatter(y_re, y_im)
    fig.savefig("noisy2_no_rep.png")
