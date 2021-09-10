import torch
import math
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from model import encoder, decoder, repeater

def gen_minibatch(M,batch):
    one_hot_generator = torch.distributions.OneHotCategorical((1.0/M)*torch.ones(batch, M))
    return one_hot_generator.sample()
def train_rep(M,hidden,n,batch,sigma,epoch,learn_rate):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = encoder(M,hidden,n).to(device)
    rep = repeater(hidden,n).to(device)
    dec = decoder(M,hidden,n).to(device)
    loss_func = nn.MSELoss().to(device)
    enc_opt= optim.Adam(enc.parameters(), lr=learn_rate)
    rep_opt = optim.Adam(rep.parameters(),lr=learn_rate)
    dec_opt = optim.Adam(dec.parameters(),lr = learn_rate)
    enc.train()
    rep.train()
    dec.train()

    for i in range(epoch):
        m = gen_minibatch(M,batch)
        m = m.to(device)
        enc_opt.zero_grad()
        rep_opt.zero_grad()
        dec_opt.zero_grad()
        enc_sig = enc(m)
        shape = enc_sig.shape
        gauss = torch.normal(torch.zeros(shape),std=sigma).to(device)
        noisy = enc_sig +gauss
        mid_sig = rep(noisy)
        shape = mid_sig.shape
        gauss = torch.normal(torch.zeros(shape),std=sigma).to(device)
        noisy = enc_sig +gauss
        m_hat = dec(noisy)
        loss = loss_func(m_hat, m)
        loss.backward()
        enc_opt.step()
        rep_opt.step()
        dec_opt.step()
        if i % 1000 == 0:
            print(i, loss.item())



    return enc, rep, dec

def valid_rep(enc,rep,dec,M,batch,sigma):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = gen_minibatch(M,batch)
    loss_func = nn.MSELoss().to(device)
    enc = enc.to(device)
    rep = rep.to(device)
    dec = dec.to(device)
    enc.eval()
    rep.eval()
    dec.eval()
    with torch.no_grad():
        m = gen_minibatch(M,batch)
        m = m.to(device)
        enc_sig = enc(m)
        shape = enc_sig.shape
        gauss = torch.normal(torch.zeros(shape),std=sigma).to(device)
        noisy1 = enc_sig + gauss
        mid_sig = rep(noisy1)
        shape = mid_sig.shape
        gauss = torch.normal(torch.zeros(shape),std=sigma).to(device)
        noisy2 = mid_sig + gauss
        m_hat = dec(noisy2)

    score = 0
    m_np = m.detach().to("cpu").numpy()
    m_hat_np = m_hat.detach().to("cpu").numpy()
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
    m_re = mid_sig[:, 0,:].detach().numpy()
    m_im = mid_sig[:, 1,:].detach().numpy()
    y_re = noisy2[:, 0,:].detach().numpy()
    y_im = noisy2[:, 1,:].detach().numpy()
    x_amp = np.unique(x_re**2 + x_im**2)
    m_rep = np.unique(m_re**2+m_im**2)
    print(f"encoder sig points: {x_amp.shape}")
    print(f"repeater sig points: {m_rep.shape}")
    print(f"enc points :{x_amp}")
    fig = plt.figure(figsize=(5,5))
    plt.xlim([-3,3])
    plt.ylim([-3,3])
    plt.grid()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.scatter(x_re, x_im)
    fig.savefig("encoder.png")
    fig = plt.figure(figsize=(5,5))
    plt.xlim([-3,3])
    plt.ylim([-3,3])
    plt.grid()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.scatter(n_re, n_im)
    fig.savefig("noisy1.png")
    fig = plt.figure(figsize=(5,5))
    plt.xlim([-3,3])
    plt.ylim([-3,3])
    plt.grid()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.scatter(m_re, m_im)
    fig.savefig("repeater.png")
    fig = plt.figure(figsize=(5,5))
    plt.xlim([-3,3])
    plt.ylim([-3,3])
    plt.grid()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.scatter(y_re, y_im)
    fig.savefig("noisy2.png")
