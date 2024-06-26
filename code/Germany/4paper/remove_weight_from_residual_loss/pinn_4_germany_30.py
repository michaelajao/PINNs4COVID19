# -*- coding: utf-8 -*-

import os
import sys
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from scipy import integrate

# 将上两级目录加入到系统路径中（constants.py所在目录）
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
from data.constants_util import *
import numpy as np
import pandas as pd

torch.manual_seed(0)


# 加载数据
country = 'Germany'
N = 84323763
days =117

data = pd.read_csv('../../4paper/remove_weight_from_residual_loss/84323763_0_117_0_53_real_data_estimated.csv')

# plot susceptibles
plt.plot(data['susceptibles'])
plt.show()


pf = pd.read_csv('../remove_weight_from_residual_loss/84323763_0_117_0_53_real_data_estimated.csv')/N
# date = np.array(pf['date'])
date = np.array(pd.date_range(start='03/06/2021',periods=days,normalize=True).strftime('%Y-%m-%d'))

# 归一化，便于计算 infectives,removed,susceptibles
I_raw = np.array(pf['infectives'])*25
R_raw = np.array(pf['removed'])*15
N = N/N
S_raw = N-I_raw-R_raw

# plot S_raw
plt.plot(pf['infectives'])
plt.show()


top = int(np.argmax(I_raw))

data_size = days
# 训练集/测试集
train_size = 30
test_size = -train_size

device = get_device()

# SIR model 输出为单位时间内的每个舱室的个体数
def covid_sir(u, t, beta, gamma):
    S, I, R= u
    dS_dt = -beta*S*I
    dI_dt = beta*S*I - gamma*I
    dR_dt = gamma*I

    return np.array([dS_dt, dI_dt, dR_dt])


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.001)


# SIR NN
class Net_SIR(nn.Module):
    def __init__(self, layers=None,):
        super(Net_SIR, self).__init__()

        self.layers = torch.nn.Sequential(
            nn.Linear(1,64),nn.Tanh(), # 1 means time
            nn.Linear(64,64),nn.Tanh(),
            nn.Linear(64,3),nn.Sigmoid() # values of each compartments relative to SIR 
        )

    def forward(self, x):
        output = self.layers(x)
        return output

    def set_layer(self, layers):
        self.layers = layers


# 初始化 NNs
pinn_sir = Net_SIR()
# pinn_sir.apply(init_weights)
pinn_sir = pinn_sir.to(device)


def network(t):
    input = t.reshape(-1,1)
    input = torch.from_numpy(input).float().to(device)
    return pinn_sir(input)*N


# LOSS = data loss + residual loss
def residual_loss(t,beta,gamma):
    # 以天为单位
    dt = 1
    u = network(t)

    st2, st1 = u[1:,0], u[:-1,0]
    it2, it1 = u[1:,1], u[:-1,1]
    rt2, rt1 = u[1:,2], u[:-1,2]

    ds = (st2 - st1)/dt
    di = (it2 - it1)/dt
    dr = (rt2 - rt1)/dt

    loss_s = (-(beta)*st1*it1-ds)**2
    loss_i = ((beta)*st1*it1-(gamma)*it1-di)**2
    loss_r = ((gamma)*it1-dr)**2
    loss_n = (u[:,0]+u[:,1]+u[:,2]- N)**2 # normalization constraint
    return loss_s.sum(), loss_i.sum(), loss_r.sum(), loss_n.sum()

  
def init_loss(t,S0,I0,R0):
    u = network(t)

    loss_s0 = (u[0,0] - S0)**2
    loss_i0 = (u[0,1] - I0)**2
    loss_r0 = (u[0,2] - R0)**2

    return loss_s0+ loss_i0+ loss_r0


def data_loss(t,index):
    u = network(t)
    loss_st = (u[index,0] - torch.from_numpy(S_raw[index]).to(device))**2
    loss_it = (u[index,1] - torch.from_numpy(I_raw[index]).to(device))**2
    loss_rt = (u[index,2] - torch.from_numpy(R_raw[index]).to(device))**2
    return loss_st.sum(), loss_it.sum(), loss_rt.sum()


# fixed 参数
beta_raw = 0.25 
gamma_raw = 0.15

nn_parameters = list(pinn_sir.parameters())

optimizer = optim.Adam(nn_parameters, lr=1e-4, weight_decay=0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.998)

beta_trained = Variable(torch.tensor([beta_raw]).to(device), requires_grad=True)
gamma_trained = Variable(torch.tensor([gamma_raw]).to(device), requires_grad=True)

optimizer_v = optim.Adam([beta_trained, gamma_trained], lr=1e-4, weight_decay=0.01)
scheduler_v = optim.lr_scheduler.StepLR(optimizer_v, step_size=5000, gamma=0.998)

S0 = S_raw[0]
I0 = I_raw[0]
R0 = R_raw[0]

# 训练集离散时间点
t_points = np.linspace(0,days,days+1)[:-1]
# 训练集时间点shuffle
index = torch.randperm(train_size)

total_epoch = []
total_data_loss_epoch = []
total_residual_loss_epoch = []

early_stopping = 250
#双头队列
loss_history = deque(maxlen=early_stopping + 1)

alpha_0 = 150
alpha_1 = 60000
alpha_2 = 50

pinn_sir.zero_grad()

for epoch in range(200000):  # loop over the dataset multiple times
    running_loss = 0.0        
    # zero the parameter gradients
    optimizer.zero_grad()
    optimizer_v.zero_grad()

    loss_s,loss_i,loss_r,loss_n = residual_loss(t_points,beta_trained,gamma_trained)
    loss_st, loss_it, loss_rt = data_loss(t_points, index)
    loss_init = init_loss(t_points,S0,I0,R0)

    loss = (alpha_0)*(loss_st+ 750*loss_it+ 300*loss_rt)+ (alpha_1)*(loss_s+ loss_i+ loss_r+ loss_n)+alpha_2*loss_init

    running_loss += loss.item()
    loss_history.append(running_loss)
    total_epoch.append(running_loss)
    
    total_data_loss_epoch.append((loss_st.item()+loss_it.item()+loss_rt.item()))
    total_residual_loss_epoch.append((loss_s+loss_i+loss_r+loss_n).item())

    # 如果队列满了并且弹出的第一个值小于队列中剩余的最小值，表明该值为最小值，在队列长度个epoch内loss不再下降
    if len(loss_history) > early_stopping and loss_history.popleft() < min(loss_history):
      print(f"Early stopping at [{epoch}] times. No train loss improvement in [{early_stopping}] epochs.")
      break
    
    loss.backward()
    optimizer.step()
    scheduler.step()

    optimizer_v.step()
    scheduler_v.step()
    
    if (epoch % 1000 == 0):   
        lr = optimizer.param_groups[0]['lr']
        print(f'epoch = {epoch}, loss: {running_loss}, lr: {lr}.')
        print(f'data loss: {(loss_st.item()+loss_it.item()+loss_rt.item())}, S loss: {loss_st.item()}, I loss: {loss_it.item()}, R loss: {loss_rt.item()}.')
        print(f'residual loss: {(loss_s+loss_i+loss_r+loss_n).item()}, loss_s: {loss_s.item()}, loss_i: {loss_i.item()}, loss_r: {loss_r.item()}, loss_n: {loss_n.item()}.')
        print(f'init loss: {loss_init}.')

print('Finished Training')


# 绘制loss图
plot_log_loss(country, total_epoch,total_data_loss_epoch,total_residual_loss_epoch,train_size)

t = np.linspace(0,days,days+1)[:-1]

u = network(t)
st = u[:,0].cpu().detach().numpy()
it = u[:,1].cpu().detach().numpy()
rt = u[:,2].cpu().detach().numpy()

# plot st vs real data
plt.plot(st, label='st', color='red')
plt.plot(S_raw, label='S_real', color='black')
plt.legend()
plt.show()

beta, gamma = beta_trained.cpu().detach().numpy(), gamma_trained.cpu().detach().numpy()

# change beta and gamma to integr type rather than array
beta = beta[0]
gamma = gamma[0]

# 利用训练参数求解ODE SIR
def covid_sir(u, t, beta, gamma):
    S, I, R = u
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

u0 = [S0, I0, R0]
res = integrate.odeint(covid_sir, u0, t, args=(beta, gamma))
S_ode, I_ode, R_ode = res.T

def plot_result_comparation(country, pre_data, data_type, real_data, ode_data,train_size):
    if not os.path.exists(path_results):
        os.makedirs(path_results)

    plt.figure(figsize=(16,9))
    t = np.linspace(0,len(pre_data),len(pre_data)+1)[:-1]

    plt.plot(t, real_data, color ='black' ,label=f'{data_type}_Real') 
    plt.scatter(t[:train_size], real_data[:train_size], color ='black', marker='*', label=f'{data_type}_Train')  # type: ignore
    plt.plot(t, pre_data, color ='red' ,label=f'{data_type}_PINNs') 
    if data_type != 'Ic':
        plt.plot(t, ode_data, color ='green' ,label=f'{data_type}_SIR') 

    plt.xlabel('Time t (days)', fontsize=25)
    plt.ylabel('Numbers of individuals', fontsize=25)

    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.legend(fontsize=25)

    # plt.savefig(path_results+f'{country}_{data_type}_{train_size}_result.pdf', dpi=600)
    plt.show()

plot_result_comparation(country,st,'S',S_raw,S_ode,train_size)
plot_result_comparation(country,it,'I',I_raw,I_ode,train_size)
plot_result_comparation(country,rt,'R',R_raw,R_ode,train_size)

# 保存结果
save_results_fixed(country,date,st,it,rt,S_raw,I_raw,R_raw,S_ode,I_ode,R_ode,train_size)

# 保存MSE和MAE
save_error_result_sir(country,S_raw,I_raw,R_raw,st,it,rt,S_ode,I_ode,R_ode,train_size)

# 保存参数学习结果
save_parameters_learned(country,beta_raw,gamma_raw,beta,gamma,train_size)

# 打印beta和gamma
print(f'peak index is: {top}.')
print(f'alpha_0: {alpha_0}, alpha_1: {alpha_1}, alpha_2: {alpha_2}.')
print(f'beta_raw: {beta_raw}, gamma_raw: {gamma_raw}.')
print(f'beta and gamma are: {beta}, {gamma}.')
