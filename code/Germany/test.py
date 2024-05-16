import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from collections import deque
import numpy as np
import pandas as pd
from scipy import integrate
from prettytable import PrettyTable
from sklearn import metrics
import matplotlib.pyplot as plt

# Constants and utilities
path_results = '../results/'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(0)
alpha_0 = 150
alpha_1 = 60000
alpha_2 = 50

def get_device():
    print('Using device:', device)
    if device.type == 'cuda':
        print(f'Using device: {torch.cuda.get_device_name(0)}')
        print(f'Memory Usage: Allocated: {round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1)} GB, Cached: {round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1)} GB')
        torch.cuda.set_device(0)
    return device

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.001)

class Net_SIR(nn.Module):
    def __init__(self):
        super(Net_SIR, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 3), nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)

def covid_sir(u, t, beta, gamma):
    S, I, R = u
    dS_dt = -beta * S * I
    dI_dt = beta * S * I - gamma * I
    dR_dt = gamma * I
    return [dS_dt, dI_dt, dR_dt]

def network(t):
    input = t.reshape(-1, 1)
    input = torch.from_numpy(input).float().to(device)
    return pinn_sir(input) * N

def residual_loss(t, beta, gamma):
    dt = 1
    u = network(t)
    st2, st1 = u[1:, 0], u[:-1, 0]
    it2, it1 = u[1:, 1], u[:-1, 1]
    rt2, rt1 = u[1:, 2], u[:-1, 2]
    ds, di, dr = (st2 - st1) / dt, (it2 - it1) / dt, (rt2 - rt1) / dt
    loss_s = ((-beta * st1 * it1 - ds) ** 2).sum()
    loss_i = ((beta * st1 * it1 - gamma * it1 - di) ** 2).sum()
    loss_r = ((gamma * it1 - dr) ** 2).sum()
    loss_n = ((u[:, 0] + u[:, 1] + u[:, 2] - N) ** 2).sum()
    return loss_s, loss_i, loss_r, loss_n

def init_loss(t, S0, I0, R0):
    u = network(t)
    loss_s0 = (u[0, 0] - S0) ** 2
    loss_i0 = (u[0, 1] - I0) ** 2
    loss_r0 = (u[0, 2] - R0) ** 2
    return loss_s0 + loss_i0 + loss_r0

def data_loss(t, index):
    u = network(t)
    loss_st = ((u[index, 0] - torch.from_numpy(S_raw[index]).to(device)) ** 2).sum()
    loss_it = ((u[index, 1] - torch.from_numpy(I_raw[index]).to(device)) ** 2).sum()
    loss_rt = ((u[index, 2] - torch.from_numpy(R_raw[index]).to(device)) ** 2).sum()
    return loss_st, loss_it, loss_rt

def plot_result_comparation(country, pre_data, data_type, real_data, ode_data, train_size):
    plt.figure(figsize=(16, 9))
    t = np.linspace(0, len(pre_data), len(pre_data) + 1)[:-1]
    plt.plot(t, real_data, color='black', label=f'{data_type}_Real')
    plt.scatter(t[:train_size], real_data[:train_size], color='black', marker='*', label=f'{data_type}_Train')
    plt.plot(t, pre_data, color='red', label=f'{data_type}_PINNs')
    plt.plot(t, ode_data, color='green', label=f'{data_type}_ODE')
    plt.xlabel('Time t (days)', fontsize=25)
    plt.ylabel('Numbers of individuals', fontsize=25)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.legend(fontsize=25)
    plt.savefig(path_results + f'{country}_{data_type}_results_comparation.pdf', dpi=600)
    plt.close()

def save_results_fixed(country, date, st, it, rt, S_raw, I_raw, R_raw, S_ode, I_ode, R_ode, train_size):
    pf_pinn = pd.DataFrame({
        'date': date, 'susceptibles': st, 'infectives': it, 'removed': rt,
        'S': S_raw, 'I': I_raw, 'R': R_raw, 'S_ode': S_ode, 'I_ode': I_ode, 'R_ode': R_ode
    })
    pf_pinn.to_csv(path_results + f"{country}_{train_size}_pinn_fixed_result.csv", index=False, sep=',')

def save_parameters_learned(country, beta_raw, gamma_raw, beta, gamma, train_size):
    parameters_table = PrettyTable(['', 'init', 'learned'])
    parameters_table.add_row(['beta', beta_raw, beta])
    parameters_table.add_row(['gamma', gamma_raw, gamma])
    result_file_name = path_results + f'{country}_{train_size}_parameters_learned_result.txt'
    if os.path.exists(result_file_name):
        os.remove(result_file_name)
    with open(result_file_name, 'a') as f:
        f.write(str(parameters_table) + '\n')

def save_error_result_sir(country, S, I, R, st, it, rt, S_ode, I_ode, R_ode, train_size):
    pinn_mse_s = metrics.mean_squared_error(S, st)
    pinn_mse_i = metrics.mean_squared_error(I, it)
    pinn_mse_r = metrics.mean_squared_error(R, rt)
    pinn_mae_s = metrics.mean_absolute_error(S, st)
    pinn_mae_i = metrics.mean_absolute_error(I, it)
    pinn_mae_r = metrics.mean_absolute_error(R, rt)
    pinn_mse_sir = pinn_mse_s + pinn_mse_i + pinn_mse_r
    pinn_mae_sir = pinn_mae_s + pinn_mae_i + pinn_mae_r
    error_table = PrettyTable(['', 'mse', 'mae'])
    error_table.add_row(['sir', pinn_mse_sir, pinn_mae_sir])
    error_table.add_row(['s', pinn_mse_s, pinn_mae_s])
    error_table.add_row(['i', pinn_mse_i, pinn_mae_i])
    error_table.add_row(['r', pinn_mse_r, pinn_mae_r])
    result_file_name = path_results + f'{country}_{train_size}_error_result.txt'
    if os.path.exists(result_file_name):
        os.remove(result_file_name)
    with open(result_file_name, 'a') as f:
        f.write(str(error_table) + '\n')

def plot_log_loss(country, loss, data_loss, residuals_loss):
    plt.figure(figsize=(16, 9))
    eps = np.linspace(0, len(loss), len(loss))
    plt.plot(eps, np.log10(loss), linewidth=1, label='loss')
    plt.plot(eps, np.log10(data_loss), linewidth=1, label='data_loss')
    plt.plot(eps, np.log10(residuals_loss), linewidth=1, label='residuals_loss')
    plt.title('train loss', fontsize=20)
    plt.xlabel('epoch', fontsize=20)
    plt.legend(loc=7, fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(path_results + f'{country}_log_loss.pdf', dpi=600)
    plt.close()

# Load data
country = 'Germany'
N = 84323763
days = 117

pf = pd.read_csv('../../code/Germany/4paper/remove_weight_from_residual_loss/84323763_0_117_0_53_real_data_estimated.csv') / N
date = np.array(pd.date_range(start='03/06/2021', periods=days, normalize=True).strftime('%Y-%m-%d'))

I_raw = np.array(pf['infectives']) * 25
R_raw = np.array(pf['removed']) * 15
S_raw = 1 - I_raw - R_raw

data_size = days
train_size = 50

# Initialize model
pinn_sir = Net_SIR().to(device)

# Training setup
t_points = np.linspace(0, data_size, data_size + 1)[:-1]
index = torch.randperm(train_size)
optimizer = optim.Adam(pinn_sir.parameters(), lr=1e-4, weight_decay=0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.998)
beta_trained = Variable(torch.tensor([0.25]).to(device), requires_grad=True)
gamma_trained = Variable(torch.tensor([0.15]).to(device), requires_grad=True)
optimizer_v = optim.Adam([beta_trained, gamma_trained], lr=1e-4, weight_decay=0.01)
scheduler_v = optim.lr_scheduler.StepLR(optimizer_v, step_size=5000, gamma=0.998)

S0, I0, R0 = S_raw[0], I_raw[0], R_raw[0]

total_epoch, total_data_loss_epoch, total_residual_loss_epoch = [], [], []
early_stopping = 500
loss_history = deque(maxlen=early_stopping + 1)

# Training loop
pinn_sir.zero_grad()
for epoch in range(20000):
    running_loss = 0.0
    optimizer.zero_grad()
    optimizer_v.zero_grad()

    loss_s, loss_i, loss_r, loss_n = residual_loss(t_points, beta_trained, gamma_trained)
    loss_st, loss_it, loss_rt = data_loss(t_points, index)
    loss_init = init_loss(t_points, S0, I0, R0)
    loss = (alpha_0) * (loss_st + 750 * loss_it + 300 * loss_rt) + (alpha_1) * (loss_s + loss_i + loss_r + loss_n) + alpha_2 * loss_init

    running_loss += loss.item()
    loss_history.append(running_loss)
    total_epoch.append(running_loss)
    total_data_loss_epoch.append((loss_st.item() + loss_it.item() + loss_rt.item()))
    total_residual_loss_epoch.append((loss_s + loss_i + loss_r + loss_n).item())

    if len(loss_history) > early_stopping and loss_history.popleft() < min(loss_history):
        print(f"Early stopping at [{epoch}] times. No train loss improvement in [{early_stopping}] epochs.")
        break

    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer_v.step()
    scheduler_v.step()

    if epoch % 1000 == 0:
        lr = optimizer.param_groups[0]['lr']
        print(f'epoch = {epoch}, loss: {running_loss}, lr: {lr}.')
        print(f'data loss: {(loss_st.item() + loss_it.item() + loss_rt.item())}, S loss: {loss_st.item()}, I loss: {loss_it.item()}, R loss: {loss_rt.item()}.')
        print(f'residual loss: {(loss_s + loss_i + loss_r + loss_n).item()}, loss_s: {loss_s.item()}, loss_i: {loss_i.item()}, loss_r: {loss_r.item()}, loss_n: {loss_n.item()}.')
        print(f'init loss: {loss_init}.')

print('Finished Training')

# Plot loss
plot_log_loss(country, total_epoch, total_data_loss_epoch, total_residual_loss_epoch)

# ODE integration
t = np.linspace(0, days, days + 1)[:-1]
u = network(t)
st, it, rt = u[:, 0].cpu().detach().numpy(), u[:, 1].cpu().detach().numpy(), u[:, 2].cpu().detach().numpy()
beta, gamma = beta_trained.cpu().detach().numpy()[0], gamma_trained.cpu().detach().numpy()[0]
u0 = [S0, I0, R0]
res = integrate.odeint(covid_sir, u0, t, args=(beta, gamma))
S_ode, I_ode, R_ode = res.T

# Save and plot results
save_results_fixed(country, date, st, it, rt, S_raw, I_raw, R_raw, S_ode, I_ode, R_ode, train_size)
plot_result_comparation(country, st, 'S', S_raw, S_ode, train_size)
plot_result_comparation(country, it, 'I', I_raw, I_ode, train_size)
plot_result_comparation(country, rt, 'R', R_raw, R_ode, train_size)
save_parameters_learned(country, 0.25, 0.15, beta, gamma, train_size)
save_error_result_sir(country, S_raw, I_raw, R_raw, st, it, rt, S_ode, I_ode, R_ode, train_size)

print(f'beta_raw: 0.25, gamma_raw: 0.15.')
print(f'beta and gamma are: {beta}, {gamma}.')

# plot the results for the paper

