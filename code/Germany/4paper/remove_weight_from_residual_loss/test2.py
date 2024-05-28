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

# Ensure reproducibility
torch.manual_seed(0)

# Constants
COUNTRY = 'Germany'
POPULATION = 84323763
DAYS = 117
TRAIN_SIZE = 30
EARLY_STOPPING = 250
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
STEP_SIZE = 5000
GAMMA = 0.998
EPOCHS = 200000

# Load data
data_path = '../../4paper/remove_weight_from_residual_loss/84323763_0_117_0_53_real_data_estimated.csv'
data = pd.read_csv(data_path)

# Plot susceptibles
plt.plot(data['susceptibles'])
plt.show()

# Normalize data
pf = pd.read_csv(data_path) / POPULATION
date = pd.date_range(start='03/06/2021', periods=DAYS, normalize=True).strftime('%Y-%m-%d')

I_raw = np.array(pf['infectives']) * 25
R_raw = np.array(pf['removed']) * 15
N = POPULATION / POPULATION
S_raw = N - I_raw - R_raw

# Plot infectives
plt.plot(pf['infectives'])
plt.show()

# Training and test sets
t_points = np.linspace(0, DAYS, DAYS + 1)[:-1]
index = torch.randperm(TRAIN_SIZE)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def covid_sir(u, t, beta, gamma):
    S, I, R = u
    dS_dt = -beta * S * I
    dI_dt = beta * S * I - gamma * I
    dR_dt = gamma * I
    return np.array([dS_dt, dI_dt, dR_dt])

# Neural Network for SIR
class NetSIR(nn.Module):
    def __init__(self):
        super(NetSIR, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 3),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.001)

# Initialize and move the network to the device
pinn_sir = NetSIR().to(device)
pinn_sir.apply(init_weights)

def network(t):
    input = torch.from_numpy(t.reshape(-1, 1)).float().to(device)
    return pinn_sir(input) * POPULATION

def residual_loss(t, beta, gamma):
    dt = 1
    u = network(t)
    st2, st1 = u[1:, 0], u[:-1, 0]
    it2, it1 = u[1:, 1], u[:-1, 1]
    rt2, rt1 = u[1:, 2], u[:-1, 2]

    ds = (st2 - st1) / dt
    di = (it2 - it1) / dt
    dr = (rt2 - rt1) / dt

    loss_s = (-(beta) * st1 * it1 - ds) ** 2
    loss_i = ((beta) * st1 * it1 - (gamma) * it1 - di) ** 2
    loss_r = ((gamma) * it1 - dr) ** 2
    loss_n = (u[:, 0] + u[:, 1] + u[:, 2] - POPULATION) ** 2  # normalization constraint
    return loss_s.sum(), loss_i.sum(), loss_r.sum(), loss_n.sum()

def init_loss(t, S0, I0, R0):
    u = network(t)
    loss_s0 = (u[0, 0] - S0) ** 2
    loss_i0 = (u[0, 1] - I0) ** 2
    loss_r0 = (u[0, 2] - R0) ** 2
    return loss_s0 + loss_i0 + loss_r0

def data_loss(t, index):
    u = network(t)
    loss_st = (u[index, 0] - torch.from_numpy(S_raw[index]).to(device)) ** 2
    loss_it = (u[index, 1] - torch.from_numpy(I_raw[index]).to(device)) ** 2
    loss_rt = (u[index, 2] - torch.from_numpy(R_raw[index]).to(device)) ** 2
    return loss_st.sum(), loss_it.sum(), loss_rt.sum()

# Set up training parameters
beta_trained = Variable(torch.tensor([0.25]).to(device), requires_grad=True)
gamma_trained = Variable(torch.tensor([0.15]).to(device), requires_grad=True)

nn_parameters = list(pinn_sir.parameters())
optimizer = optim.Adam(nn_parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

optimizer_v = optim.Adam([beta_trained, gamma_trained], lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler_v = optim.lr_scheduler.StepLR(optimizer_v, step_size=STEP_SIZE, gamma=GAMMA)

S0, I0, R0 = S_raw[0], I_raw[0], R_raw[0]

# Training loop
loss_history = deque(maxlen=EARLY_STOPPING + 1)
total_epoch, total_data_loss_epoch, total_residual_loss_epoch = [], [], []

alpha_0, alpha_1, alpha_2 = 150, 60000, 50

pinn_sir.zero_grad()

for epoch in range(EPOCHS):
    optimizer.zero_grad()
    optimizer_v.zero_grad()

    loss_s, loss_i, loss_r, loss_n = residual_loss(t_points, beta_trained, gamma_trained)
    loss_st, loss_it, loss_rt = data_loss(t_points, index)
    loss_init = init_loss(t_points, S0, I0, R0)

    loss = (alpha_0) * (loss_st + 750 * loss_it + 300 * loss_rt) + (alpha_1) * (loss_s + loss_i + loss_r + loss_n) + alpha_2 * loss_init

    running_loss = loss.item()
    loss_history.append(running_loss)
    total_epoch.append(running_loss)
    total_data_loss_epoch.append((loss_st.item() + loss_it.item() + loss_rt.item()))
    total_residual_loss_epoch.append((loss_s + loss_i + loss_r + loss_n).item())

    if len(loss_history) > EARLY_STOPPING and loss_history.popleft() < min(loss_history):
        print(f"Early stopping at [{epoch}] times. No train loss improvement in [{EARLY_STOPPING}] epochs.")
        break

    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer_v.step()
    scheduler_v.step()

    if epoch % 1000 == 0:
        lr = optimizer.param_groups[0]['lr']
        print(f'Epoch: {epoch}, Loss: {running_loss}, LR: {lr}')
        print(f'Data Loss: {loss_st.item() + loss_it.item() + loss_rt.item()}, S Loss: {loss_st.item()}, I Loss: {loss_it.item()}, R Loss: {loss_rt.item()}')
        print(f'Residual Loss: {(loss_s + loss_i + loss_r + loss_n).item()}, S: {loss_s.item()}, I: {loss_i.item()}, R: {loss_r.item()}, N: {loss_n.item()}')
        print(f'Init Loss: {loss_init.item()}')

print('Training Finished')

# Plot loss curves
def plot_loss_curve(country, total_epoch, total_data_loss_epoch, total_residual_loss_epoch, train_size):
    plt.figure(figsize=(10, 5))
    plt.plot(total_epoch, label='Total Loss')
    plt.plot(total_data_loss_epoch, label='Data Loss')
    plt.plot(total_residual_loss_epoch, label='Residual Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss Curve for {country}')
    plt.legend()
    plt.show()

plot_loss_curve(COUNTRY, total_epoch, total_data_loss_epoch, total_residual_loss_epoch, TRAIN_SIZE)

# Prediction
t = np.linspace(0, DAYS, DAYS + 1)[:-1]
u = network(t)
st, it, rt = u[:, 0].cpu().detach().numpy(), u[:, 1].cpu().detach().numpy(), u[:, 2].cpu().detach().numpy()

# Plot results
def plot_results(predictions, true_values, label):
    plt.plot(predictions, label=f'Predicted {label}', color='red')
    plt.plot(true_values, label=f'True {label}', color='black')
    plt.legend()
    plt.show()

plot_results(st, S_raw, 'Susceptibles')
plot_results(it, I_raw, 'Infectives')
plot_results(rt, R_raw, 'Recovered')

# Calculate ODE solution with trained parameters
def solve_ode(u0, t, beta, gamma):
    return integrate.odeint(covid_sir, u0, t, args=(beta, gamma)).T

beta, gamma = beta_trained.item(), gamma_trained.item()
S_ode, I_ode, R_ode = solve_ode([S0, I0, R0], t, beta, gamma)

# Plot comparison
def plot_result_comparison(country, predicted, real, ode, train_size, label):
    plt.figure(figsize=(16, 9))
    t = np.linspace(0, len(predicted), len(predicted) + 1)[:-1]
    plt.plot(t, real, color='black', label=f'{label}_Real')
    plt.scatter(t[:train_size], real[:train_size], color='black', marker='*', label=f'{label}_Train')
    plt.plot(t, predicted, color='red', label=f'{label}_Predicted')
    if label != 'Infectives':
        plt.plot(t, ode, color='green', label=f'{label}_ODE')
    plt.xlabel('Time (days)')
    plt.ylabel(f'Number of {label}')
    plt.legend()
    plt.show()

plot_result_comparison(COUNTRY, st, S_raw, S_ode, TRAIN_SIZE, 'Susceptibles')
plot_result_comparison(COUNTRY, it, I_raw, I_ode, TRAIN_SIZE, 'Infectives')
plot_result_comparison(COUNTRY, rt, R_raw, R_ode, TRAIN_SIZE, 'Recovered')

# Save results
def save_results(country, dates, st, it, rt, S_raw, I_raw, R_raw, S_ode, I_ode, R_ode, train_size):
    results_dir = f'results/{country}'
    os.makedirs(results_dir, exist_ok=True)
    results = pd.DataFrame({
        'date': dates,
        'S_pred': st,
        'I_pred': it,
        'R_pred': rt,
        'S_true': S_raw,
        'I_true': I_raw,
        'R_true': R_raw,
        'S_ode': S_ode,
        'I_ode': I_ode,
        'R_ode': R_ode
    })
    results.to_csv(os.path.join(results_dir, f'{country}_results.csv'), index=False)

save_results(COUNTRY, date, st, it, rt, S_raw, I_raw, R_raw, S_ode, I_ode, R_ode, TRAIN_SIZE)

# Save MSE and MAE
def save_errors(country, S_raw, I_raw, R_raw, st, it, rt, S_ode, I_ode, R_ode, train_size):
    mse_s = np.mean((S_raw - st) ** 2)
    mse_i = np.mean((I_raw - it) ** 2)
    mse_r = np.mean((R_raw - rt) ** 2)
    mae_s = np.mean(np.abs(S_raw - st))
    mae_i = np.mean(np.abs(I_raw - it))
    mae_r = np.mean(np.abs(R_raw - rt))
    
    errors_dir = f'errors/{country}'
    os.makedirs(errors_dir, exist_ok=True)
    errors = {
        'MSE_S': mse_s,
        'MSE_I': mse_i,
        'MSE_R': mse_r,
        'MAE_S': mae_s,
        'MAE_I': mae_i,
        'MAE_R': mae_r
    }
    with open(os.path.join(errors_dir, f'{country}_errors.json'), 'w') as f:
        json.dump(errors, f, indent=4)

save_errors(COUNTRY, S_raw, I_raw, R_raw, st, it, rt, S_ode, I_ode, R_ode, TRAIN_SIZE)

# Save parameters
def save_parameters(country, beta_raw, gamma_raw, beta, gamma, train_size):
    params_dir = f'parameters/{country}'
    os.makedirs(params_dir, exist_ok=True)
    params = {
        'beta_raw': beta_raw,
        'gamma_raw': gamma_raw,
        'beta_trained': beta,
        'gamma_trained': gamma
    }
    with open(os.path.join(params_dir, f'{country}_parameters.json'), 'w') as f:
        json.dump(params, f, indent=4)

save_parameters(COUNTRY, 0.25, 0.15, beta, gamma, TRAIN_SIZE)

# Print beta and gamma
print(f'Beta: {beta}, Gamma: {gamma}')
