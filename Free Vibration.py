# Solving free vibration using PINNs
# Considering: M = 1 kg , C = 1 N.s/m , K = 1 N/m

import numpy as np
import matplotlib.pyplot as plt
from smt.sampling_methods import LHS
import torch
import torch.nn as nn
import time as timelib

device = torch.device(0)
st = timelib.time()


def exact_sol(t):
    o = (
        0.5
        * torch.exp(-t / 2)
        * (
            torch.sin(torch.sqrt(torch.tensor(3)) * t / 2) / torch.sqrt(torch.tensor(3))
            + torch.cos(torch.sqrt(torch.tensor(3)) * t / 2)
        )
    )
    return o


# Define the domain intervals for each dimension
xlimits = np.array([[0.0, 10.0], [0.0, 0.0]])

# Create an LHS sampling instance
sampling = LHS(xlimits=xlimits)

# Generate 10000 samples
num_samples = 1000
samples = sampling(num_samples)

# Visualize the samples
plt.plot(samples[:, 0], samples[:, 1], "o")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

print(f"Generated {num_samples} samples with LHS.")
# t = np.linspace(0,1,100)

# x = np.ones_like(t) * 0.0
x = samples[:, 0]
x = torch.from_numpy(x)
x = x.float().to(device)
x.requires_grad = True

x_init = torch.tensor(0.0, dtype=torch.float32, requires_grad=True).to(device)
y_init = torch.tensor(0.5, dtype=torch.float32, requires_grad=True).to(device)
dy_init = torch.tensor(0.0, dtype=torch.float32, requires_grad=True).to(device)


class PINN(nn.Module):
    def __init__(self, layers) -> None:
        super(PINN, self).__init__()
        self

        self.x = x
        self.x_init = x_init
        self.y_init = y_init
        self.dy_init = dy_init

        self.loss_rec = []

        self.layers = layers
        self.net = nn.Sequential()
        for i in range(len(layers) - 2):
            self.net.add_module(f"layer_{i}", nn.Linear(layers[i], layers[i + 1]))
            self.net.add_module(f"activation_{i}", nn.Tanh())
            # self.net.add_module(f'dropout_{i}', nn.Dropout(p=self.dropout_prob))  # Add dropout
            # self.net.add_module(f'batchnorm_{i}', nn.BatchNorm1d(layers[i+1])) # Add batchnorm
        self.net.add_module("output", nn.Linear(layers[-2], layers[-1]))
        self.net.add_module(f"activation_output", nn.Tanh())

        self.adam = torch.optim.Adam(self.net.parameters(), lr=5e-3)
        self.lbfgs = torch.optim.LBFGS(
            self.net.parameters(),
            lr=1,
            max_iter=2000,
            max_eval=2000,
            tolerance_grad=0,
            tolerance_change=0,
            history_size=500,
            line_search_fn="strong_wolfe",
        )

    def forward(self, x):
        u = self.net(x.view(-1, 1))
        return u[:, 0]

    def ODE_loss(self):
        y = self.forward(self.x)
        dy = torch.autograd.grad(y.sum(), self.x, create_graph=True)[0]
        d2y = torch.autograd.grad(dy.sum(), self.x, create_graph=True)[0]
        return torch.mean(torch.square(d2y + dy + y))

    def boundary_loss(self):
        y = self.forward(self.x_init)
        dy = torch.autograd.grad(y.sum(), self.x_init, create_graph=True)[0]

        return torch.mean(torch.square(y - self.y_init)) + torch.mean(
            torch.square(dy - self.dy_init)
        )

    def closure(self):
        self.adam.zero_grad()
        self.lbfgs.zero_grad()

        loss = self.boundary_loss() + self.ODE_loss()
        self.loss_rec.append(loss.detach().cpu().item())

        print(
            f"\r epoch {len(self.loss_rec)} , loss : {loss.detach().cpu().item():5e} , time : {timelib.time()-st:.2f} s",
            end="",
        )
        if len(self.loss_rec) % 100 == 0:
            print("")

        loss.backward()

        return loss

    def train(self, optimizer, epoch):
        try:
            c = 0
            for i in optimizer:
                for j in range(epoch[c]):
                    if i == self.adam:
                        ls = self.closure()
                        i.step()
                    else:
                        i.step(self.closure)

                c += 1
        except KeyboardInterrupt:
            print("")
            print("intrrupted by user")

    def plot(self):
        with torch.no_grad():
            plt.figure(figsize=(5, 2.5))
            plt.semilogy(range(len(self.loss_rec)), self.loss_rec)
            plt.ylim([0, max(self.loss_rec)])
            plt.xlim([0, len(self.loss_rec)])
            plt.ylabel("Loss")
            plt.xlabel("Epoch")


ode_solver = PINN([1, 8, 8, 8, 1]).to(device)
optimizer = [ode_solver.adam, ode_solver.lbfgs]
epoch = [100, 200]


ode_solver.train(optimizer, epoch)

ode_solver.plot()

l = torch.linspace(0, 10, 1000).to(device)
y = ode_solver(l)
y_ex = exact_sol(l)
plt.plot(l.detach().cpu().numpy(), y.detach().cpu().numpy())
plt.plot(l.detach().cpu().numpy(), y_ex.detach().cpu().numpy())
