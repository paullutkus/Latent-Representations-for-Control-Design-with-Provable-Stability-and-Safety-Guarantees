import torch
from torch import nn
import params


class LyapunovGeneral(nn.Module):
    def __init__(self, n_layers, features, z_eq):
        super().__init__()
        self.z_eq = torch.tensor(z_eq.cpu().detach().numpy()).float() #z_eq.detach()
        self.z_eq.requires_grad = False
        layers = nn.ModuleList([])
        layers.append(nn.Linear(2, features, bias=False))
        layers.append(nn.Tanh())
        for i in range(n_layers - 2):
            layers.append(nn.Linear(features, features, bias=False))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(features, 1, bias=False))
        self.layers = layers
        self.lam = 0.1 # 0.1
        print("coeff:", self.lam)

    def forward(self, z):
        z0 = z - self.z_eq.unsqueeze(0)
        z = z0
        #z = z - self.z_eq.unsqueeze(0)
        for layer in self.layers:
            z = layer(z)
        #print("z-squared shape", (z**2).shape)
        #print("norm z shape", torch.linalg.norm(z, dim=1, keepdims=True).shape)
        v = z**2 + self.lam*torch.linalg.norm(z0, dim=1, keepdims=True)**2 #0.1
        #print("result shape", v.shape)
        return v.squeeze()


class LyapunovEllipse(nn.Module):
    def __init__(self, n_layers, features, z_eq):
        super().__init__()
        self.z_eq = z_eq
        self.z_eq = self.z_eq.detach()
        layers = nn.ModuleList([])
        layers.append(nn.Linear(2, features))
        layers.append(nn.Tanh())
        for i in range(n_layers-2):
                layers.append(nn.Linear(features, features))
                layers.append(nn.Tanh())
        layers.append(nn.Linear(features, params.d_z*params.d_z))
        self.layers = layers  

    def forward(self, z):
        if len(z.shape) == 2:
            z_bar = z - self.z_eq.unsqueeze(0)
        if len(z.shape) == 1:
            z_bar = z - self.z_eq
        o = z
        for layer in self.layers:
            o = layer(o)
        if len(z.shape) == 2:
            L = o.reshape(-1, params.d_z, params.d_z)
            P = L @ torch.transpose(L, 1, 2) + torch.eye(params.d_z).unsqueeze(0).expand(z.shape[0], -1, -1)
            #print("lhs shape", z.unsqueeze(-1).shape)
            Vz_lhs = torch.bmm(P, z_bar.unsqueeze(-1))
            Vz = torch.bmm(z_bar.unsqueeze(1), Vz_lhs).squeeze()
            return Vz
        else:
            L = o.reshape(params.d_z, params.d_z)
            P = L @ L.T + torch.eye(params.d_z)
            Vz_lhs = P @ z_bar.unsqueeze(-1)
            Vz = (z_bar.unsqueeze(0) @ Vz_lhs).squeeze()
            return Vz

