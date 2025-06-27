import numpy as np
import cvxpy as cp
import torch
import torch.nn as nn
import params
import matplotlib
import matplotlib.pyplot as plt
from cartpole import _dxdt_torch
from matplotlib import ticker
from torch.func import jacrev, grad, vmap
from collections import OrderedDict
from lyapunov import LyapunovEllipse, LyapunovGeneral



# learn special-form (ellipse, general) Lypaunov function over latent space trajectories 
def mlp_lyapunov_reparam(Z, epochs=500, lr=1e-4, plot=True, grid_dens=500, rho=0.5, z_eq=None, parameterization="general", features=128):
    # 1) train lyapunov 
    # 2) plot levelsets using matplotlib 
    # 3) compute lipschitz constant by gridding

    use_pd_loss = False
    use_dyn_loss = True
    use_lb_loss = False
    use_grad_loss = False
    print("pd loss:", use_pd_loss)
    print("dyn loss:", use_dyn_loss)
    print("lb loss", use_lb_loss)
    print("grad loss", use_grad_loss)

    print("rho:", rho)

    c = 0.1
    print("c lb", c)

    l_dyn = 1 
    l_pd = 1
    l_lb = 1
    l_grad = 1e-6
    print("l_dyn", l_dyn)
    print("l_pd", l_pd)
    print("l_lb", l_lb)
    print("l_grad", l_grad)

    Z = torch.tensor(Z).float().to("cuda")

    Z_blob = Z.reshape(-1, params.d_z)
    Z1_max = torch.max(Z_blob[:,0]); Z1_max += 0.1*Z1_max
    Z1_min = torch.min(Z_blob[:,0]); Z1_min -= 0.1*torch.abs(Z1_min)
    Z2_max = torch.max(Z_blob[:,1]); Z2_max += 0.1*Z2_max
    Z2_min = torch.min(Z_blob[:,1]); Z2_min -= 0.1*torch.abs(Z2_min)
    Z1_pts = torch.linspace(Z1_min, Z1_max, grid_dens)
    Z2_pts = torch.linspace(Z2_min, Z2_max, grid_dens)
    ZZ, WW = torch.meshgrid(Z1_pts, Z2_pts)

    print("using grid data, grid_density:", grid_dens)

    grid_data = torch.dstack([ZZ, WW]).reshape(-1, params.d_z)

    #features = 128 # suggested: 128
    print("features:", features)

    n_layers = 3 # suggested: 3
    if parameterization == "general":
        V = LyapunovGeneral(n_layers, features, z_eq=z_eq)
    elif parameterization == "ellipse":
        V = LyapunovEllipse(n_layers, features, z_eq=z_eq)
    print(V)

    V_opt = torch.optim.AdamW(params=V.parameters(), lr=lr)

    
    def grad_loss(V, Z):
        Z = Z.reshape(-1, params.d_z)
        grads = vmap(jacrev(V))(Z)
        return torch.sum(torch.norm(grads, dim=1)**2)


    def dyn_loss(V, Z):
        Zi = Z[:,:-1].reshape(-1, params.d_z)
        Zf = Z[:,1:].reshape(-1, params.d_z)

        Vi = V(Zi)
        Vf = V(Zf)

        return torch.sum(nn.functional.relu((Vf-rho*Vi))**2)


    def lb_loss(V, Z):
        return torch.sum(nn.functional.relu(c*torch.linalg.norm(Z, dim=1) - V(Z))**2)


    V.train()
    for i in range(epochs):
        V_opt.zero_grad()

        loss = torch.tensor([0.])
        if use_dyn_loss:
            loss += l_dyn*dyn_loss(V, Z) 
        if use_lb_loss:
            loss += l_lb * lb_loss(V, grid_data) 
        if use_grad_loss:
            loss += l_grad * grad_loss(V, Z)

        loss.backward()
        V_opt.step()

        if i % 1000 == 0:
            print("it {}:".format(i), loss)

    if plot:
        fig, ax = plt.subplots(1)
        fig.set_size_inches(10, 10)

        alpha = torch.max(V(Z_blob)).item()
        print("alpha", alpha)

        grads = vmap(grad(V))(Z_blob)
        print(grads.shape)

        lip = torch.max(torch.linalg.norm(grads.squeeze(), dim=-1))
        print("Lipschitz constant", lip)

        VV = V(torch.dstack([ZZ, WW]).reshape(-1, params.d_z)).reshape(grid_dens, grid_dens)
        ax.set_title("Rho: {0:.2f}, Local Lipschitz Over Data: {1:.2f}".format(rho, lip.item()))
        lvls = np.concatenate([np.linspace(0, alpha, 30), np.linspace(alpha+0.1*alpha, 10*alpha, 5)])
        cf = ax.contourf(ZZ.cpu().detach().numpy(), WW.cpu().detach().numpy(), VV.cpu().detach().numpy(), levels=50)
        #ax.contour(ZZ.cpu().detach().numpy(), WW.cpu().detach().numpy(), VV.cpu().detach().numpy(), levels=lvls)#, colors=['k'])
        fig.colorbar(cf)
        cntr = plt.contour(ZZ.cpu().detach().numpy(), WW.cpu().detach().numpy(), VV.cpu().detach().numpy(), [alpha], colors=['w'])
        ax.clabel(cntr, inline=True, colors=['w'], fontsize=14)
        for Zi in Z:
            ax.plot(Zi[:,0].cpu().detach().numpy(), Zi[:,1].cpu().detach().numpy(), alpha=0.25)
        plt.show()

    return V, rho, alpha, lip


# compute Jacobians and cost matrices for LQR
def get_LQR_params(ae, fdyn, ret_AB=False, original_system=False, u_cost=1):
    if not original_system:
        ae.eval()
        if params.control_affine or params.linear_state_space:
            fdyn_drift, fdyn_cntrl = fdyn
            if params.linear_state_space_offset:
                for f in fdyn_drift:
                    f.eval()
                fdyn_drift, fdyn_offset = fdyn_drift
            else:
                fdyn_drift.eval()
                fdyn_cntrl.eval()
        elif not (params.enc_true_dyn or params.reverse_ae):
            fdyn.eval()

    with torch.no_grad():
        if not original_system:
            x_0 = torch.unsqueeze(torch.tensor([0., 0., 0., 0.]), 0)
            z_0 = ae.encode(x_0)
            z_0 = torch.tensor(z_0.cpu().detach().numpy())
            u_0 = torch.tensor([[0.]])
        else:
            z_0 = torch.tensor([0.,0.,0.,0.,]).unsqueeze(0)

        if original_system:
            f = _dxdt_torch
            A = jacrev(f, argnums=0)(torch.tensor([0.,0.,0.,.0]), torch.tensor(0.))
            B = jacrev(f, argnums=1)(torch.tensor([0.,0.,0.,.0]), torch.tensor(0.)).unsqueeze(-1)
            A = torch.eye(params.d_x) + A
        elif params.control_affine:
            f_wrap = lambda z, u: fdyn(torch.hstack((z, u)))
            grad_drift = torch.autograd.functional.jacobian(fdyn_drift, z_0)
            grad_cntrl = fdyn_cntrl(z_0).T
            A = grad_drift.squeeze()
            B = grad_cntrl
        elif params.linear_state_space:
            A = fdyn_drift(z_0).reshape(params.d_z, params.d_z)
            B = fdyn_cntrl(z_0).reshape(params.d_z, params.d_u)
        else:
            jac_xu = jacrev(fdyn)( torch.cat((z_0[0], u_0[0]), dim=0) )
            A = jac_xu[:,:-1]
            B = jac_xu[:,-1].unsqueeze(-1)

        if params.learn_residual and not original_system:
            A = torch.eye(params.d_z) + A

        if original_system:
            Q = 1*torch.eye(params.d_x)
            R = u_cost*torch.eye(params.d_u) #1e-3, # PREV USING THIS: 1
            N = 0*torch.ones((params.d_x, params.d_u))
        else:
            Q = 1*torch.eye(params.d_z)
            R = u_cost*torch.eye(params.d_u) #1e-3, # PREV USING THIS: 1
            N = 0*torch.ones((params.d_z, params.d_u))
        
        if not ret_AB:
            print("A shape", A.shape)
            print("B shape", B.shape)
            print("Q shape", Q.shape)
            print("R shape", R.shape)
            print("A\n", A)
            print("B\n", B)

        P_f = Q
        P_t = [P_f]
        for i in range(200): #1e4
            P_next = A.T@P_t[-1]@A - (A.T@P_t[-1]@B + N) @ torch.linalg.inv(R + B.T@P_t[-1]@B) @ (B.T@P_t[-1]@A + N.T) + Q
            P_t.append(P_next)

        P = P_t[-1]
        F = torch.linalg.inv(R + B.T@P@B) @ (B.T@P@A + N.T)

    if ret_AB:
        return F, z_0, A, B, Q, R

    P = A - B @ F
    e = torch.linalg.eigvals(P)
    print("eigenvalues:\n", e)
    return F, z_0, Q, R



# class to initialize LQR controller for specified latent dynamics
class LQR(nn.Module):
    def __init__(self, ae, fdyn, original_system=False, u_cost=1):
        super().__init__()
        self.ae = ae
        self.fdyn = fdyn
        self.F, self.z_0, self.A, self.B, self.Q, self.R = get_LQR_params(ae, fdyn, ret_AB=True, original_system=original_system, u_cost=u_cost)
        self.F = self.F.cpu().detach()
        self.z_0.requires_grad = False
        self.u_cost = u_cost


    def forward(self, z):
        return -self.F.cpu() @ (z.cpu() - self.z_0.cpu()).T



