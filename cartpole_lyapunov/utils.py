import torch
import numpy as np
import matplotlib.pyplot as plt
from integration import _flow, _flow_rk4
import params
from losses import gamma_forwards, gamma_backwards
from tqdm import tqdm
from torch.func import grad
import pickle



# save object as '.pkl'
def pickle_object(name, thing):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(thing, f)


# load object from '.pkl'
def unpickle_object(name):
    with open(name + '.pkl', 'rb') as f:
        thing = pickle.load(f)
    return thing


def rollout_parallel_rk4(ae, fdyn, lqr, X0, T=1000, save=False):
    # assume X0 is shape (N, d_x)

    # get first control input:
    X = torch.tensor(X0)
    X0_torch = X
    if save:
        X_traj = [X]

    for t in tqdm(range(T)):
        Z = ae.encode(X)
        U = lqr(Z).to(device='cuda')
        X = _flow_rk4(X, params.DT, U)
        if save:
            X_traj.append(X)

    if save:
        X_traj = torch.stack(X_traj, dim=1)
        return X_traj.cpu().detach().numpy()
    else:
        X_traj = torch.stack([X0_torch, X], dim=1)
        return X_traj.cpu().detach().numpy()


# rolls out trajectories of the true dynamics under the closed-loop latent controller,
# computes gamma-forwards, gamma_backwards, true residual, and Lipschitz constant
def rollout_trajectories(ae, fdyn, lqr, X0, n_traj=100, T=200, plot=True, V_filter=None, inner=None, V=None, a0=None, n_per_axis=None,
                         mstep_gammas=None, closed_loop_conjugacy=False):
    X = []
    Z = []
    U = [] 
    mstep_max_gamma_forwards = []
    max_gamma_forwards = []
    max_gamma_backwards = []
    L = []
    R = [] # residual
    for x0 in tqdm(X0):
        x = x0
        z = ae.encode(torch.tensor(x0).float())
        Xi = [x]
        Zi = [z]
        Ui = []
        for t in range(T):
            u = lqr(z).item()
            if closed_loop_conjugacy:
                RHS = (fdyn[0](z).reshape(params.d_z, params.d_z) @ z.reshape(params.d_z, 1) +\
                       fdyn[1](z).reshape(params.d_z, params.d_u) @ torch.tensor(u).reshape(params.d_u, 1)).squeeze()
                u_opt = closed_loop_conjugacy_control(x, u, RHS)

            Ui.append(u)

            if closed_loop_conjugacy:
                x = _flow(x, params.DT, u_opt)[-1]
            else:
                x = _flow(x, params.DT, u)[-1]

            if V is not None:
                if closed_loop_conjugacy:
                    u_fE = u_opt
                else:
                    u_fE = u
                fE = (fdyn[0](z).reshape(params.d_z, params.d_z) @ z.reshape(params.d_z, 1) +\
                      fdyn[1](z).reshape(params.d_z, params.d_u) @ torch.tensor(u_fE).reshape(params.d_u, 1)).squeeze()
                Ef = ae.encode(torch.tensor(x).float())
                R.append(torch.abs(V(fE) - V(Ef)).cpu().item())
            z = ae.encode(torch.tensor(x).float())
            if V is not None:
                L.append(torch.linalg.norm(grad(V)(z)).cpu().item())
            Xi.append(x)
            Zi.append(z)
        Xi = np.array(Xi)
        #gamma_fwd = np.max(gamma_forwards(Xi, Zi[:-1], Ui, ae, fdyn))
        if mstep_gammas is not None:
            if closed_loop_conjugacy:
                gammas_fwd, mstep_gammas_fwd = gamma_forwards(Xi, Zi[:-1], Ui, ae, fdyn, mstep_gammas=mstep_gammas)
            else:
                gammas_fwd, mstep_gammas_fwd = gamma_forwards(Xi, Zi[:-1], Ui, ae, fdyn, mstep_gammas=mstep_gammas)

            mstep_max_gamma_forwards.append(mstep_gammas_fwd)
        else:
            gammas_fwd = gamma_forwards(Xi, Zi[:-1], Ui, ae, fdyn)
        gamma_fwd = np.max(gammas_fwd)
        gamma_bwd = np.max(gamma_backwards(Xi, Zi[:-1], Ui, ae, fdyn))
        max_gamma_forwards.append(gamma_fwd)
        max_gamma_backwards.append(gamma_bwd)
        X.append(Xi) 
        Z.append(Zi)            
        U.append(Ui)
    X = np.array(X)
    Z = np.array([torch.vstack(Zi).cpu().detach().numpy() for Zi in Z])
    U = np.array(U)
    if plot:
        #print("plotting latent trajectories")
        #print("total number of trajectories:", Z.shape)
        for Zi in Z: 
            plt.plot(Zi[:,0], Zi[:,1])

        if a0 is not None:
            Zflat = Z.reshape(-1, params.d_z)
            rxh = np.max(Zflat[:,0], axis=0)  
            rxl = np.min(Zflat[:,0], axis=0)
            ryh = np.max(Zflat[:,1], axis=0)
            ryl = np.min(Zflat[:,1], axis=0)
            eps = max([abs(rxh), abs(rxl), abs(ryh), abs(ryl)]) / 3
            rxh += eps; rxl -= eps; ryh += eps; ryl -= eps
            X_pts = torch.linspace(rxl, rxh, n_per_axis)
            Y_pts = torch.linspace(ryl, ryh, n_per_axis)
            XX, YY = torch.meshgrid(X_pts, Y_pts)
            VV = V(torch.dstack([XX, YY]).reshape(-1, params.d_z)).reshape(XX.shape)
            plt.contour(XX.cpu().detach().numpy(), YY.cpu().detach().numpy(), VV.cpu().detach().numpy(), [a0], colors=['r'])

        plt.show()

    print("max x:", np.max(np.abs(X[:,:,0])))
    print("max v:", np.max(np.abs(X[:,:,1])))
    print("max theta:", np.max(np.abs(X[:,:,2])))
    print("max w:", np.max(np.abs(X[:,:,3])))

    if V is not None:
        if mstep_gammas is not None:
            mstep_max_gamma_forwards = np.array(mstep_max_gamma_forwards)
            return X, Z, U, (max(max_gamma_forwards), max(max_gamma_backwards)), max(L), max(R), np.max(mstep_max_gamma_forwards, axis=0)
        else:
            return X, Z, U, (max(max_gamma_forwards), max(max_gamma_backwards)), max(L), max(R)

    return X, Z, U, (max(max_gamma_forwards), max(max_gamma_backwards))
