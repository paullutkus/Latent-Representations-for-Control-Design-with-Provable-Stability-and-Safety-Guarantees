import torch
import numpy as np
import matplotlib.pyplot as plt
from integration import _flow
import params
from losses import gamma_forwards, gamma_backwards
from tqdm import tqdm
from torch.func import grad
import pickle



def pickle_object(name, thing):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(thing, f)


def unpickle_object(name):
    with open(name + '.pkl', 'rb') as f:
        thing = pickle.load(f)
    return thing


def rollout_trajectories(ae, fdyn, lqr, X0, n_traj=100, T=200, plot=True, V_filter=None, inner=None, V=None, a0=None, n_per_axis=None):
    #X0 = r_x0*(np.random.rand(n_traj, params.d_x) - 0.5)
    X = []
    Z = []
    U = [] 
    max_gamma_forwards = []
    max_gamma_backwards = []
    L = []
    R = [] # residual
    for x0 in tqdm(X0):
        x = x0
        z = ae.encode(torch.tensor(x0).float())
        '''
        if V_filter is not None and inner is not None:
            #print(V_filter(z) <= inner)
            keep = (V_filter(z) <= inner)
        else:
            keep = (torch.linalg.norm(z) <= r_z0)
        if keep:
        '''
        Xi = [x]
        Zi = [z]
        Ui = []
        for t in range(T):
            u = lqr(z).item()
            Ui.append(u)
            x = _flow(x, params.DT, u)[-1]
            if V is not None:
                fE = (fdyn[0](z).reshape(params.d_z, params.d_z) @ z.reshape(params.d_z, 1) +\
                      fdyn[1](z).reshape(params.d_z, params.d_u) @ torch.tensor(u).reshape(params.d_u, 1)).squeeze()
                Ef = ae.encode(torch.tensor(x).float())
                R.append(torch.abs(V(fE) - V(Ef)).cpu().item())
            z = ae.encode(torch.tensor(x).float())
            if V is not None:
                L.append(torch.linalg.norm(grad(V)(z)).cpu().item())
            Xi.append(x)
            Zi.append(z)
        Xi = np.array(Xi)
        #Zi = np.array(Zi)
        #print(Xi.shape)
        #print(len(Zi))
        #print(len(Ui))
        gamma_fwd = np.max(gamma_forwards(Xi, Zi[:-1], Ui, ae, fdyn))
        gamma_bwd = np.max(gamma_backwards(Xi, Zi[:-1], Ui, ae, fdyn))
        #print(gamma)
        max_gamma_forwards.append(gamma_fwd)
        max_gamma_backwards.append(gamma_bwd)
        X.append(Xi) 
        Z.append(Zi)            
        U.append(Ui)
    X = np.array(X)
    Z = np.array([torch.vstack(Zi).cpu().detach().numpy() for Zi in Z])
    U = np.array(U)
    if plot:
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
        return X, Z, U, (max(max_gamma_forwards), max(max_gamma_backwards)), max(L), max(R)
    return X, Z, U, (max(max_gamma_forwards), max(max_gamma_backwards))
