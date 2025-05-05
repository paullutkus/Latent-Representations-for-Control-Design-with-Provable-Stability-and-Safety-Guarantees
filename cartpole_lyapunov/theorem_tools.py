import numpy as np
import torch
import matplotlib.pyplot as plt
import params 
from scipy.spatial import ConvexHull
from matplotlib.path import Path
from controls import LQR
from utils import rollout_trajectories



# Z (ndarray): preimage argument
# r (float): grid each axis over [-r, r]
# n_per_axis (int): grid points per axis
def compute_preimage(ae, Z, r, n_per_axis, uniform_sampling=False):
    if uniform_sampling: 
        pts = 2*r*(torch.rand((n_per_axis**4, params.d_x)) - 0.5)
    else:
        axes = []
        for i in range(params.d_x):
            axes.append(torch.linspace(-r, r, n_per_axis))
        pts = torch.dstack(torch.meshgrid(axes)).reshape(-1, params.d_x)

    #print(pts)
    EX = ae.encode(pts).cpu().detach().numpy()
    #print("after encode")
    
    # Compute convex hull of D_z, and define path 
    # object to check inclusion
    hull = ConvexHull(Z.reshape(-1, params.d_z))
    hull_path = Path(Z.reshape(-1, params.d_z)[hull.vertices])

    preimage = []
    for x, z in zip(pts.cpu().detach().numpy(), EX):
        if hull_path.contains_point((z[0], z[1])):
            #print("contained")
            preimage.append(x)
    preimage = np.array(preimage)
    return preimage


def plot_lyapunov_lvlsets(V, ae, fdyn, X, Z, a0, n_per_axis=100):
    lqr = LQR(ae, fdyn)
    _, Z, _, (gamma, _), L = rollout_trajectories(ae, fdyn, lqr, X, n_traj=200, T=200, plot=True, V=V)
    print(Z.shape)

    print("gamma fwd:", gamma)
    print("L:", L)
    print("a0:", a0)
    print("Ly/p:", L*gamma*10)

    Zflat = Z.reshape(-1, params.d_z)
    fig, ax = plt.subplots(1)
    fig.set_size_inches(10, 10)
    #alpha = torch.max(V(Z_blob)).item()

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

    cf = ax.contourf(XX.cpu().detach().numpy(),YY.cpu().detach().numpy(), VV.cpu().detach().numpy(), levels=50)

    for Zi in Z:
        ax.plot(Zi[:,0], Zi[:,1])
    
    cntr = ax.contour(XX.cpu().detach().numpy(), YY.cpu().detach().numpy(), VV.cpu().detach().numpy(), [a0, L*gamma*10], colors=['w', 'r'])
    proxy = [plt.Rectangle((0,0),1,1,fc=fc) for fc in cntr.get_edgecolors()]
    ax.legend(proxy, ["test", "Ly/p"])

    plt.show()

    '''
    print("alpha", alpha)
    #print(jacrev(V)(Z_blob).shape)
    #grads = vmap(jacrev(V),in_dims=0)(Z_blob)
    #grads = vmap(jacrev(V))(Z_blob)

    #print(V(Z_blob).shape)
    grads = vmap(grad(V))(Z_blob)
    print(grads.shape)
    lip = torch.max(torch.linalg.norm(grads.squeeze(), dim=-1))
    #lip=torch.tensor([0.])
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
    '''

    

