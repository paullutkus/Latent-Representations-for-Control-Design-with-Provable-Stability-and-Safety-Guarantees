import numpy as np
import torch
import matplotlib.pyplot as plt
import params 
from scipy.spatial import ConvexHull
from matplotlib.path import Path
from controls import LQR
from utils import rollout_trajectories
import cartpole
from torch.func import vmap, jacrev 
import seaborn as sns
from integration import _flow
from tqdm import tqdm



# compute preimage of D_z in x-space
# Z (ndarray): preimage argument
# r (float): grid each axis over [-r, r]
# n_per_axis (int): grid points per axis
def compute_preimage(ae, Z, r, n_per_axis, uniform_sampling=False, V=None, X=None, n_samples=None, eps=None):
    if X is not None:
        pts = torch.tensor(sample_from_eps_net(X, eps, n_samples)).float()
    elif uniform_sampling: 
        pts = 2*r*(torch.rand((n_per_axis**4, params.d_x)) - 0.5)
    else:
        axes = []
        for i in range(params.d_x):
            axes.append(torch.linspace(-r, r, n_per_axis))
        pts = torch.dstack(torch.meshgrid(axes)).reshape(-1, params.d_x)

    EX = ae.encode(pts).cpu().detach().numpy()
    
    # Compute convex hull of D_z, and define path 
    # object to check inclusion
    hull = ConvexHull(Z.reshape(-1, params.d_z))
    hull_path = Path(Z.reshape(-1, params.d_z)[hull.vertices])

    preimage = []
    for x, z in zip(pts.cpu().detach().numpy(), EX):
        if hull_path.contains_point((z[0], z[1])):
            preimage.append(x)
    preimage = np.array(preimage)
    return preimage


# displays and plots alpha_0, Residual, L\gamma/(1-\rho), etc. levelsets in the latent space
def plot_lyapunov_lvlsets(V, ae, fdyn, X, Z, a0, rho, n_per_axis=100, only_rollout=False):
    lqr = LQR(ae, fdyn)
    Dx = X[(V(ae.encode(torch.tensor(X)))).cpu() <= a0]
    _, _, _, (gamma, _), L, R = rollout_trajectories(ae, fdyn, lqr, Dx, n_traj=200, T=200, plot=True, V=V, a0=a0, n_per_axis=n_per_axis)

    print("gamma fwd:", gamma)
    print("L:", L)
    print("a0:", a0)
    print("rho:", rho)
    print("Ly/p:", L*gamma*(1/(1-rho)))
    print("R:", R)

    _, Z, _, (_, _), _, _ = rollout_trajectories(ae, fdyn, lqr, X, n_traj=200, T=200, plot=False, V=V, a0=a0, n_per_axis=n_per_axis)

    if not only_rollout:
        Zflat = Z.reshape(-1, params.d_z)
        fig, ax = plt.subplots(1)
        fig.set_size_inches(10, 10)

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
        
        #cntr_outlines = ax.contour(XX.cpu().detach().numpy(), YY.cpu().detach().numpy(), VV.cpu().detach().numpy(), [R, L*gamma*(1/(1-rho)), a0], colors=['k', 'k', 'k'], linewidths=4)
        #cntr = ax.contour(XX.cpu().detach().numpy(), YY.cpu().detach().numpy(), VV.cpu().detach().numpy(), [R, L*gamma*(1/(1-rho)), a0], colors=['g', 'r', 'w'], linewidths=2)
        #ax.legend(proxy, ["(VoEof)(x)-(VoFoE)(x)", "Ly/p", "a0"])

        cntr_outlines = ax.contour(XX.cpu().detach().numpy(), YY.cpu().detach().numpy(), VV.cpu().detach().numpy(), [R, a0], colors=['k', 'k'], linewidths=4)
        cntr = ax.contour(XX.cpu().detach().numpy(), YY.cpu().detach().numpy(), VV.cpu().detach().numpy(), [R, a0], colors=['g', 'w'], linewidths=2)
        cntr_outlines2 = ax.contour(XX.cpu().detach().numpy(), YY.cpu().detach().numpy(), VV.cpu().detach().numpy(), [L*gamma*(1/(1-rho))], colors=['k'], linewidths=4)
        cntr2 = ax.contour(XX.cpu().detach().numpy(), YY.cpu().detach().numpy(), VV.cpu().detach().numpy(), [L*gamma*(1/(1-rho))], colors=['r'], linewidths=2)
        proxy = [plt.Rectangle((0,0),1,1,fc=fc) for fc in cntr.get_edgecolors()]
        ax.legend(proxy, ["(VoEof)(x)-(VoFoE)(x)", "a0"])
        plt.show()

    return Z, gamma, L, R


# plot magnitude of violation (if any) of Lyapunov decrease condition
# for trajectories projected into latent space
# Dx: trajectories, shape -- (N, T, d_x)
def plot_violation(V, rho, Dx, ae, L, gamma, a0, n_per_axis=150, plot_contours=True):
    #Dz = ae.encode(Dx.reshape(-1, params.d_x)).reshape(Dx.shape[0], Dx.shape[1], params.d_z)
    Dz = Dx
    fig, ax = plt.subplots(1)
    ax.set_title("V(f(x))-rho*V(x)")
    fig.set_size_inches(10, 10)
    VDz = V(Dz.reshape(-1, params.d_z)).reshape(Dz.shape[0], Dz.shape[1], 1)
    dVDz = (VDz[:,1:] - rho*VDz[:,:-1]).reshape(-1)
    sc = ax.scatter(Dz[:,:-1].reshape(-1, params.d_z).cpu()[:,0], 
                    Dz[:,:-1].reshape(-1, params.d_z).cpu()[:,1], 
                    c=dVDz.cpu().detach().numpy())
    fig.colorbar(sc)

    if plot_contours:
        Zflat = Dz.reshape(-1, params.d_z).cpu().detach().numpy()
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
        
        cntr_outlines = ax.contour(XX.cpu().detach().numpy(), YY.cpu().detach().numpy(), VV.cpu().detach().numpy(), [L*gamma*(1/(1-rho)), a0], colors=['k', 'k'], linewidths=4)
        cntr = ax.contour(XX.cpu().detach().numpy(), YY.cpu().detach().numpy(), VV.cpu().detach().numpy(), [L*gamma*(1/(1-rho)), a0], colors=['r', 'w'], linewidths=2)
        proxy = [plt.Rectangle((0,0),1,1,fc=fc) for fc in cntr.get_edgecolors()]
        ax.legend(proxy, ["Ly/(1-p)", "a0"])

    plt.show()


# render figure used for cartpole Lyapunov example in the paper
def plot_figure_final(V, ae, EX, r_ax0, r_ax1, res, a0, lyp, n_per_axis=200, n_samples=None, xth_traj=None):

    # Initialize figure and axes
    fig = plt.figure(figsize=(12, 12))
    gs = plt.GridSpec(2, 2)
    ax_1 = fig.add_subplot(gs[0, 0])
    ax_2 = fig.add_subplot(gs[0, 1])
    ax_3 = fig.add_subplot(gs[1, 0])
    ax_4 = fig.add_subplot(gs[1, 1])
    axes = [ax_1, ax_2, ax_3, ax_4]
    plt.subplots_adjust(hspace=0.20, wspace=0.2) 

    for i, ax in enumerate(axes):
        if (i == 0) or (i == 1):

            if i == 0:
                (rx, ry) = r_ax0 
            elif i == 1:
                (rx, ry) = r_ax1

            y_ax = np.linspace(-ry, ry, n_per_axis)
            x_ax = np.linspace(-rx, rx, n_per_axis)
            XX, YY = np.meshgrid(x_ax, y_ax) 
            
            # (th, w) slice
            if i == 0:
                X = np.dstack([np.zeros((n_per_axis, n_per_axis)),
                               np.zeros((n_per_axis, n_per_axis)),
                               XX, 
                               YY]).reshape(-1, 4)
            # (th, x) slice
            elif i == 1: 
                X = np.dstack([XX,
                               np.zeros((n_per_axis, n_per_axis)),
                               YY,
                               np.zeros((n_per_axis, n_per_axis))]).reshape(-1, 4)
 
            ZZ = V(ae.encode(torch.tensor(X).float())).reshape(n_per_axis, n_per_axis).cpu().detach().numpy()
            cm = sns.color_palette("Set2")
            cs = ax.contourf(XX, YY, ZZ, levels=[0, res/150, res, lyp, a0, 100*a0], colors=['red', cm[-3], cm[-4], sns.color_palette("Spectral")[-1], cm[2]])
            ax.contour(XX, YY, ZZ, levels=[0, res, lyp, a0, 100*a0], colors=['k', 'k', 'k', 'k'])

            proxy = [plt.Rectangle((0,0),1,1,fc=fc,ec='k') for fc in cs.get_facecolors()]

            ax.legend(proxy, [r'$E^{-1}(0)$', r'$\overline{V}(x)\leq\max_{x\in E^{-1}(\mathcal{D}_z)}\frac{|R(x)|}{1-\rho}$', r'$\overline{V}(x)\leq L \gamma/(1-\rho)$', r'$\overline{V}(x)\leq \alpha_0$'])
            
            if (i == 1) and (xth_traj is not None):
                print("x-theta traj:", xth_traj.shape)
                for traj in xth_traj:
                    ax.plot(traj[:,0], traj[:,2], linestyle='dashed', color='black')
                    ax.plot(traj[0,0], traj[0,2], 'ko', markersize=4)
                    ax.arrow(traj[:,0][-2], traj[:,2][-2], traj[:,0][-1] - traj[:,0][-2], traj[:,2][-1] - traj[:,2][-2], 
                              head_width=0.75*0.01, head_length=0.75*0.02, fc='k', ec='k')

            # (th, w) slice
            if i == 0:
                ax.set_title(r"Sublevel sets of $(V\circ E)(0,0,\theta,\dot{\theta})$", fontsize=16)
                ax.set_ylabel(r'$\theta$', fontsize=16, labelpad=-8)
                ax.set_xlabel(r'$\dot{\theta}$', fontsize=16, labelpad=0)

            # (th, x) slice
            elif i == 1:
                ax.set_title(r"Sublevel sets of $(V\circ E)(x, 0, \theta, 0)$", fontsize=16)
                ax.set_ylabel(r'$\theta$', fontsize=16, labelpad=-8)
                ax.set_xlabel(r'$x$', fontsize=16, labelpad=0)

        if i == 2:
            Z = EX
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

            cf = ax.contourf(XX.cpu().detach().numpy(),YY.cpu().detach().numpy(), VV.cpu().detach().numpy(), levels=50)

            for Zi in Z:
                ax.plot(Zi[:,0], Zi[:,1], alpha=0.25)
            
            cm = sns.color_palette("Set2")
            cntr_outlines = ax.contour(XX.cpu().detach().numpy(), YY.cpu().detach().numpy(), VV.cpu().detach().numpy(), [res, lyp, a0], colors=['k', 'k', 'k'], linewidths=4)
            cntr = ax.contour(XX.cpu().detach().numpy(), YY.cpu().detach().numpy(), VV.cpu().detach().numpy(), [res, lyp, a0], 
                              colors=[cm[-3], cm[-4], sns.color_palette("Spectral")[-1]], linewidths=2)
            proxy = [plt.Rectangle((0,0),1,1,fc=fc,ec='k') for fc in cntr.get_edgecolors()]

            ax.legend(proxy, [r'$V(z)=\max_{x\in E^{-1}(D_z)}\frac{|R(x)|}{1-\rho}$', r'$V(z)=L\gamma/(1-\rho)$', r'$V(z)=\alpha_0$'])
            ax.clabel(cntr, inline=True, colors=['w'], fontsize=14, fmt='%1.1f')
            
            #fig.colorbar(cf)
            ax.set_title("$V(z)$", fontsize=16)
            ax.set_ylabel("$z_2$", fontsize=16, labelpad=-8)
            ax.set_xlabel("$z_1$", fontsize=16, labelpad=0)

        if i == 3:
            Z_proj = EX
            plt.title("$(V \circ E)(x(t))$ ", fontsize=16)
            Tmax = int(Z_proj.shape[1]/100)
            Vz = V(torch.tensor(Z_proj.reshape(-1, params.d_z))).reshape(Z_proj.shape[0], Z_proj.shape[1])
            Vzmax = torch.max(Vz[:,Tmax:])
            colors = sns.light_palette("seagreen", n_colors=Z_proj.shape[0]-100, reverse=False)

            for idx, (z_proj, color) in enumerate(zip(Z_proj, colors[:-50])):
                T = z_proj.shape[0]
                ax.plot(V(torch.tensor(z_proj))[:int(T/2)].cpu().detach().numpy(), color=color, alpha=-(1-0.3)/(len(Z_proj)-1)*idx + 1) 

            ax.axhline(y = res, color = 'k', linestyle = '--', alpha=0.9, linewidth=2.0, label=r'$\overline{V}(x)=\max_{x\in E^{-1}(D_z)}\frac{|R(x)|}{1-\rho}$')
            ax.axhline(y = lyp, color = 'k', linestyle = ':', alpha=0.9, linewidth=2.0, label=r'$\overline{V}(x)=L\gamma/(1-\rho)$') 
            ax.legend()
            ax.set_xlabel("$t$", fontsize=16, labelpad=0)
           
    plt.show()
            

# inspect whether the epsilon-net of a set of trajectories is forward invariant by sampling
def verify_invariance(ae, fdyn, ranges=[[-0.1,0.1],[-0.1,0.1],[-0.1,0.1],[-0.1,0.1]], T=100, stabilize=True, n_per_axis=10, X0=None, manifold=None):
    fig1 = plt.figure(figsize=(12, 12))
    gs = plt.GridSpec(2, 2)
    ax_11 = fig1.add_subplot(gs[0, 0])
    ax_12 = fig1.add_subplot(gs[0, 1])
    ax_13 = fig1.add_subplot(gs[1, 0])
    ax_14 = fig1.add_subplot(gs[1, 1])
    axes1 = [ax_11, ax_12, ax_13, ax_14]
    plt.subplots_adjust(hspace=0.20, wspace=0.2) 

    fig2 = plt.figure(figsize=(12, 12))
    gs = plt.GridSpec(2, 2)
    ax_21 = fig2.add_subplot(gs[0, 0])
    ax_22 = fig2.add_subplot(gs[0, 1])
    ax_23 = fig2.add_subplot(gs[1, 0])
    ax_24 = fig2.add_subplot(gs[1, 1])
    axes2 = [ax_21, ax_22, ax_23, ax_24]
    plt.subplots_adjust(hspace=0.20, wspace=0.2) 


    lqr = LQR(ae, fdyn)
    if X0 is None:
        # Create grid
        grid_axes = []
        for r in ranges:
            grid_axes.append(np.linspace(r[0], r[1], n_per_axis))
        initial_conditions = np.stack(np.meshgrid(grid_axes), axis=-1).reshape(-1, params.d_x)
    else:
        initial_conditions = X0.reshape(-1, params.d_x)

    for x in tqdm(initial_conditions):
        x = torch.tensor(x).float()
        z = ae.encode(torch.unsqueeze(x, 0))
        x = x.cpu()
        U = []
        traj = []
        dists = []
        elem_dists = [[], [], [], []]
        traj.append(x.detach().numpy())
        for t in range(T):
            if stabilize:
                u = lqr(z).item()
                U.append(u)
            else:
                u = 0.
            x_prev = x
            x = _flow(x, cartpole.DT, u)[-1]
            dists.append(np.min(np.linalg.norm((manifold.reshape(-1, params.d_x)[:,params.symbols] -\
                                                       x.reshape( 1, params.d_x)[:,params.symbols]), axis=1))) 
            for i in range(params.d_x):
                elem_dists[i].append(np.min(np.abs(manifold.reshape(-1, params.d_x)[:,i] - x.reshape(1, params.d_x)[:,i])))
            traj.append(x)
            z = ae.encode(torch.tensor(x.reshape(-1, 4)).float()) 
        traj = np.array(traj)
        dists = np.array(dists)
        
        for i in range(params.d_x):
            axes1[i].plot(traj[:,i])
            axes2[i].plot(elem_dists[i])

    axes1[0].set_title("x")
    axes1[1].set_title("v")
    axes1[2].set_title("th")
    axes1[3].set_title("w")

    axes2[0].set_title("x-eps")
    axes2[1].set_title("v-eps")
    axes2[2].set_title("th-eps")
    axes2[3].set_title("w-eps")

    plt.show()
        
    return traj


# sample points from within epsilon of a set of trajectories
def sample_from_eps_net(X, eps, n_samples):
    N = X.shape[0]*X.shape[1]
    idx = np.random.randint(0, N, size=n_samples)
    X_sample = X.reshape(-1, params.d_x)[idx] + eps*(np.random.rand(n_samples, params.d_x) - 0.5)
    return X_sample


