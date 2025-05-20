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

    #print(Z.shape)

    #print(X.shape)
    #Z = ae.encode(torch.tensor(X.reshape(-1, params.d_x))).reshape(X.shape[0], X.shape[1], params.d_z).cpu().detach().numpy()

    if not only_rollout:
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


# DX: trajectories, shape -- (N, T, d_x)
def plot_violation(V, rho, Dx, ae, L, gamma, a0, n_per_axis=150, plot_contours=True):
    #Dz = ae.encode(Dx.reshape(-1, params.d_x)).reshape(Dx.shape[0], Dx.shape[1], params.d_z)
    Dz = Dx
    fig, ax = plt.subplots(1)
    ax.set_title("V(f(x))-rho*V(x)")
    fig.set_size_inches(10, 10)
    VDz = V(Dz.reshape(-1, params.d_z)).reshape(Dz.shape[0], Dz.shape[1], 1)
    dVDz = (VDz[:,1:] - rho*VDz[:,:-1]).reshape(-1)
    print(dVDz.shape)
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


def plot_figure_final(V, ae, EX, r_ax0, r_ax1, res, a0, lyp, n_per_axis=200, n_samples=None):

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

            '''
            if n_samples is None:
                n_samples = n_per_axis**2
            X_samples = 2*torch.tensor([[rx, ry]])*(torch.rand(n_samples, 2) - 0.5)

            # (th, w) slice
            if i == 0:
                X_samples_full_coords = torch.hstack([torch.zeros_like(X_samples), X_samples])
            # (th, x) slice
            elif i == 1:
                X_samples_full_coords = torch.hstack([X_samples[:,0].reshape(-1, 1),
                                                      torch.zeros(n_samples, 1),
                                                      X_samples[:,1].reshape(-1, 1),
                                                      torch.zeros(n_samples, 1)])
            V_samples = V(ae.encode(X_samples_full_coords))
            X_samples = X_samples.cpu().detach().numpy()

            for v, x in zip(V_samples, X_samples):
                if v <= res:
                    ax.plot(x[0], x[1], color='pink', marker='.')
                elif v <= lyp:
                    ax.plot(x[0], x[1], color='green', marker='.')
                elif v <= a0:
                    ax.plot(x[0], x[1], color='blue', marker='.')
                else:
                    ax.plot(x[0], x[1], color='red', marker='.') 
            '''

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
            print(sns.color_palette("tab10"))
            cm = sns.color_palette("Set2")
            cs = ax.contourf(XX, YY, ZZ, levels=[0, res/15, res, lyp, a0, 100*a0], colors=['red', cm[-3], cm[-4], sns.color_palette("Spectral")[-1], cm[2]])
            ax.contour(XX, YY, ZZ, levels=[0, res, lyp, a0, 100*a0], colors=['k', 'k', 'k', 'k'])

            print(cs.get_facecolors())
            proxy = [plt.Rectangle((0,0),1,1,fc=fc,ec='k') for fc in cs.get_facecolors()]

            ax.legend(proxy, [r'$E^{-1}(0)$', r'$\overline{V}(x)\leq\max_{x\in E^{-1}(Z)} R(x)$', r'$\overline{V}(x)\leq L \gamma/(1-\rho)$', r'$\overline{V}(x)\leq \alpha_0$'])

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
                ax.plot(Zi[:,0], Zi[:,1], alpha=0.25)
            
            cm = sns.color_palette("Set2")
            cntr_outlines = ax.contour(XX.cpu().detach().numpy(), YY.cpu().detach().numpy(), VV.cpu().detach().numpy(), [res, lyp, a0], colors=['k', 'k', 'k'], linewidths=4)
            cntr = ax.contour(XX.cpu().detach().numpy(), YY.cpu().detach().numpy(), VV.cpu().detach().numpy(), [res, lyp, a0], 
                              colors=[cm[-3], cm[-4], sns.color_palette("Spectral")[-1]], linewidths=2)
            proxy = [plt.Rectangle((0,0),1,1,fc=fc,ec='k') for fc in cntr.get_edgecolors()]

            ax.legend(proxy, [r'$V(z)=\max_{x\in E^{-1}(D_z)}R(x)$', r'$V(z)=L\gamma/(1-\rho)$', r'$V(z)=\alpha_0$'])

            #cntr_outlines = ax.contour(XX.cpu().detach().numpy(), YY.cpu().detach().numpy(), VV.cpu().detach().numpy(), [R, a0], colors=['k', 'k'], linewidths=4)
            #cntr = ax.contour(XX.cpu().detach().numpy(), YY.cpu().detach().numpy(), VV.cpu().detach().numpy(), [R, a0], colors=['g', 'w'], linewidths=2)
            #cntr_outlines2 = ax.contour(XX.cpu().detach().numpy(), YY.cpu().detach().numpy(), VV.cpu().detach().numpy(), [L*gamma*(1/(1-rho))], colors=['k'], linewidths=4)
            #cntr2 = ax.contour(XX.cpu().detach().numpy(), YY.cpu().detach().numpy(), VV.cpu().detach().numpy(), [L*gamma*(1/(1-rho))], colors=['r'], linewidths=2)
                        #ax.legend(proxy, ["", ""])

            
            #fig.colorbar(cf)
            ax.set_title("$V(z)$", fontsize=16)
            ax.set_ylabel("$z_2$", fontsize=16, labelpad=-8)
            ax.set_xlabel("$z_1$", fontsize=16, labelpad=0)



        if i == 3:
            Z_proj = EX
            plt.title("$(V \circ E)(x(t))$ ", fontsize=16)
            print(Z_proj.shape[0])
            Tmax = int(Z_proj.shape[1]/100)
            Vz = V(torch.tensor(Z_proj.reshape(-1, params.d_z))).reshape(Z_proj.shape[0], Z_proj.shape[1])
            Vzmax = torch.max(Vz[:,Tmax:])
            #colors = sns.color_palette("viridis", n_colors=Z_proj.shape[0]+66)
            colors = sns.light_palette("seagreen", n_colors=Z_proj.shape[0]-100, reverse=False)

            #for idx, (z_proj, color) in enumerate(zip(Z_proj, colors[:-50])):
            for idx, (z_proj, color) in enumerate(zip(Z_proj, colors[:-50])):

                T = z_proj.shape[0]
                ax.plot(V(torch.tensor(z_proj))[:int(T/2)].cpu().detach().numpy(), color=color, alpha=-(1-0.3)/(len(Z_proj)-1)*idx + 1) ### THIS ONE WAS FOR PREVIOUS PLOT

                #ax.plot(V(torch.tensor(z_proj))[:int(T/3)].cpu().detach().numpy(), color=color, alpha=-(1-0.3)/(len(Z_proj)-1)*idx + 1) ### THIS ONE WAS FOR PREVIOUS PLOT
                #ax.plot(V(torch.tensor(z_proj))[:int(T/3)].cpu().detach().numpy(), alpha=0.1)

            #ax.axhline(y = 0.025, color = 'k', linestyle = '--', alpha=0.4, label="Attractive Invariant Set")
            ax.axhline(y = res, color = 'k', linestyle = '--', alpha=0.4, label=r'$\overline{V}(x)=\max_{x\in E^{-1}(D_z)}R(x)$')
            ax.axhline(y = lyp, color = 'k', linestyle = ':', alpha=0.4, label=r'$\overline{V}(x)=L\gamma/\rho$')
            ax.legend()
            ax.set_xlabel("$t$", fontsize=16, labelpad=0)
           



    plt.show()
            

                
                


            


def plot_figure(V, LQR, ae, rx1, rx2, Z, Z_proj, ais, alpha, res, grid_dens=100):
    fig = plt.figure(figsize=(12, 12))
    gs = plt.GridSpec(2, 2)
    ax_1 = fig.add_subplot(gs[0, 0])
    ax_2 = fig.add_subplot(gs[0, 1])
    ax_3 = fig.add_subplot(gs[1, 0])
    ax_4 = fig.add_subplot(gs[1, 1])
    axes = [ax_1, ax_2, ax_3, ax_4]
    plt.subplots_adjust(hspace=0.20, wspace=0.2) 

    for i, ax in enumerate(axes):
        if i == 0 or i == 1:
            if i == 0:
                X1a = torch.linspace(-rx1[0], rx1[0], grid_dens)
                X2a = torch.linspace(-rx2[0], rx2[0], grid_dens)
                X1b = torch.linspace(-rx1[0], rx1[0], int(grid_dens)) #grid_dens / 6
                X2b = torch.linspace(-rx2[0], rx2[0], int(grid_dens)) #grid_dens / 6
            elif i == 1:
                X1a = torch.linspace(-rx1[1], rx1[1], grid_dens)
                X2a = torch.linspace(-rx2[1], rx2[1], grid_dens)
                X1b = torch.linspace(-rx1[1], rx1[1], int(grid_dens)) #grid_dens / 6
                X2b = torch.linspace(-rx2[1], rx2[1], int(grid_dens)) #grid_dens / 6

            XXa, YYa = torch.meshgrid(X1a, X2a)
            XXb, YYb = torch.meshgrid(X1b, X2b)

            if i == 0:
                pts_2da = torch.dstack([XXa, YYa]).reshape(-1, 2)
                pts_4da = torch.hstack([torch.zeros_like(pts_2da), pts_2da])
                pts_2db = torch.dstack([XXb, YYb]).reshape(-1, 2)
                pts_4db = torch.hstack([torch.zeros_like(pts_2db), pts_2db])
            elif i == 1:
                pts_4db = torch.hstack([torch.zeros_like(pts_2db), pts_2db])
            elif i == 1:
                pts_2da = torch.dstack([XXa, YYa]).reshape(-1, 2)
                pts_4da = torch.hstack([pts_2da[:,0].unsqueeze(-1), torch.zeros_like(pts_2da[:,0].unsqueeze(-1)), 
                                        pts_2da[:,1].unsqueeze(-1), torch.zeros_like(pts_2da[:,1].unsqueeze(-1))])
                pts_2db = torch.dstack([XXb, YYb]).reshape(-1, 2)
                pts_4db = torch.hstack([pts_2db[:,0].unsqueeze(-1), torch.zeros_like(pts_2db[:,0].unsqueeze(-1)), 
                                        pts_2db[:,1].unsqueeze(-1), torch.zeros_like(pts_2db[:,1].unsqueeze(-1))])


                #pts_4db = torch.hstack([pts_2db, torch.zeros_like(pts_2db)])

            pts_Za = ae.encode(pts_4da)
            pts_Zb = ae.encode(pts_4db)

            #for pt in pts_Za:
            #    if V(pt) <= alpha:
            #        pt = pt.cpu().detach().numpy()
            #        ax.plot(pt[0], pt[1], 'bo')
            #    else:
            #        pt = pt.cpu().detach().numpy()
            #        ax.plot(pt[0], pt[1], 'ro')
            VV = V(pts_Za).reshape(grid_dens, grid_dens)
            ax.contour(XXa.cpu().detach().numpy(), YYa.cpu().detach().numpy(), VV.cpu().detach().numpy(), levels=[res, alpha])

            u_Z = LQR(pts_Zb).T
            '''
            if i == 0:
                ax.set_title("$(V \circ E)(x)$ and $f(x, (\pi\circ E)(x))$, $\dot{x}_1=x_1=0$", fontsize=16)
                ax.set_ylabel("$\dot{x}_2$", fontsize=16, labelpad=-8)
                ax.set_xlabel("$x_2$", fontsize=16, labelpad=0)
                fxu = cartpole.dxdt_torch(pts_4db, u_Z.to("cuda"))[:,(2, 3)]
            if i == 1:
                ax.set_ylabel("$x_2$", fontsize=16, labelpad=-8)
                ax.set_xlabel("$x_1$", fontsize=16, labelpad=0)
                ax.set_title("$(V \circ E)(x)$ and $f(x, (\pi\circ E)(x))$, $\dot{x}_1=\dot{x}_2=0$", fontsize=16)
                fxu = cartpole.dxdt_torch(pts_4db, u_Z.to("cuda"))[:,(0,2)]
            '''
 
            #ax.quiver(XXb.cpu().detach().numpy(), YYb.cpu().detach().numpy(), fxu[:,0].cpu().detach().numpy(), fxu[:,1].cpu().detach().numpy(), label='$f(x,(\pi\circ E)(x))$', 
            #          color=sns.color_palette("Set2")[1])
            #cntr = ax.contour(XX.cpu().detach().numpy(), YY.cpu().detach().numpy(), VV.cpu().detach().numpy(), [L*gamma*(1/(1-rho)), a0], colors=['r', 'w'], linewidths=2)
            #proxy = [plt.Rectangle((0,0),1,1,fc=fc) for fc in cntr.get_edgecolors()]
            #ax.legend(proxy, ["Ly/(1-p)", "a0"])

            #ax.cntr 
            ax.legend()

        if i == 2:
            print("gets into 2")
            Z = torch.tensor(Z).float().to("cuda")

            Z_blob = Z.reshape(-1, params.d_z)
            Z1_max = torch.max(Z_blob[:,0])
            Z1_max += 0.1 * Z1_max
            Z1_min = torch.min(Z_blob[:,0])
            Z1_min -= 0.1 * torch.abs(Z1_min)
            Z2_max = torch.max(Z_blob[:,1])
            Z2_max += 0.1 * Z2_max
            Z2_min = torch.min(Z_blob[:,1])
            Z2_min -= 0.1 * torch.abs(Z2_min)
            Z1_pts = torch.linspace(Z1_min, Z1_max, grid_dens)
            Z2_pts = torch.linspace(Z2_min, Z2_max, grid_dens)
            ZZ, WW = torch.meshgrid(Z1_pts, Z2_pts)

            grads = vmap(jacrev(V),in_dims=0)(Z_blob)
            lip = torch.max(torch.linalg.norm(grads.squeeze(), dim=-1))
            VV = V(torch.dstack([ZZ, WW]).reshape(-1, params.d_z)).reshape(grid_dens, grid_dens)
            cf = ax.contourf(ZZ.cpu().detach().numpy(), WW.cpu().detach().numpy(), VV.cpu().detach().numpy(), levels=50)
            #cf2 = ax.contour(ZZ.cpu().detach().numpy(), WW.cpu().detach().numpy(), VV.cpu().detach().numpy(), levels=[ais, alpha], colors=2*['w'])
            cf2 = ax.contour(ZZ.cpu().detach().numpy(), WW.cpu().detach().numpy(), VV.cpu().detach().numpy(), levels=[ais], colors=['w'])
            cf3 = ax.contour(ZZ.cpu().detach().numpy(), WW.cpu().detach().numpy(), VV.cpu().detach().numpy(), levels=[alpha], colors=['w'])
            ax.clabel(cf2, inline=True, colors=['w'], fontsize=14)
            ax.clabel(cf3, inline=True, colors=['w'], fontsize=14)


            fig.colorbar(cf)
            for i, Zi in enumerate(Z):
                if i == 100:
                    ax.plot(Zi[:,0].cpu().detach().numpy(), Zi[:,1].cpu().detach().numpy(), alpha=0.25, label='$f_z(z,\pi(z))$',color='red')
                else:
                    ax.plot(Zi[:,0].cpu().detach().numpy(), Zi[:,1].cpu().detach().numpy(), alpha=0.25)
            ax.legend()
            ax.set_title("$V(z)$", fontsize=16)
            ax.set_ylabel("$z_2$", fontsize=16, labelpad=-8)
            ax.set_xlabel("$z_1$", fontsize=16, labelpad=0)


        if i == 3:
            print("gets into 3")
            plt.title("$(V \circ E)(x(t))$ ", fontsize=16)
            print(Z_proj.shape[0])
            Tmax = int(Z_proj.shape[1]/100)
            Vz = V(torch.tensor(Z_proj.reshape(-1, params.d_z))).reshape(Z_proj.shape[0], Z_proj.shape[1])
            Vzmax = torch.max(Vz[:,Tmax:])
            #colors = sns.color_palette("viridis", n_colors=Z_proj.shape[0]+66)
            colors = sns.light_palette("seagreen", n_colors=Z_proj.shape[0]-100, reverse=False)

            #for idx, (z_proj, color) in enumerate(zip(Z_proj, colors[:-50])):
            for idx, (z_proj, color) in enumerate(zip(Z_proj, colors[:-50])):

                T = z_proj.shape[0]
                ax.plot(V(torch.tensor(z_proj))[:int(T/2)].cpu().detach().numpy(), color=color, alpha=-(1-0.3)/(len(Z_proj)-1)*idx + 1) ### THIS ONE WAS FOR PREVIOUS PLOT

                #ax.plot(V(torch.tensor(z_proj))[:int(T/3)].cpu().detach().numpy(), color=color, alpha=-(1-0.3)/(len(Z_proj)-1)*idx + 1) ### THIS ONE WAS FOR PREVIOUS PLOT
                #ax.plot(V(torch.tensor(z_proj))[:int(T/3)].cpu().detach().numpy(), alpha=0.1)

            #ax.axhline(y = 0.025, color = 'k', linestyle = '--', alpha=0.4, label="Attractive Invariant Set")
            ax.axhline(y = ais, color = 'k', linestyle = '--', alpha=0.4, label="Attractive Invariant Set")
            ax.legend()
            ax.set_xlabel("$t$", fontsize=16, labelpad=0)



    plt.show()


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
            print(r)
            grid_axes.append(np.linspace(r[0], r[1], n_per_axis))
        print(np.meshgrid(grid_axes))
        initial_conditions = np.stack(np.meshgrid(grid_axes), axis=-1).reshape(-1, params.d_x)
        print(initial_conditions.shape)
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
        #axes1[0].plot(traj[:,0])
        #axes1[1].plot(traj[:,1])
        #axes1[2].plot(traj[:,2])
        #axes1[3].plot(traj[:,3])
        
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

    #plt.plot(traj[:,0], label='x')
    #plt.plot(traj[:,1], label='v')
    #plt.plot(traj[:,2], label='theta')
    #plt.plot(traj[:,3], label='theta-dot')
    #plt.plot(dists, label="distance to trajectories")

    #plt.legend()
    plt.show()
        
    return traj


def sample_from_eps_net(X, eps, n_samples):
    N = X.shape[0]*X.shape[1]
    idx = np.random.randint(0, N, size=n_samples)
    #print("T1 shape:", X.reshape(-1, params.d_x)[idx].shape)
    #print("T2 shape:", (eps*(np.random.rand(N, params.d_x) - 0.5)).shape)
    X_sample = X.reshape(-1, params.d_x)[idx] + eps*(np.random.rand(n_samples, params.d_x) - 0.5)
    return X_sample



