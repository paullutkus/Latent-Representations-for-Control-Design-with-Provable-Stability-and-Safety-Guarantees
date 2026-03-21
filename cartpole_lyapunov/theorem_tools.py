import numpy as np
import torch
import matplotlib.pyplot as plt
import params 
from scipy.spatial import ConvexHull
from matplotlib.path import Path
from controls import LQR
from utils import rollout_trajectories, rollout_parallel_rk4
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
def plot_lyapunov_lvlsets(V, ae, fdyn, X, Z, a0, rho, n_per_axis=100, only_rollout=False, k_eval=[2, 3, 4, 5, 6, 7, 8, 9, 10]):
    lqr = LQR(ae, fdyn)
    Dx = X[(V(ae.encode(torch.tensor(X)))).cpu() <= a0]
    k_max = 200
    #k_eval = np.array([2, 3, 4, k_max])
    k_eval = np.array(k_eval + [k_max])
    print("k_eval:", k_eval)
    _, _, _, (gamma, _), L, R, gamma_k = rollout_trajectories(ae, fdyn, lqr, Dx, n_traj=200, T=k_max, plot=True, V=V, a0=a0, n_per_axis=n_per_axis, mstep_gammas=k_eval)

    print("gamma fwd:", gamma)
    print("L:", L)
    print("a0:", a0)
    print("rho:", rho)
    print("Ly/(1-p):", L*gamma*(1/(1-rho)))
    print("R:", R)

    print("k:", k_eval)
    print("gamma_k forward:", gamma_k)
    print("Ly_k/(1-p^k):", L*gamma_k/(1-rho**k_eval))

    
    plt.plot([L*gamma*(1/(1-rho))] + list((L*gamma_k/(1-rho**k_eval))), label="Ly_k/(1-p^k):")
    print("xticks arg 1", np.arange(len(k_eval)+1))
    print("mid", k_eval)
    print("xticks arg 2", [1] + list(k_eval))
    plt.xticks(np.arange(len(k_eval)+1), [1] + list(k_eval))
    plt.legend()
    plt.show()

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

def plot_eigenvalues(ae, fdyn, lqr, roa, roa_per_axis, T_roa=3500, n_chunks=0):
        roa_axes_x_chunks = np.linspace(-roa[0], roa[0], n_chunks+1)
        X_roa_flatten_vw_chunks = []
        X_roa_success_alpha_chunks = []
        for chunk_idx in range(n_chunks):
            roa_axes = [np.linspace(roa_axes_x_chunks[chunk_idx], roa_axes_x_chunks[chunk_idx+1], roa_per_axis[0] // n_chunks)]
            for ii in range(1, params.d_x):
                roa_axes.append(np.linspace(-roa[ii], roa[ii], roa_per_axis[ii]))
            roa_pts = np.stack(np.meshgrid(*roa_axes, indexing='ij'), axis=-1)
            roa_og_shape = roa_pts.shape

            ### Rollout Trajectories ###
            roa_pts = roa_pts.reshape(-1, params.d_x)
            #T_roa = 6000
            roa_pts = roa_pts.astype('float32')
            X_roa, U_roa = rollout_parallel_rk4(ae, fdyn, lqr, roa_pts, T_roa, save=True, return_u=True)
            #U_roa = U_roa[0]
            #for X_traj in X_roa:
            #    ax.plot(X_traj[:,0], X_traj[:,1])
            #f_cl = lambda x: cartpole.dxdt_torch(x, lqr(ae.encode(x)).to(device='cuda'))
            def f_cl(x):
                z = ae.encode(x.reshape(1, 4))
                u = lqr(z).to(device='cuda')
                x = x.reshape(1, 4)
                dxdt = cartpole.dxdt_torch(x, u)
                return dxdt
            X_roa = torch.tensor(X_roa).to(device='cuda')
            print(X_roa.shape)
            #dxdt = f_cl(X_roa)
            jac_fn = torch.func.vmap(torch.func.jacfwd(f_cl))
            grads = jac_fn(X_roa[chunk_idx]).squeeze()
            eigs = torch.linalg.eigvals(grads)

            #print(dxdt.shape)
            #print(grads.shape)
            #print(eigs.shape)
            for i in range(4):
                plt.title("eigenvalues over time")
                plt.plot(eigs[:,i].cpu().detach().numpy())
            plt.show()

            min_eigs = torch.min(torch.abs(torch.real(eigs)), dim=1)[0].cpu().detach().numpy()
            plt.title("min eigenvalue")
            plt.plot(min_eigs)
            plt.show()

            max_eigs = torch.max(torch.abs(torch.real(eigs)), dim=1)[0].cpu().detach().numpy()
            plt.title("max eigenvalue")
            plt.plot(max_eigs)
            plt.show()

            for X in X_roa:
                X = X.cpu().detach().numpy()
                plt.title("x(t)", fontsize=18)
                plt.plot(X[:,0])
                plt.ylabel("x", fontsize=12)
                plt.xlabel("t", fontsize=12)
            plt.show()

            for X in X_roa:
                X = X.cpu().detach().numpy()
                plt.title("Theta(t)", fontsize=18)
                plt.plot(X[:,2])
                plt.ylabel("Theta", fontsize=12)
                plt.xlabel("t", fontsize=12)
            plt.show()

            for X in X_roa:
                X = X.cpu().detach().numpy()
                plt.title("Cart acc.", fontsize=18)
                plt.plot(X[1:,1]- X[:-1,1])
                plt.ylabel("acc.", fontsize=12)
                plt.xlabel("t", fontsize=12)
            plt.show()

            print("U ROA SHAPE", U_roa.shape)
            for U in U_roa:
                #U = U.cpu().detach().numpy()
                plt.title("u(t) Steady-state", fontsize=18)
                plt.plot(U[200:])
                plt.ylabel("u", fontsize=12)
                plt.xlabel("t", fontsize=12)
                #locs, labels = plt.xticks()
                #for label in labels:
                #    print(label.get_text())
                #plt.xticks(locs, labels)
                plt.xticks([])
            plt.show()

            for X in X_roa:
                X = X.cpu().detach().numpy()
                plt.title("theta-x phase portrait")
                plt.plot(X[:,0], X[:,2])
            plt.show()
            #for x in dxdt[0]:
            #    print(x)

# render figure used for cartpole Lyapunov example in the paper
def plot_figure_final(V, ae, EX, r_ax0, r_ax1, res, a0, lyp, n_per_axis=200, n_samples=None, xth_traj=None, gamma_k_lvl=None, 
                      roa=False, roa_per_axis=None, fdyn=None, lqr=None, save_traj=False, n_chunks=0, X_test=None):

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


            ### BEGIN REGION OF ATTRACTION PLOTS ###
            #y_ax_roa = np.linspace(-roa, roa, roa_per_axis)
            #x_ax_roa = np.linspace(-roa, roa, roa_per_axis)
            #XX_roa, YY_roa = np.meshgrid(x_ax, y_ax)
            ### Create initial hypercube grid and rollout trajectories on grid
            
            if (roa is not None) and (i == 1):
                roa_axes_x_chunks = np.linspace(-roa[0], roa[0], n_chunks+1)
                X_roa_flatten_vw_chunks = []
                X_roa_success_alpha_chunks = []
                for chunk_idx in range(n_chunks):
                    roa_axes = [np.linspace(roa_axes_x_chunks[chunk_idx], roa_axes_x_chunks[chunk_idx+1], roa_per_axis[0] // n_chunks)]
                    ### THIS IS THE PROBLEM ###
                    for ii in range(1, params.d_x):
                        roa_axes.append(np.linspace(-roa[ii], roa[ii], roa_per_axis[ii]))
                    #print(np.meshgrid(*roa_axes))
                    roa_pts = np.stack(np.meshgrid(*roa_axes, indexing='ij'), axis=-1)
                    roa_og_shape = roa_pts.shape

                    ### Rollout Trajectories ###
                    roa_pts = roa_pts.reshape(-1, params.d_x)
                    T_roa = 3500
                    #X_roa, _, _, _ = rollout_trajectories(ae, fdyn, lqr, roa_pts, T=T_roa, plot=True)
                    roa_pts = roa_pts.astype('float32')
                    X_roa = rollout_parallel_rk4(ae, fdyn, lqr, roa_pts, T_roa, save=save_traj)
                    if not save_traj:
                        T_roa=1
                    #for X_traj in X_roa:
                    #    ax.plot(X_traj[:,0], X_traj[:,1])
                    #print("comparison", np.sum((X_roa[:,0].reshape(roa_og_shape) - roa_pts.reshape(roa_og_shape))**2))

                    ### Swap v and theta axes ###
                    X_roa_xth_index = np.transpose(X_roa.reshape(roa_og_shape[:-1]+(-1, params.d_x)), (0, 2, 1, 3, 4, 5))
                    #X_roa_xth_index2 = np.swapaxes(X_roa.reshape(roa_og_shape[:-1] + (101, 4)), 1, 2)
                    #print("trans v swap", np.sum((X_roa_xth_index1 - X_roa_xth_index2)**2))

                    #X_roa_xth_index = X_roa.reshape(roa_og_shape[:-1] + (101, 4))
                    #print("x_roa_xth_index shape", X_roa_xth_index.shape)

                    #for ii in [0, 1, 2]:
                    #    for jj in [0, 1, 2]:
                    #        for X in X_roa_xth_index[ii,jj].reshape(-1, T_roa+1, params.d_x):
                    #            ax.plot(X[:,0], X[:,2])
                    #for X in X_roa_xth_index[1,0].reshape(-1, 101, 4):
                    #    ax.plot(X[:,0], X[:,2])
                    #for X in X_roa_xth_index[2,0].reshape(-1, 101, 4):
                    #    ax.plot(X[:,0], X[:,2])

                    ### Compute success rate for trajectories
                    #print("roa trajectories shape:", X_roa.shape)
                    # hypothesis: shape should be (N, T, dx)
                    X_roa_flatten_vw = X_roa_xth_index.reshape(X_roa_xth_index.shape[0], 
                                                               X_roa_xth_index.shape[1], 
                                                               X_roa_xth_index.shape[2]*X_roa_xth_index.shape[3],
                                                               T_roa+1,
                                                               params.d_x)
                    X_roa_final_pos = X_roa_flatten_vw[:,:,:,-1,0]
                    X_roa_init_pos  = X_roa_flatten_vw[:,:,:, 0,0]
                    X_roa_final_angle= X_roa_flatten_vw[:,:,:,-1, 2]
                    X_roa_final_w= X_roa_flatten_vw[:,:,:,-1, 3]

                    #print("X_roa_final_pos\n", X_roa_final_pos)
                    #print("X_roa_init_pos\n", X_roa_init_pos)
                    #print("ratio shape:", X_roa_pos_ratio.shape)
                    # 80% of initial position means success
                    #X_roa_success_bool = X_roa_pos_ratio <= 10000.
                    #X_roa_pos_ratio = np.abs(X_roa_final_pos)+1e-4 / np.abs(X_roa_init_pos)+1e-4
                    X_roa_success_bool = np.logical_and((np.abs(X_roa_final_pos) <= roa[0]/2), (np.abs(X_roa_final_angle) <= 0.1)) #roa[0] / 2
                    X_roa_success_bool = np.logical_and(X_roa_success_bool, (np.abs(X_roa_final_w) <= 0.1))
                    #print("roa succ bool", X_roa_success_bool.shape)
                    #print("X_roa succ bool sum ax 2", np.sum(X_roa_success_bool, axis=2).shape)
                    X_roa_success_alpha = np.sum(X_roa_success_bool, axis=2) / X_roa_success_bool.shape[-1]

                    X_roa_flatten_vw_chunks.append(X_roa_flatten_vw)
                    X_roa_success_alpha_chunks.append(X_roa_success_alpha)


                    #print("success bool shape", X_roa_success_bool.shape)
                    
                # USE ALPHA VALUES TO ASSESS PERCENT SUCCESS

                ### END REGION OF ATTRACTION PLOTS ### 

            
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
            ax.contour(XX, YY, ZZ, levels=[gamma_k_lvl], colors=['k'])
      
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

            if (roa is not None) and (i == 1):
                # plot success bool 
                #print("about to plot")
                #print(X_roa_success_bool)
                for ii in range(n_chunks):
                    X_roa_flatten_vw = X_roa_flatten_vw_chunks[ii]
                    X_roa_success_alpha = X_roa_success_alpha_chunks[ii]
                    ax.scatter(X_roa_flatten_vw[:,:,0,0].reshape(-1,4)[:,0], 
                               X_roa_flatten_vw[:,:,0,0].reshape(-1,4)[:,2],
                               color='b', alpha=X_roa_success_alpha.reshape(-1))
                if X_test is not None:
                    X_test = X_test.cpu().detach().numpy()
                    for X_test_traj in tqdm(X_test[:15000]):
                        ax.plot(X_test_traj[:,0], X_test_traj[:,2])
                #print("done plotting")


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


