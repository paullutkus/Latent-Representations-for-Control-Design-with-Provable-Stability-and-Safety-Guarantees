import torch
from torch.func import vmap, jacrev 
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
import gymnasium as gym
import params
import cartpole
#from torchdiffeq import odeint_adjoint as odeint
from torchdiffeq import odeint
from integration import _flow_diff, _flow
from cartpole import CartpoleRenderer
from controls import LQR
from losses import cartpole_reward, gamma_backwards, gamma_forwards
from tqdm import tqdm



# plots metrics for grid of trained models
def plot_experiment_new(exp):
    metrics = ["rewards", "completion_rate", "gamma"]
    n_combinations = len(exp[metrics[0]])
    plt.cla()
    for s in metrics:
        fig, ax = plt.subplots()
        fig.set_size_inches(10, 10)
        ax.set_title("{} vs training epoch".format(s))
        for i in range(n_combinations):
            configuration = exp[s][i]
            for j, run in enumerate(configuration[1]):
                ax.plot(run, label=configuration[0]+"\nrun {}".format(j+1))
        ax.legend(fontsize='xx-small')
        plt.show()

        fig, ax = plt.subplots()
        fig.set_size_inches(10, 10)
        ax.set_title("{} vs training epoch (avg'd over {} runs)".format(s, len(exp[s][i][1])))
        for i in range(n_combinations):
            if len(exp[s][i][1]) > 0:
                all_runs = exp[s][i][1]
                min_len = min([len(run) for run in all_runs])
                avg_run = np.array(all_runs[0][-min_len:]).astype(np.float64)
                for run in all_runs[1:]:
                    avg_run += np.array(run[-min_len:]).astype(np.float64)
                avg_run /= len(all_runs)
                ax.plot(avg_run, label=exp[s][i][0])
        ax.legend(fontsize='xx-small')
        plt.show()


# DEPRECATED: plots slices of \overline{V} in x-space with
# overlayed closed-loop vector field
def plot_lyapunov_slice(V, LQR, ae, rx1, rx2, Z, Z_proj, ais, alpha, grid_dens=100):

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
                X1b = torch.linspace(-rx1[0], rx1[0], int(grid_dens/8)) #grid_dens / 6
                X2b = torch.linspace(-rx2[0], rx2[0], int(grid_dens/8)) #grid_dens / 6
            elif i == 1:
                X1a = torch.linspace(-rx1[1], rx1[1], grid_dens)
                X2a = torch.linspace(-rx2[1], rx2[1], grid_dens)
                X1b = torch.linspace(-rx1[1], rx1[1], int(grid_dens/8)) #grid_dens / 6
                X2b = torch.linspace(-rx2[1], rx2[1], int(grid_dens/8)) #grid_dens / 6

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

            VV = V(pts_Za).reshape(grid_dens, grid_dens)
            ax.contourf(XXa.cpu().detach().numpy(), YYa.cpu().detach().numpy(), VV.cpu().detach().numpy())

            u_Z = LQR(pts_Zb).T
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
 
            ax.quiver(XXb.cpu().detach().numpy(), YYb.cpu().detach().numpy(), fxu[:,0].cpu().detach().numpy(), fxu[:,1].cpu().detach().numpy(), label='$f(x,(\pi\circ E)(x))$', 
                      color=sns.color_palette("Set2")[1])
            ax.legend()

        if i == 2:
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
            cf2 = ax.contour(ZZ.cpu().detach().numpy(), WW.cpu().detach().numpy(), VV.cpu().detach().numpy(), levels=[ais, alpha], colors=2*['w'])
            ax.clabel(cf2, inline=True, colors=['w'], fontsize=14)


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


# plots latent Lyapunov function 
def plot_lyapunov(Z, V, rho, lvls, grid_dens=100, traj_data=None):
    Z = torch.tensor(Z).float().to("cuda")

    Z_blob = Z.reshape(-1, params.d_z)
    Z1_max = torch.max(Z_blob[:,0])
    Z1_min = torch.min(Z_blob[:,0])
    Z2_max = torch.max(Z_blob[:,1])
    Z2_min = torch.min(Z_blob[:,1])
    Z1_pts = torch.linspace(Z1_min, Z1_max, grid_dens)
    Z2_pts = torch.linspace(Z2_min, Z2_max, grid_dens)
    ZZ, WW = torch.meshgrid(Z1_pts, Z2_pts)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(20, 10)
    alpha = torch.max(V(Z_blob)).item()
    grads = vmap(jacrev(V),in_dims=0)(Z_blob)
    lip = torch.max(torch.linalg.norm(grads.squeeze(), dim=-1))
    VV = V(torch.dstack([ZZ, WW]).reshape(-1, params.d_z)).reshape(grid_dens, grid_dens)
    ax1.set_title("Rho: {0:.2f}, Local Lipschitz Over Data: {1:.2f}".format(rho, lip.item()))
    cf = ax1.contourf(ZZ.cpu().detach().numpy(), WW.cpu().detach().numpy(), VV.cpu().detach().numpy(), levels=50)
    fig.colorbar(cf)
    cntr = ax1.contour(ZZ.cpu().detach().numpy(), WW.cpu().detach().numpy(), VV.cpu().detach().numpy(), lvls, colors=len(lvls)*['w'])
    ax1.clabel(cntr, inline=True, colors=len(lvls)*['w'], fontsize=14)
    for Zi in Z:
        ax1.plot(Zi[:,0].cpu().detach().numpy(), Zi[:,1].cpu().detach().numpy(), alpha=0.25)
    ax1.plot(traj_data[0][:,0], traj_data[0][:,1], 'r--', linewidth=3)
    ax2.set_title("(V o E)(x(t))")
    for z_proj in traj_data:
        ax2.plot(V(torch.tensor(z_proj)).cpu().detach().numpy())
    plt.show()


# auxiliary plotting function for quadratic levelsets (probably unused...)
def plot_quadratic_level(ax, ellipse_spec=None, origin_spec=None):
    t = np.linspace(0, 2*np.pi, 100)
    x = 1*np.cos(t)
    y = 1*np.sin(t)
    p = torch.tensor(np.hstack([x.reshape(-1, 1), y.reshape(-1, 1)])).float()

    if ellipse_spec is not None:
        (P, b), h = ellipse_spec
        A = torch.tensor(P).float()
        p_ellipse = torch.linalg.solve(A, (p.T - torch.tensor(b).to("cuda").reshape(2, 1))).T
        ax.plot(p_ellipse[:,0].cpu(), p_ellipse[:,1].cpu())
        ax.plot(1e-2*p[:,0].cpu(), 1e-2*p[:,1].cpu(), 'r-')

    if origin_spec is not None:
        (b, r) = origin_spec
        ax.plot(r*(p[:,0].cpu()-b[0]), 
                r*(p[:,1].cpu()-b[1]), 'r-')


# plots n-number of T-long latent trajects sampled from r-hypercube
def plot_latent_trajectories(ae, fdyn, r, n, T=10, ax=None, plot_quadratic=False, ellipse_spec=None, origin_spec=None, lqr=None):
    ext_ax = True
    if ax is None:
        fig, ax = plt.subplots()
        ext_ax = False

    fdyn_drift, fdyn_cntrl = fdyn
    if lqr is None:
        lqr = LQR(ae, fdyn)
    z_eq = ae.encode(torch.tensor([[0.,0.,0.,0.]]))
    print("z_eq", z_eq)
    pts = []
    traj_dset = []
    x_init_cond = []
    for i in tqdm(range(n)):
        x = 2*r*(torch.rand((1, 4)) - 0.5)
        z = ae.encode(x)
        while torch.linalg.norm(z - z_eq) >= r:
            #z = r*(torch.rand((1, 2)) - 0.5) + z_eq
            x = 2*r*(torch.rand((1, 4)) - 0.5)
            z = ae.encode(x)
        x_init_cond.append(x.cpu().detach().numpy()[0])
        pts.append(z.cpu().detach().numpy()[0])
        traj = [z.cpu().detach().numpy()[0]]
        collected = False
        for t in range(T):
            u = lqr(z).item()
            z_prev = z
            z = (fdyn_drift(z).reshape(1, params.d_z, params.d_z) @ z.unsqueeze(-1) +\
                 fdyn_cntrl(z).reshape(1, params.d_z, params.d_u) @ torch.tensor([u]).float().reshape(1, params.d_u, 1))[:,:,0]
            pts.append(z.cpu().detach().numpy()[0])
            traj.append(z.cpu().detach().numpy()[0])
        traj = np.array(traj)
        traj_dset.append(traj)
        ax.plot(traj[:,0], traj[:,1], marker=".")
        ax.plot(traj[-1,0], traj[-1,1], marker="*", color="red")

    if plot_quadratic:
        plot_quadratic_level(ax, ellipse_spec=ellipse_spec, origin_spec=origin_spec)

    if not ext_ax:
        plt.show()
        return np.array(pts), np.array(traj_dset), np.array(x_init_cond)
    else:
        return ax


# plots the preimage of an r-norm ball by sampling
def plot_preimage_norm_ball(ae, r, i=2, ival=0., j=3, jval=0., n=100):
    N = 4*[True]
    N[i] = False
    N[j] = False
    rem_ind = []
    for k in range(4):
        if N[k] == True:
            rem_ind.append(k)
    x_eq = torch.tensor([[0., 0., 0., 0.]])
    z_eq = ae.encode(x_eq)
    X = 10*(torch.rand((n, 4)) - 0.5)
    X[:,i] = ival
    X[:,j] = jval
    ind = (torch.linalg.norm(ae.encode(X) - z_eq, dim=1) <= r)
    preimage = X[ind]
    not_preimage = X[(~ind)]

    plt.scatter(not_preimage[:,rem_ind[0]], not_preimage[:,rem_ind[1]], color='red')
    plt.scatter(preimage[:,rem_ind[0]], preimage[:,rem_ind[1]])
    plt.show()


# plot the stability (blue) or instability (red) of initial conditions projected into the latent space
# compute gamma-forwards over all trajectories inspected (used as a metric during training)
def plot_stability(ae, fdyn, n_pts, low, high, tol, T, visualize=True, latent_traj=False, video=False, frame=None, compute_gamma=False, disable_plot=False):
    lqr = LQR(ae, fdyn)
    if (frame is not None) and (not disable_plot):
        plt.title(str(frame))
    if not video:
        fig = plt.figure(figsize=(25, 25))
        fig.suptitle("(eps={:.2f}, delta={:.2f})-Robustness on cart position ({} steps) (|angle| < {:.2f})".format(high[2], tol, T, tol))
        if params.d_z == 3:
            ax = fig.add_subplot(projection='3d')
        elif params.d_z == 2:
            axes = [fig.add_subplot(2,2,i) for i in range(1, 5)]
    rewards = []
    success = []
    gamma = []
    if not disable_plot:
        bar = tqdm(range(n_pts))
    else:
        bar = range(n_pts)
    for i in bar:
        x0 = np.random.uniform(low=low, high=high)
        z0 = ae.encode(torch.tensor(x0.reshape(-1, 4)).float()).cpu().detach().numpy()[0]
        x = x0
        if visualize:
            disp = CartpoleRenderer(x)
        terminate = False
        t = 0
        x_traj = [x]
        z_traj = []
        u_traj = []
        while not terminate:
            z = ae.encode(torch.tensor(x.reshape(-1, 4)).float())
            z_traj.append(z)
            u = lqr(z).item()
            x = _flow(x, cartpole.DT, u)[-1]
            u_traj.append(u)
            x_traj.append(x)
            if visualize:
                disp.state = x
                disp.render()
            t += 1
            if abs(x[2]) >= tol:
                success.append(0)
                if not video:
                    for j, ax in enumerate(axes):
                        if params.d_z == 3:
                            ax.scatter(z0[0], z0[1], z0[2], color='red')
                        elif params.d_z == 2:
                            ax.scatter(z0[0], z0[1], color='red', marker='o')
                            ax.scatter(z0[0], z0[1], c=x0[j], marker=".",
                                       cmap='viridis', norm='linear', vmin=low[j], vmax=high[j])
                elif not disable_plot:
                    plt.scatter(z0[0], z0[1], color='red', marker='o')
                terminate = True
            if t >= T:
                success.append(1)
                if not video:
                    for j, ax in enumerate(axes):
                        if params.d_z == 3:
                            ax.scatter(z0[0], z0[1], z0[2], color='blue')
                        elif params.d_z == 2:
                            ax.scatter(z0[0], z0[1], color='blue', marker='o')
                            ax.scatter(z0[0], z0[1], c=x0[j], marker=".",
                                       cmap='viridis', norm='linear', vmin=low[j], vmax=high[j])
                elif not disable_plot:
                    plt.scatter(z0[0], z0[1], color='blue', marker='o')
                terminate = True
        x_traj = np.array(x_traj)
        u_traj = np.array(u_traj)
        if success[-1] == 1:
            traj_reward = cartpole_reward(x_traj, u_traj[:,np.newaxis], np.array(lqr.Q.cpu()), np.array(lqr.R.cpu()))
            rewards.append(traj_reward)
            if compute_gamma:
                gamma_traj = gamma_forwards(x_traj, z_traj, u_traj, ae, fdyn)
            gamma.append(np.max(gamma_traj))
    if latent_traj and not video:
        for ax in axes:
            plot_latent_trajectories(ae, fdyn, 0.05, 128, T=1000, ax=ax)
    if not video:
        plt.show()
    if len(rewards) > 0:
        print(max(gamma))
        return sum(rewards) / len(rewards), sum(success) / len(success), max(gamma)
    else:
        return 0, sum(success)/ len(success), 0


# plot reconstructed trajectories in projections of the x-space 
# used to evaluate backwards-conjugacy after training
def plot_trajectories(ae, fdyn, X, U, N, steps=10, video=None):
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
    else:
        fdyn.eval()
    with torch.no_grad():
        idx = np.random.permutation(np.arange(X.shape[0]))[:N] 
        for i in idx:
            if params.ode:
                z = ae.encode(X[i][0].unsqueeze(0))
                T_eval = params.DT * torch.arange(steps)
                fdyn_ = lambda t, z: fdyn( torch.cat((z.squeeze(), U[i,int(U.shape[1]*t/(X.shape[1]*params.DT))] ))  )
                z_pred = odeint(fdyn_, z, T_eval, method=params.ode_method).squeeze()
                Xhat = ae.decode(z_pred)
            elif params.enc_true_dyn:
                z = ae.encode(X[i][0].unsqueeze(0))
                Xhat = []
                for t in range(U.shape[1]):
                    z = _flow_diff(fdyn, z, U[i][t].unsqueeze(0), params.DT)[0]
                    Xhat.append(ae.decode(z).squeeze())
                Xhat = np.array(Xhat)
            else:
                z = ae.encode(X[i][0].unsqueeze(0))
                Z = [z.squeeze()]
                for t in range(U.shape[1]):
                    if params.learn_residual or params.linear_state_space_offset:
                        z_tmp = z
                    if params.control_affine:
                        z = fdyn_drift(z) + (fdyn_cntrl(z).unsqueeze(-1) @ U[i, t].unsqueeze(0).unsqueeze(-1))[:,:,0]
                    elif params.linear_state_space:
                        z = (fdyn_drift(z).reshape(params.d_z, params.d_z) @ z.reshape(params.d_z, 1) +\
                             fdyn_cntrl(z).reshape(params.d_z, params.d_u) @ U[i,t].reshape(params.d_u, 1)).squeeze()
                        z = z.unsqueeze(0)
                    else:
                        z = fdyn(torch.hstack( (z, U[i, t].unsqueeze(-1)) ))
                    if params.learn_residual:
                        z = z + z_tmp
                    if params.linear_state_space_offset:
                        z = z + fdyn_offset(z_tmp)
                    Z.append(z.squeeze())
                Xhat = [ae.decode(z.unsqueeze(0)).squeeze().cpu() for z in Z]
                Xhat = np.array(Xhat)

            if video:
                return Xhat
            else:
                plt.title("angle vs angular velocity, ex. {}".format(i))
                if params.system == 'cartpole-gym':
                    plt.plot(X[i][:steps,0], X[i][:steps,1], label="true dynamics")
                    plt.plot(Xhat[:,0], Xhat[:,1], label="evolve in latent then decode")
                elif params.system == 'cartpole-custom':
                    plt.plot(X[i].cpu()[:steps,2], X[i].cpu()[:steps,3], label="true dynamics")
                    plt.plot(Xhat[:,2], Xhat[:,3], label="evolve in latent then decode")
                plt.legend()
                plt.show()
                plt.title("cart position vs cart velocity, ex. {}".format(i))
                if params.system == 'cartpole-gym':
                    plt.plot(X[i][:steps,0], X[i][:steps,1], label="true dynamics")
                    plt.plot(Xhat[:,0], Xhat[:,1], label="evolve in latent then decode")
                elif params.system == 'cartpole-custom':
                    plt.plot(X[i].cpu()[:steps,0], X[i].cpu()[:steps,1], label="true dynamics")
                    plt.plot(Xhat[:,0], Xhat[:,1], label="evolve in latent then decode")
                plt.legend()
                plt.show()
