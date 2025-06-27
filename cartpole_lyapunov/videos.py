import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import params
from plotting import plot_stability
from cartpole import dxdt_torch
from controls import LQR
from integration import _flow



# video of x-space trajectories under closed-loop latent controller
# projected into the latent space
def latent_projections_video(lqr, ae, fdyn, r_x, N=4, T=100, fname='latent_projections.mp4'):
    fdyn_drift, fdyn_cntrl = fdyn
    fig = plt.figure(figsize=(13, 8))
    Z = []
    Zl = []
    for i in range(N):
        x = 2*r_x*(np.random.rand(4) - 0.5)
        print("initial x:", x)
        z = ae.encode(torch.tensor(x).float())
        zl = z
        print("initial z:", z.detach().numpy())
        Zi = [z.detach().numpy()]
        Zli = [zl.detach().numpy()]
        for t in range(T):
            ux = lqr(z).item()
            uz = lqr(zl).item()
            x = _flow(x, params.DT, ux)[-1]
            z = ae.encode(torch.tensor(x).float())
            zl = (fdyn_drift(zl).reshape(params.d_z, params.d_z) @ zl.reshape(params.d_z, 1) +\
                  fdyn_cntrl(zl).reshape(params.d_z, params.d_u) @ torch.tensor(uz).float().reshape(1, 1)).squeeze()
            Zi.append(z.detach().numpy())
            Zli.append(zl.detach().numpy())
        Z.append(np.array(Zi))
        Zl.append(np.array(Zli))
    Z = np.array(Z)
    Zl = np.array(Zl)
    print(Z.shape)
    print(Zl.shape)

    def render_frame(i):
        plt.cla()
        plt.title("Projected trajectories of original system (dashed -- latent only, solid -- projected")
        for (Zi, Zli) in zip(Z, Zl):
            plt.plot(Zi[:i,0], Zi[:i,1])
            plt.plot(Zli[:i,0], Zli[:i,1], linestyle="--")
            plt.legend()

    render_frame(0)
    animation = anim.FuncAnimation(fig, render_frame, T, interval=100)
    animation.save(fname, writer='ffmpeg', fps=16)


# video of forward conjugacy over training
def encoder_diagram_video(ae_list, fdyn_list, n_pts=500, eps=np.pi/3, T=500,
                          tol=np.pi/2, fname='enc_anim.mp4'):
    fig = plt.figure(figsize=(13, 8))
    low = params.d_x*[-eps]
    high = params.d_x*[eps]
    frame_list = []
    def render_frame(i):
        plt.cla()
        plt.title(str(i))
        ae = ae_list[i]
        fdyn_drift, fdyn_cntrl = fdyn_list[i]
        lqr = LQR(ae_list[i], fdyn_list[i])
        X = np.random.uniform(low, high, size=(n_pts, params.d_x))
        X = torch.tensor(X).float()
        Ef = ae_list[i].encode(dxdt_torch(X, lqr(ae_list[i].encode(X))))
        Z = ae.encode(X)
        U = lqr(Z).T
        fE =(fdyn_drift(Z).reshape(-1, params.d_z, params.d_z) @ Z.unsqueeze(-1) +\
             fdyn_cntrl(Z).reshape(-1, params.d_z, params.d_u) @ U.unsqueeze(-1)).squeeze()
        l = torch.mean(torch.linalg.norm(Ef - fE, dim=1)).detach().numpy()
        frame_list.append(l)
        plt.plot(frame_list)
    render_frame(0)
    animation = anim.FuncAnimation(fig, render_frame, len(ae_list), interval=100)
    animation.save(fname, writer='ffmpeg', fps=24)


# video of stability of initial conditions projected into latent space
# over the course of training
def latent_space_video(ae_list, fdyn_list, n_pts=500, eps=np.pi/3, T=500,
                       tol=np.pi/2, fname='z_anim.gif'):
    fig = plt.figure(figsize=(13, 8))
    low = 4*[-eps]
    high = 4*[eps]
    rewards = []
    completion_rates = []
    gammas = []
    def render_frame(i):
        plt.cla()
        avg_reward, completion_rate, gamma_max = plot_stability(ae_list[i], fdyn_list[i], n_pts, low, high, tol, T, visualize=False, 
                                                                latent_traj=True, video=True, frame=i, compute_gamma=True)
        rewards.append(avg_reward)
        completion_rates.append(completion_rate)
        gammas.append(gamma_max)
    #render_frame(0)
    animation = anim.FuncAnimation(fig, render_frame, len(ae_list), interval=100)
    animation.save(fname, writer='ffmpeg', fps=24)
    return rewards, completion_rates, gammas


# video of backwards conjugacy (reconstructions of x-space trajectories)
# over the course of training
def training_example_video(frames, X, fname='train_ex_anim.mp4'):
    fig = plt.figure(figsize=(13, 8))
    def render_frame(i):
        plt.cla()
        plt.title("cart position (x) / pole angle (theta) vs cart velocity (v) / pole angular velocity (w)")
        plt.plot(X[0][params.video_idx][:,2], X[0][params.video_idx][:,3], label="(theta, w) true trajectory")
        plt.plot(frames[i][:,2], frames[i][:,3], label="(theta, w) decoded latent trajectory")
        plt.plot(X[0][params.video_idx][:,0], X[0][params.video_idx][:,1], label="(x, v) true trajectory")
        plt.plot(frames[i][:,0], frames[i][:,1], label="(x, v) decoded latent trajectory")
        plt.legend()
    render_frame(0)
    animation = anim.FuncAnimation(fig, render_frame, len(frames), interval=100)
    animation.save(fname, writer='ffmpeg', fps=24)


