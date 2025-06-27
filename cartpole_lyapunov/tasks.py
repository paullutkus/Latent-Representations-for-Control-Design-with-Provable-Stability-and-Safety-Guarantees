import time
import params
import torch
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import cartpole
from cartpole import CartpoleRenderer
from integration import _flow



# simulate cartpole under closed-loop latent controller, perturb periodically
# visualize and plot trajectories
def stabilize_cartpole(LQR, ae, T=10, tmax=1000, stabilize=False, eps=0.3, visualize=False):
    if params.system == 'cartpole-gym':
        env = gym.make('InvertedPendulum-v5', render_mode='human', 
                       reset_noise_scale=0.00)
        for i in range(T):
            obs, _ = env.reset()
            terminated = False
            while not terminated:
                z = ae.encode(torch.tensor(np.expand_dims(obs, axis=0)).float())
                u = LQR(z)
                action = u[0].detach() 
                obs, _, terminated, _, _ = env.step(action)
                time.sleep(0.0)
        env.close()
    elif params.system == 'cartpole-custom':
        for i in range(T):
            x = torch.tensor(np.random.uniform(low=[-eps, -eps, -eps, -eps],
                                               high=[eps,  eps,  eps,  eps])).float()
            z = ae.encode(torch.unsqueeze(x, 0))
            x = x.cpu()
            if visualize:
                disp = CartpoleRenderer(x)
            A_err = []
            U = []
            traj = []
            traj.append(x.detach().numpy())
            for t in range(tmax):
                if stabilize:
                    u = LQR(z).item()
                    U.append(u)
                else:
                    u = 0.

                x_prev = x
                x = _flow(x, cartpole.DT, u)[-1]
                traj.append(x)

                if (t < tmax - 1) and (i == 0):
                    x_hat0 = ae.decode(LQR.A @ z[0]).cpu()
                    x0 = _flow(x_prev, cartpole.DT, 0.0)[-1]
                    A_err.append(np.linalg.norm(x_hat0.detach().numpy() - x0))
                else:
                    plt.title("||Dec(A@E(x))-f(x,0)||: locally_linear={}".format(params.linear_state_space))
                    plt.plot(A_err)
                    plt.show()
                    plt.plot(U)
                    plt.show()
                    break

                if (t % 100 == 0) and t != 0:
                    x += np.random.normal(0, 1e-1)

                if visualize:
                    disp.state = x
                    disp.render()

                z = ae.encode(torch.tensor(x.reshape(-1, 4)).float())

            if visualize:
                disp.close()

            traj = np.array(traj)
            plt.plot(traj[:,0], label='x')
            plt.plot(traj[:,1], label='v')
            plt.plot(traj[:,2], label='theta')
            plt.plot(traj[:,3], label='theta-dot')
            plt.legend()
            plt.show()
            
            return traj


