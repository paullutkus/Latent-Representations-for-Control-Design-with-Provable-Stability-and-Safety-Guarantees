import time
import params
import torch
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import cartpole
from cartpole import CartpoleRenderer
from integration import _flow


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
            #eps = 1e-3 # 1e-2
            x = torch.tensor(np.random.uniform(low=[-eps, -eps, -eps, -eps],
                                               high=[eps,  eps,  eps,  eps])).float()

            #x = torch.tensor([eps, -eps, -eps, eps])
            x = torch.tensor([0.1, 0.40, -0.1, -0.20])
            #if torch.sign(x[0]) != torch.sign(x[3]):
            #    x[2] *= -1
            #if torch.sign(x[2]) != torch.sign(x[1]):
            #    x[1] *= -1
            #if torch.sign(x[0]) == torch.sign(x[1]):
            #    x[1] *= -1
            #if torch.sign(x[2]) == torch.sign(x[3]):
            #    x[3] *= -1

            z = ae.encode(torch.unsqueeze(x, 0))
            x = x.cpu()
            if visualize:
                disp = CartpoleRenderer(x)
            A_err = []
            U = []
            traj = []
            traj.append(x.detach().numpy())
            for t in range(tmax):
                #print("Made it into sim")
                if stabilize:
                    u = LQR(z).item()
                    U.append(u)
                else:
                    u = 0.
                #print("LQR computed")
                x_prev = x
                x = _flow(x, cartpole.DT, u)[-1]# + 2e-4*(np.random.rand() - 0.5)
                traj.append(x)
                if (t < tmax - 1) and (i == 0):
                    #print("computing A error")
                    '''
                    print(LQR.A.shape)
                    print(z[0].shape)
                    print((LQR.A @ z[0]).shape)
                    print("\n")
                    print(LQR.B.shape)
                    print(LQR(z).shape)
                    print((LQR.B @ LQR(z)).shape)
                    '''
                    x_hat0 = ae.decode(LQR.A @ z[0]).cpu()# + (LQR.B @ LQR(z))[:,0])
                    x0 = _flow(x_prev, cartpole.DT, 0.0)[-1]
                    A_err.append(np.linalg.norm(x_hat0.detach().numpy() - x0))
                else:
                    plt.title("||Dec(A@E(x))-f(x,0)||: locally_linear={}".format(params.linear_state_space))
                    plt.plot(A_err)
                    plt.show()
                    plt.plot(U)
                    plt.show()
                    break

                #if (t % 100 == 0) and t != 0:
                #    x += np.random.normal(0, 1e-1)
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




