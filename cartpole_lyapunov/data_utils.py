import torch
#from torch.func import grad
from torch.autograd import grad
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import pickle
import os
import params
import cartpole
from cartpole import dxdt_torch
from controls import LQR
from integration import _flow, _flow_diff
from tqdm import tqdm
from tqdm.contrib import tzip
from copy import deepcopy


def make_grid_dataset(x_range, u_range, n_pts):
    axes = []
    for i in range(params.d_x):
        axes.append(torch.linspace(-x_range[i], x_range[i], n_pts))
    for i in range(params.d_u):
        axes.append(torch.linspace(-u_range[i], u_range[i], n_pts))
    pts = torch.meshgrid(axes)
    pts = torch.stack(pts, dim=-1)
    pts = pts.reshape(-1, params.d_x+params.d_u)
    #print("dset shape:", pts.reshape(-1, params.d_x+params.d_u).shape)
    X0 = pts[:,:-1]
    U0 = pts[:,-1].unsqueeze(-1)
    #print(X0.shape)
    #print(U.shape)
    X = []
    U = []
    for x0, u0 in zip(tqdm(X0), U0):
        redo = True
        while redo:
            Xi = [] 
            Ui = []
            x = _flow(x0.cpu(), cartpole.DT, u0.cpu().item())[-1]
            Xi.append(torch.tensor(x).float())
            for t in range(params.traj_len - 2):
                u = 2*params.u_range*(np.random.rand() - 0.5)
                x = _flow(x, cartpole.DT, u)[-1]
                Xi.append(torch.tensor(x).float())
                Ui.append(torch.tensor(u).float())
            Xi = torch.vstack(Xi)
            Ui = torch.vstack(Ui)
            if not( (torch.abs(Xi[:,2]) >= torch.pi/2).any() or (torch.abs(Xi[:,0]) >= 3).any() ):
                redo = False
                
        #print(Xi)
        #print(Ui)
        #X.append(torch.vstack(Xi))
        #U.append(torch.vstack(Ui))
        X.append(Xi)
        U.append(Ui)

    #X = torch.tensor(X).float()
    #print(X[0].shape)
    #print(U[0].shape)
    #X = torch.vstack(X)
    X = torch.stack(X, dim=0)
    U = torch.stack(U, dim=0)
    #print(X.shape)
    #print(X0.shape)
    #print(X1.shape)
    #print(U0)
    #print(U)
    #print(U0)
    #print(U)
    X = torch.cat([X0.unsqueeze(1), X], dim=1)
    U = torch.cat([U0.unsqueeze(1), U], dim=1)
    #U = U.reshape(-1, 1, 1)
    print("X shape", X.shape)
    print("U shape", U.shape)
    X1_drift = []
    for x in tqdm(X.reshape(-1, params.d_x)):
        X1_drift.append(_flow(x.cpu(), cartpole.DT, 0.)[-1])
    X1_drift = torch.tensor(X1_drift).float()
    X_drift = torch.cat([X.reshape(-1, params.d_x).unsqueeze(1), X1_drift.unsqueeze(1)], dim=1)
    print("X_drift shape", X_drift.shape)

    Data_X = (X, X_drift)
    Data_U = (U, None)
    Data_Xtest = (X, X_drift)
    Data_Utest = (U, None)

    data = (Data_X, Data_U, Data_Xtest, Data_Utest)

    path = os.path.abspath('') + '/data/' + params.dataset
    print("saving to: ", path)
    with open(path, 'wb') as file:
        pickle.dump(data, file)
    print("dataset saved")

    return data



def mix_dataset(X, U, ae, fdyn, ep):
    assert params.linear_state_space
    T = X.shape[1]
    N = X.shape[0]

    #print("X shape start", X.shape)
    #print("U shape start", U.shape)

    fdyn_drift, fdyn_cntrl = fdyn
    lqr = LQR(ae, fdyn)
    n_data = int((min(ep, params.mix_end) - params.mix_start) * N / (params.mix_end - params.mix_start))
    #print("n_data", n_data)
    X0 = 2*params.x_range_active*(torch.rand(n_data, params.d_x) - 0.5)
    Z0 = ae.encode(X0)
    U0 = lqr(Z0).T.to("cuda")
    Xact = X0.unsqueeze(1)
    Uact = U0.unsqueeze(1)
    for t in range(T-1):
        Xact = torch.hstack([Xact, _flow_diff(dxdt_torch, Xact[:,-1], Uact[:,-1], cartpole.DT)[-1].unsqueeze(1)])
        if t < T-2:
            Z = ae.encode(Xact[:,-1])
            Uact = torch.hstack([Uact, lqr(Z).T.to("cuda").unsqueeze(1)])

    #print("Xact shape", Xact.shape)
    #print("Uact shape", Uact.shape)
    end_idx = N - n_data
    X = torch.vstack([X[:end_idx], Xact])
    U = torch.vstack([U[:end_idx], Uact])
    #print("X shape end", X.shape)
    #print("U shape end", U.shape)
    return X, U



def fit_dset_to_m(x_data, u_data):
    #params.learn_drift = False
    #params.m_schedule=15*[0]
    #params.m=15
    print("before:")
    print(x_data.shape)
    print(u_data.shape)
    x_data_windows = []
    u_data_windows = []
    for i in range(x_data.shape[1] // (params.m+1)):
        begin_x = int(i*(params.m+1))
        end_x = begin_x+params.m+1
        begin_u = int(i*params.m)
        end_u = begin_u + params.m 
        x_data_windows.append(x_data[:,begin_x:end_x])
        u_data_windows.append(u_data[:,begin_u:end_u])
    x_data_windows.append(x_data[:,-(params.m+1):])
    u_data_windows.append(u_data[:,-params.m:])
    x_data_fit = torch.vstack(x_data_windows)
    u_data_fit = torch.vstack(u_data_windows)
    print("after:")
    print(x_data_fit.shape)
    print(u_data_fit.shape)
    #X_lqr = [torch.tensor(x_data_lqr).float(), torch.tensor(x_data_lqr).float()]
    #U_lqr = [torch.tensor(u_data_lqr).float(), None]
    #for X_lqr_i in X_lqr:
    #    X_lqr_i.requires_grad = True
    #U_lqr[0].requires_grad = True
    #Xtest_lqr = X_lqr
    #Utest_lqr = U_lqr
    #for a in Xtest_lqr:
    #    print(a.requires_grad)
    #print(Utest_lqr[0].requires_grad)
    return x_data_fit, u_data_fit


def adversarial_training(X, U, ae, fdyn):
    assert params.linear_state_space or params.learn_drift

    d_x = params.d_x
    d_z = params.d_z
    d_u = params.d_u
    fdyn_drift, fdyn_cntrl = fdyn
    X, X_drift = X
    U, _ = U

    T_drift = X_drift.shape[1]
    T = X.shape[1]
    m = U.shape[1]
 
    eps_x = 1e-1
    eps_u = 5e-1
    delta_x = 1.
    delta_u = 5.

    def make_X(X1, U, Xref=None, integrator='diff'):
        Xnew = X1.unsqueeze(1)
        for t in range(m):
            if integrator == 'diff':
                Xt = _flow_diff(dxdt_torch, Xnew[:,t], U[:,t], cartpole.DT)[-1]
            elif integrator == 'scipy':
                X0 = Xnew[0,t].detach().numpy().reshape(d_x)
                U0 = U[0,t].detach().numpy().reshape(d_u)[0]
                Xt = torch.tensor(_flow(X0, cartpole.DT, U0)[-1]).reshape(1, d_x).float()
                for (Xnew_i, U_i) in zip(Xnew[1:,t], U[1:,t]):
                    Xnew_i = Xnew_i.detach().numpy()
                    U_i = U_i.detach().numpy()[0]
                    Xt_i = torch.tensor(_flow(Xnew_i, cartpole.DT, U_i)[-1]).float()
                    Xt = torch.cat([Xt, Xt_i.unsqueeze(0)], dim=0)
            Xnew = torch.cat([Xnew, Xt.unsqueeze(1)], dim=1)
            if Xref is not None:
                print(torch.linalg.norm(Xnew[:,t+1] - Xref[:,t+1])) # t+1 or t??
        return Xnew

    def L(X1, U, integrator='diff'):
        X = make_X(X1, U, integrator=integrator)
        Z = ae.encode(X[:,:-1].reshape(-1, d_x))
        ### CHECK THAT THESE SHAPES ARE CORRECT !!! ###
        print(ae.encode(X[:,1:].reshape(-1, d_x)).shape)
        print((fdyn_drift(Z).reshape(-1, d_z, d_z) @ Z.unsqueeze(-1) +\
               fdyn_cntrl(Z).reshape(-1, d_z, d_u) @ U.reshape(-1, d_u, 1)).squeeze().shape)
        l = torch.linalg.norm(ae.encode(X[:,1:].reshape(-1, d_x)) -\
                              (fdyn_drift(Z).reshape(-1, d_z, d_z) @ Z.unsqueeze(-1) +\
                               fdyn_cntrl(Z).reshape(-1, d_z, d_u) @ U.reshape(-1, d_u, 1)).squeeze(), dim=1
                             )
        return torch.mean(l)

    X1 = X[:,0]
    X1_pert = X1
    U_pert = U
    X1_pert.requires_grad = True
    U_pert.requires_grad = True
    for t in range(10):
        ### BACKPROP HERE ONLY W.R.T TO THE FIRST COLUMN
        #make_X(X1, U, Xref=X, integrator='scipy')
        #DX1 = grad(L, argnums=0)(X1_pert, U_pert)
        DX1 = grad(L(X1_pert, U_pert), X1_pert)[0]
        DU = grad(L(X1_pert, U_pert), U_pert)[0]
        #print(DX)
        #print(DU)
        #Xpert_prev = X_pert
        #Upert_prev = U_pert

        ## WRONG: Xpert should be calculated by updating the initial conditions 
        # and then recomputing all the trajectories 
        X1_pert = X1_pert + eps_x*DX1
        U_pert = U_pert + eps_u*DU
        ## THIS IS WRONG EVEN FOR THE T=2 CASE: you're changing the question and the answer !!!

        #print("Loss:", L(X1_pert, U_pert, integrator='scipy'))
        if torch.linalg.norm(X1_pert - X1) <= delta_x:
            res = (X1_pert - X1)
            res = delta_x * res /torch.linalg.norm(res)
            X1_pert = X1 + res
        if torch.linalg.norm(U_pert - U) <= delta_u:
            res = (U_pert - U)
            res = delta_u * res /torch.linalg.norm(res)
            U_pert = U + res
        ##  ^^ THESE SHOULD ONLY BE COMPUTED W.R.T. THE FIRST COLUMN (INITIAL CONDITOIN

        #if torch.linalg.norm(X - Xpert) and torch.linalg.norm(U - Upert) <= 1e-5:
        #    break

    '''
    Xpert_drift = X_drift[:,:-1]
    for t in range(10):
        #print(Xpert_drift[:,:-1].shape)
        #print(torch.zeros_like(X_drift[:,:-1,:1]).shape)
        DX_drift = grad(L, argnums=0)(Xpert_drift, torch.zeros_like(Xpert_drift[:,:,:1]))
        Xpert_prev_drift = Xpert_drift
        Xpert_drift = Xpert_drift + eps_x*DX_drift
        #if torch.linalg.norm(Xpert_drift - X_drift) <= delta_x:
        #    res = (Xpert_drift - X_drift)
        #    res = delta_x * res / torch.linalg.norm(res)
        #    Xpert_drift = X_drift + res
        print("Loss drift:", L(Xpert_drift, torch.zeros_like(Xpert_drift[:,:,:1])))
        #if torch.linalg.norm(Xpert_prev_drift - Xpert_drift) <= 1e-5:
        #    break

        
        #print(delta_x / torch.linalg.norm(Xpert))
        #print(delta_u / torch.linalg.norm(Upert))
    '''
    X_pert = make_X(X1_pert, U_pert, integrator='scipy')
    #X_pert = torch.cat((Xpert.reshape(-1, T-1, d_x), X[:, -1].reshape(-1, 1, d_x)), dim=1)
    #Xpert_drift = Xpert_drift.reshape(-1, T_drift, d_x)
    #X = [Xpert, Xpert_drift]
    X = [X_pert, X_drift]
    U = [U_pert, None]
    return X, U


def get_minibatches(X, U=None):

    # If replacement, no need to remove examples from subsequent minibatches
    if params.with_replacement:
        N = X.shape[0] // params.batch_size
        X_batches = []
        if U is not None:
            U_batches = []
        for i in range(N):
            rand_idx = np.random.permutation(np.arange(X.shape[0]))
            batch_idx = rand_idx[:params.batch_size]
            X_batches.append(X[batch_idx])
            if U is not None:
                U_batches.append(U[batch_idx])
        if U is None:
            U_batches = len(X_batches) * [None]
        return X_batches, U_batches

    # Whole dataset is less than one batch:
    if params.batch_size >= X.shape[0]:
        if U is not None:
            return [X], [U]
        else:
            return [X], len(X) * [None]

    # At least one batch:
    X_batches = []
    if U is not None:
        U_batches = []
    rem_idx = np.arange(X.shape[0])
    rand_idx = np.random.permutation(rem_idx)

    # Less than two batches:
    if len(rem_idx) // params.batch_size < 1:
        batch_idx = rand_idx[:params.batch_size]
        X_batches.append(X[batch_idx])
        if U is not None:
            U_batches.append(U[batch_idx])

        batch_idx = rand_idx[params.batch_size:]
        X_batches.append(X[batch_idx])
        if U is not None:
            U_batches.append(U[batch_idx])
        
        if U is None:
            U_batches = len(X_batches) * [None]
        return X_batches, U_batches

    # At least two batches:
    batch_idx = rand_idx[:params.batch_size]
    X_batches.append(X[batch_idx])
    if U is not None:
        U_batches.append(U[batch_idx])
    rem_idx = rand_idx[params.batch_size:]
    while True:
        batch_idx = rem_idx[:params.batch_size]
        X_batches.append(X[batch_idx])
        if U is not None:
            U_batches.append(U[batch_idx])
        if len(rem_idx) // params.batch_size < 2:
            batch_idx = rem_idx[params.batch_size:]
            X_batches.append(X[batch_idx])
            if U is not None:
                U_batches.append(U[batch_idx])
            break
        else:
            rem_idx = rem_idx[params.batch_size:]
    if U is None:
        U_batches = len(X_batches) * [None]
    return X_batches, U_batches


def load_dataset(fname='data.pkl'):
    path = os.path.abspath('') + '/data/' + fname
    with open(path, 'rb') as file:
        data = pickle.load(file)

    X, U, Xtest, Utest = data

    include_drift = False
    if type(X) is tuple:
        include_drift = True

    if include_drift:
        X, X_drift = X
        Xtest, Xtest_drift = Xtest
        #X_drift = torch.tensor(X_drift[:params.dset_size]).float()
        X_drift = torch.tensor(X_drift).float()
        #Xtest_drift = torch.tensor(Xtest_drift[:params.dset_size]).float()
        Xtest_drift = torch.tensor(Xtest_drift).float()
        U = U[0] 
        Utest = Utest[0]

    #X = torch.tensor(X[:params.dset_size]).float()
    X = torch.tensor(X).float()
    #U = torch.tensor(U[:params.dset_size]).float()
    U = torch.tensor(U).float()
    #Xtest = torch.tensor(Xtest[:params.dset_size]).float()
    Xtest = torch.tensor(Xtest).float()
    #Utest = torch.tensor(Utest[:params.dset_size]).float()
    Utest = torch.tensor(Utest).float()

    if include_drift:
        X = (X, X_drift)
        Xtest = (Xtest, Xtest_drift)
        U = (U, None)
        Utest= (Utest, None)

    data = (X, U, Xtest, Utest)
    return data


def sample_drift(eps, N, S):
    if params.system == 'cartpole-gym':
        X = []
        env = gym.make('InvertedPendulum-v5', render_mode=None,
                       reset_noise_scale=0.0)
        obs, _ = env.reset()
        Xi = [obs]
        action = np.array([0.])
        obs, _, _, _, _ = env.step(action)
        Xi.append(obs)
        X.append(Xi)
        env.close()
        print("## SAMPLING DRIFT ##")
        env = gym.make('InvertedPendulum-v5', render_mode=None,
                       reset_noise_scale=eps)
        for i in tqdm(range(N+S)):
            obs, _ = env.reset()
            Xi = [obs]
            action = np.array([0.])
            obs, _, _, _, _ = env.step(action)
            Xi.append(obs)
            X.append(Xi)
        env.close()
        X = np.array(X)
        Xtest = X[N+1:]
        Xtest = np.concatenate([np.expand_dims(X[0],0), Xtest], axis=0)
        X = X[:N+1]
        return (X, Xtest)

    elif params.system == 'cartpole-custom':
        X = []
        #env = gym.make('InvertedPendulum-v5', render_mode=None,
        #               reset_noise_scale=0.0)
        #obs, _ = env.reset()
        x = np.array([0., 0., 0., 0,])
        Xi = [x]
        u = np.array(0.)
        x = _flow(x, cartpole.DT, u)[-1]
        Xi.append(x)
        X.append(Xi)
        print("## SAMPLING DRIFT ##")
        #env = gym.make('InvertedPendulum-v5', render_mode=None,
        #               reset_noise_scale=eps)
        for i in tqdm(range(N+S)):
            #obs, _ = env.reset()
            x = np.random.uniform(low=[-eps, -eps, -eps, -eps],
                                  high=[eps, eps, eps, eps])
            Xi = [x]
            u = np.array(0.)
            x = _flow(x, cartpole.DT, u)[-1]
            Xi.append(x)
            X.append(Xi)
        X = np.array(X)
        Xtest = X[N+1:]
        Xtest = np.concatenate([np.expand_dims(X[0],0), Xtest], axis=0)
        X = X[:N+1]

        return (X, Xtest)



def make_dataset(save=True, fname='data.pkl', render_mode='human', expert_controller=False):

    include_drift = params.linear_state_space or params.learn_drift

    N = params.num_traj
    T = params.traj_len
    S = params.num_test_traj

    X = []
    U = []


    # If learning Ax + Bu
    if include_drift and (params.system == 'cartpole-gym'):
        X_drift, Xtest_drift = sample_drift(1e-3, N, S)
        print("X_drift shape", X_drift.shape)
        print("Xtest_drift shape", Xtest_drift.shape)

    # If using expert controller
    if expert_controller:
        lqr = LQR(None, None, original_system=True)

    ##########################
    ##  begin: OPEN-AI GYM  ##
    ##########################
    if params.system == 'cartpole-gym':
        env = gym.make('InvertedPendulum-v5', render_mode=render_mode,
                       reset_noise_scale=0.0)

        for i in tqdm(range(N+S)):
            obs, _ = env.reset()
            terminated = False
            truncated = False
            Xi = [obs]
            Ui = []
            while len(Xi) < T:
                if terminated or truncated:
                    obs, _ = env.reset()
                    Xi = [obs]
                    Ui = []
                action = env.action_space.sample()
                Ui.append(action)
                obs, _, terminated, truncated, _ = env.step(action)
                Xi.append(obs)
            Xi = np.array(Xi)
            Ui = np.array(Ui)
            X.append(Xi)
            U.append(Ui)
        env.close()
    ########################
    ##  end: OPEN-AI GYM  ##
    ########################

    ##############################
    ##  begin: CUSTOM CARTPOLE  ##
    ##############################
    elif params.system == 'cartpole-custom':
        
        '''
        if params.traj_len == 2:
            x = np.array([0., 0., 0., 0.])
            u = np.array(0.)
            Xi = [x]
            Ui = [np.expand_dims(u, -1)]
            x = _flow(x, cartpole.DT, u)[-1]
            Xi.append(x)
            X.append(Xi)
            U.append(Ui)
        '''

        if include_drift:
            X_drift = []
        for i in tqdm(range(N+S)):
            terminated = True
            while terminated:
                if expert_controller:
                    x = np.random.uniform(low=4*[-params.init_pert],
                                          high=4*[params.init_pert])
                else:
                    x = np.random.uniform(low=cartpole.X_BOX[0],
                                          high=cartpole.X_BOX[1])
                if expert_controller:
                    u = lqr(torch.tensor(x).float()).item()
                    #u.requires_grad = True
                else:
                    u = np.random.uniform(low=cartpole.U_BOX[0],
                                          high=cartpole.U_BOX[1])
                Tend = cartpole.DT
                Xi = [x]
                Ui = []
                terminated = False
                for t in range(T-1):
                    if include_drift:
                        Xi_drift = [x]
                        u_drift = np.array(0.)
                        #print(x.shape)
                        #print(u_drift.shape)
                        x_drift = _flow(x, cartpole.DT, u_drift)[-1]
                        Xi_drift.append(x_drift)
                        X_drift.append(Xi_drift)

                    x = _flow(x, cartpole.DT, u)[-1]

                    #x = f(x,u)
                    if (((x[2] >= 0.2).any() or (x[2] <= -0.2).any()) and not expert_controller) or\
                       ((x[2] >= np.pi/2 or x[2] <= -np.pi/2) and expert_controller):
                        #print("terminated")
                        terminated=True
                    Xi.append(x)
                    Ui.append(np.expand_dims(u, -1))
                    if expert_controller:
                        u = lqr(torch.tensor(x).float()).item()
                        #u.requires_grad = True
                    else:
                        u = np.random.uniform(low=cartpole.U_BOX[0],
                                              high=cartpole.U_BOX[1])
 
            '''
            plt.plot(np.array(Xi)[:,2], np.array(Xi)[:,3], color='b')
            plt.plot(np.array(Xi)[:,0], np.array(Xi)[:,1], color='r')
            plt.plot(np.array(Xi)[-1,2], np.array(Xi)[-1,3], 'b*')
            plt.plot(np.array(Xi)[-1,0], np.array(Xi)[-1,1], 'r*')
            print(np.array(Xi).shape) 
            plt.show()
            '''

            X.append(Xi)
            U.append(Ui)

        if include_drift:
            X_drift = np.array(X_drift)
            Xtest_drift = X_drift[int(T*N):]
            X_drift = X_drift[:int(T*N):]
        #X = np.array(X)
        #U = np.array(U)
        '''

        Xtest = X[N:]
        Utest = U[N:]
        Xtest = np.stack([np.expand_dims(X[0], 0), Xtest], axis=0)
        Utest = np.stack([np.expand_dims(U[0], 0), Utest], axis=0)
        print("Xtest shape", Xtest.shape)
        print("Utest shape", Utest.shape)

        X = X[:N]
        U = U[:N]
        print("X shape", X.shape)
        print("U shape", U.shape)

        data = (X, U, Xtest, Utest)
        if save:
            path = os.path.abspath('') + '/data/' + fname
            print("saving to: ", path)
            with open(path, 'wb') as file:
                pickle.dump(data, file)
            print("dataset saved")

        X = torch.tensor(X[:params.dset_size]).float()
        U = torch.tensor(U[:params.dset_size]).float()
        Xtest = torch.tensor(Xtest[:params.dset_size]).float()
        Utest = torch.tensor(Utest[:params.dset_size]).float()
        data = (X, U, Xtest, Utest)
        return data
        '''

    ############################
    ##  end: CUSTOM CARTPOLE  ##
    ############################

    X = np.array(X)
    U = np.array(U)

    Xtest = X[N:]
    Utest = U[N:]
    X = X[:N]
    U = U[:N]

    print("X shape:", X.shape)
    print("U shape:", U.shape)
    print("Xtest shape:", Xtest.shape)
    print("Utest shape:", Utest.shape)

    if include_drift:
        X = (X, X_drift)
        U = (U, None)
        Xtest = (Xtest, Xtest_drift)
        Utest = (Utest, None)

    data = (X, U, Xtest, Utest)
    if save:
        path = os.path.abspath('') + '/data/' + fname
        print("saving to: ", path)
        with open(path, 'wb') as file:
            pickle.dump(data, file)
        print("dataset saved")

    # Unpack before converting to float tensor
    if include_drift:
        X_drift = X[1]
        X = X[0]
        Xtest_drift = Xtest[1]
        Xtest = Xtest[0]
        U = U[0]
        Utest = Utest[0]

    X = torch.tensor(X[:params.dset_size]).float()
    if include_drift:
        X_drift = torch.tensor(X_drift[:params.dset_size]).float()
        Xtest_drift = torch.tensor(Xtest_drift[:params.dset_size]).float()
    U = torch.tensor(U[:params.dset_size]).float()
    Xtest = torch.tensor(Xtest[:params.dset_size]).float()
    Utest = torch.tensor(Utest[:params.dset_size]).float()

    # Repack into tuples to pass to training
    if include_drift:
        X = (X, X_drift)
        Xtest = (Xtest, Xtest_drift)
        U = (U, None)
        Utest = (Utest, None)

    data = (X, U, Xtest, Utest)
    return data


