import numpy as np
import torch
import torch.nn as nn
from torch.func import jacrev, vmap
import params
from data_utils import get_minibatches, adversarial_training
from losses import total_loss
from models import FF, AE, VAE
from tqdm import tqdm
from cartpole import dxdt_torch, _dxdt_torch
from controls import get_LQR_params
from plotting import plot_trajectories, plot_stability
from videos import latent_space_video, training_example_video
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from IPython.display import HTML
from copy import deepcopy



# for tracking gradients through projected dynamics
# (unused in final version)
class EncodedDynamics(nn.Module):
    def __init__(self, dynamics, ae):
        super().__init__()
        self.dynamics = dynamics
        self.dynamics.requires_grad = False
        self.ae = ae

    def forward(self, z, u):
        dzdx = vmap(jacrev(self.ae.encode))(self.ae.decode(z))
        return (dzdx @ self.dynamics(self.ae.decode(z), u).unsqueeze(-1)).squeeze()



# computes latent dynamics as composition of decoder with true dynamics
# (unused in final version)
class ReverseAEDynamics(nn.Module):
    def __init__(self, dynamics, ae):
        super().__init__()
        self.dynamics = dynamics
        self.dynamics.requires_grad = False
        self.ae = ae
    
    def forward(self, zu):
        if len(zu.shape) == 2:
            z = zu[:,:params.d_z]
            u = zu[:,params.d_z:]
            return self.ae.encode(self.dynamics(self.ae.decode(z), u))
        elif len(zu.shape) == 1:
            z = zu[:params.d_z].unsqueeze(0)
            u = zu[params.d_z:].unsqueeze(0)
            return self.ae.encode(self.dynamics(self.ae.decode(z), u)).squeeze()



# main training function to be called when running experiments
def train(X, U, Xtest, Utest, ae_0=None, fdyn_0=None):
    if ae_0 is not None:
        ae, ae_opt = ae_0
    else:
        if params.vae:
            ae = VAE(params.enc_layers, params.dec_layers,
                     residual=params.residual_ae, res_bias=params.res_bias_ae,
                     init=params.init_ae, init_bias=params.init_ae_bias)
        elif params.neural_dynamics:
            # only ae is set up for identity encoder/decoder
            assert not params.vae
            ae = AE(params.enc_layers, params.dec_layers,
                    residual=params.residual_ae, res_bias=params.res_bias_ae,
                    init=params.init_ae, init_bias=params.init_ae_bias,
                    iden=True) # iden=True is the only thing that matters here
        else:
            ae = AE(params.enc_layers, params.dec_layers,
                    residual=params.residual_ae, res_bias=params.res_bias_ae,
                    init=params.init_ae, init_bias=params.init_ae_bias)
        ae_opt = params.ae_opt(ae.parameters(), lr=params.lr)
 
    if fdyn_0 is not None:
        if params.control_affine or params.linear_state_space:
            fdyn, fdyn_opt = fdyn_0
            fdyn_drift, fdyn_cntrl = fdyn
            fdyn_drift_opt, fdyn_cntrl_opt = fdyn_opt
        else:
            fdyn , fdyn_opt = fdyn_0
    elif params.enc_true_dyn:
        fdyn = EncodedDynamics(dxdt_torch, ae)
        fdyn_opt = None
    # Needs to come before control_affine/linear_state_space to exclude them
    elif params.reverse_ae:
        assert not params.linear_state_space and not params.control_affine
        fdyn = ReverseAEDynamics(dxdt_torch, ae)
        fdyn_opt = None
    elif params.control_affine or params.linear_state_space:
        print("LATENT DRIFT")
        fdyn_drift = FF(params.fdyn_drift_layers,
                        residual=params.residual_fdyn_drift, res_bias=params.res_bias_fdyn,
                        init=params.init_fdyn_drift, init_bias=params.init_fdyn_drift_bias)
        print("LATENT CONTROL")
        fdyn_cntrl = FF(params.fdyn_cntrl_layers,
                        residual=params.residual_fdyn_cntrl, res_bias=params.res_bias_fdyn,
                        init=params.init_fdyn_cntrl, init_bias=params.init_fdyn_cntrl_bias)
        fdyn_drift_opt = params.fdyn_drift_opt(fdyn_drift.parameters(), lr=params.lr)
        fdyn_cntrl_opt = params.fdyn_cntrl_opt(fdyn_cntrl.parameters(), lr=params.lr)
        # If linear_state_space_offset=True, drift looks like A(z)^{-1}(z+off(z)):
        if params.linear_state_space_offset:
            print("LATENT OFFSET")
            fdyn_offset = FF(params.fdyn_offset_layers,
                             residual=params.residual_fdyn_offset, res_bias=params.res_bias_fdyn,
                             init=params.init_fdyn_offset, init_bias=params.init_fdyn_offset_bias)
            fdyn_offset_opt = params.fdyn_offset_opt(fdyn_offset.parameters(), lr=params.lr)
            fdyn_drift = (fdyn_drift, fdyn_offset)
            fdyn_drift_opt = (fdyn_drift_opt, fdyn_offset_opt)
        fdyn = (fdyn_drift, fdyn_cntrl)
        fdyn_opt = (fdyn_drift_opt, fdyn_cntrl_opt)

    else:
        print("LATENT DYNAMICS")
        fdyn = FF(params.fdyn_layers,
                  residual=params.residual_fdyn, res_bias=params.res_bias_fdyn,
                  init=params.init_fdyn, init_bias=params.init_fdyn_bias)
        fdyn_opt = params.fdyn_opt(fdyn.parameters(), lr=params.lr)

    print(ae)
    print(fdyn)
    ae.to("cuda")
    for f in fdyn:
        f.to("cuda")

    # Init lists for plotting
    losses = []
    test_losses = []
    losses_rec = []
    losses_mstep = []
    losses_jac = []
    losses_all = []
    frames = []
    A_norm = []
    B_norm = []
    A_err = []
    B_err = []
    eigenvalues = []

    if params.rec_batch_reduc == torch.mean:
        vid_loss_cutoff = 1.
        hrzn_loss_cutoff = 1.
    elif params.rec_batch_reduc == torch.sum:
        vid_loss_cutoff = 20.
        hrzn_loss_cutoff = 15.
    ae_list = []
    fdyn_list = []

    # For horizon scheduling
    if params.m_schedule is not None:
        m = 1
        print("horizon is now 1")

    # Active learning
    params.currently_active_learning = False
    rewards = []
    completion_rates = []
    gammas = []

    #############################################
    # PUT TQDM HERE ON A SWITCH WITH THE OTHERS #
    #############################################
    if params.tqdm_each_epoch:
        outer_bar = range(params.epochs)
    else:
        outer_bar = tqdm(range(params.epochs))
    for it in outer_bar:
        
        ###################
        ### BEGIN EPOCH ###
        ###################
        

        # For making video of latent space
        if (it % params.store_model_freq == 0) and (it > 0) and (losses_mstep[-1] <= vid_loss_cutoff): # sum -- 20, mean -- 1
            ae_list.append(deepcopy(ae))
            fdyn_list.append(deepcopy(fdyn))

            eps = np.pi/3
            low = 4*[-eps]
            high = 4*[eps]
            n_pts=100
            avg_reward, completion_rate, gamma_max = plot_stability(ae, fdyn, n_pts, low, high, tol=np.pi/2, T=250, visualize=False, 
                                                                    latent_traj=True, video=True, frame=None, compute_gamma=True,
                                                                    disable_plot=True)
            rewards.append(avg_reward)
            completion_rates.append(completion_rate)
            gammas.append(gamma_max)

            print("it: {}; ".format(it) + "avg reward: {}; ".format(avg_reward) + "completion rate: {}; ".format(completion_rate) + "gamma_max: {}".format(gamma_max))
            #print("completion rate:", completion_rate)
            #print("gamma_max:", gamma_max)

            if (completion_rate >= 0.25) and params.active_learning :
                if params.currently_active_learning == False:
                    params.mix_start = it
                    params.mix_end = params.mix_start + params.mix_len
                params.currently_active_learning = True
                print("Dataset mixing started")

        #params.currently_active_learning = False
        #if params.active_learning and (it > 0) and (losses_mstep[-1] <= hrzn_loss_cutoff) and (m == params.m):
        #    params.currently_active_learning = True

        ### Schedule horizon for multistep prediction loss
        # m = min([params.m, int(it / params.m_schedule) + 1])
        if params.m_schedule is not None:
            #if it >= 1:
            #print(params.m_schedule[m-1], it)
            #print(losses_mstep[-1])
            if (m < len(params.m_schedule)) and (it >= 1) and\
               (losses_mstep[-1] <= hrzn_loss_cutoff) and (params.m_schedule[m - 1] <= it): # sum -- 15, mean -- 1
                m += 1
                print("horizon is now {}".format(m))
        else:
            m = params.m

        # Evaluate on testset ever $test_inc iterations
        if it % params.test_inc == 0:
            ae.eval()
            if params.control_affine or params.linear_state_space:
                if params.linear_state_space_offset:
                    for f in fdyn_drift:
                        f.eval()
                else:
                    fdyn_drift.eval()
                    fdyn_cntrl.eval()
            elif fdyn_opt is not None:
                fdyn.eval()
            with torch.no_grad():
                # For X, U (U may be `None`, for stuff like drift-loss dataset)
                # of each dataset, get minibatches
                all_dsetX_minibatches = []
                all_dsetU_minibatches = []
                for dsetj_X, dsetj_U in zip(Xtest, Utest):
                    dsetj_X_minibatches, dsetj_U_minibatches = get_minibatches(dsetj_X, dsetj_U)
                    all_dsetX_minibatches.append(dsetj_X_minibatches)
                    all_dsetU_minibatches.append(dsetj_U_minibatches)

                # Group the jth minibatches of each X dataset
                Xtest_minibatches = []
                for jth_minibatches in zip(*all_dsetX_minibatches):
                    X_minibatches_j = []
                    for batch in jth_minibatches:
                        X_minibatches_j.append(batch)
                    Xtest_minibatches.append(X_minibatches_j)

                # Group the jth minibatches of each U dataset
                Utest_minibatches = []
                for jth_minibatches in zip(*all_dsetU_minibatches):
                    U_minibatches_j = []
                    for batch in jth_minibatches:
                        U_minibatches_j.append(batch)
                    Utest_minibatches.append(U_minibatches_j)

                #Xtest_minibatches, Utest_minibatches = get_minibatches(Xtest, Utest)
                Nmb_test = len(Xtest_minibatches)
                ltest = 0.
                for Xtest_batch, Utest_batch in zip(Xtest_minibatches, Utest_minibatches):
                    test_losses = total_loss(ae, fdyn, Xtest_batch,
                                             Utest_batch, m, it)
                    ltest += test_losses[0].detach()
                ltest /= Nmb_test

        # To never stop training early, set $stop_loss = 0
        #if ltest <= params.stop_loss:
        #    break

        # Initialize average losses, batch counter for this epoch
        l_ep = 0.
        l_all = []
        lrec_ep = 0.
        lmstep_ep = 0.
        ljac_ep = 0.
        counter = 0


        # For X, U (U may be `None`, for like for drift-loss dataset)
        # of each dataset, get minibatches
        all_dsetX_minibatches = []
        all_dsetU_minibatches = []
        for dsetj_X, dsetj_U in zip(X, U):
            dsetj_X_minibatches, dsetj_U_minibatches = get_minibatches(dsetj_X, dsetj_U)
            all_dsetX_minibatches.append(dsetj_X_minibatches)
            all_dsetU_minibatches.append(dsetj_U_minibatches)

        # Group the jth minibatches of each X dataset
        X_minibatches = []
        for jth_minibatches in zip(*all_dsetX_minibatches):
            X_minibatches_j = []
            for batch in jth_minibatches:
                X_minibatches_j.append(batch)
            X_minibatches.append(X_minibatches_j)

        # Group the jth minibatches of each U dataset
        U_minibatches = []
        for jth_minibatches in zip(*all_dsetU_minibatches):
            U_minibatches_j = []
            for batch in jth_minibatches:
                U_minibatches_j.append(batch)
            U_minibatches.append(U_minibatches_j)

        # Weights will be updated Nmb (Num. minibatches) number of times
        Nmb = len(X_minibatches)
    
        if params.tqdm_each_epoch:
            bar = tqdm(zip(X_minibatches, U_minibatches),
                       total=Nmb, 
                       bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', 
                       ascii=' #')
        else:
            bar = zip(X_minibatches, U_minibatches)

        for X_batch, U_batch in bar:
            ###############################
            ### BEGIN PROCESS MINIBATCH ###
            ###############################
            
            # Update model/optimizers
            if (m >= 0) and params.adversarial_training:
                X_batch, U_batch = adversarial_training(X_batch, U_batch, ae, fdyn)

            if X_batch[0].shape[0] > 0:
                loss_minibatch = train_step(X_batch, U_batch, ae, ae_opt, fdyn, fdyn_opt, m, it)
            l, (lrec, lmstep, ljac), l_all_batch = loss_minibatch
            l_ep += l.detach()
            l_all.append(l_all_batch)
            lrec_ep += lrec.detach()
            lmstep_ep += lmstep.detach()
            ljac_ep += ljac.detach()

            # Save frame for training video
            if (it <= params.video_end_epoch) and (counter % 20 == 0) and (params.make_video):
                frames.append(plot_trajectories(ae, fdyn, X[0][params.video_idx].unsqueeze(0),
                                                          U[0][params.video_idx].unsqueeze(0),
                                                          1, steps=params.traj_len,
                                                          video=True))

            # Compute Jacobians at origin/0-control, LQR dynamics eigenvalues
            if (params.plot_eigenvalues  and counter % 100 == 0)\
               or (params.plot_jac_norm  and counter % 100 == 0)\
               or (params.plot_jac_err   and counter % 100 == 0):

                # Don't factor these calculations into weight updates
                ae.eval()
                if params.control_affine or params.linear_state_space:
                    if params.linear_state_space_offset:
                        for f in fdyn_drift:
                            f.eval()
                    else:
                        fdyn_drift.eval()
                        fdyn_cntrl.eval()
                elif fdyn_opt is not None:
                    fdyn.eval()

                with torch.no_grad():

                    # Get Jacobians at origin
                    F, _, A, B, = get_LQR_params(ae, fdyn, ret_AB=True)
                    # A has shape (4, 4), B has shape (4, 1)
                    A_norm.append(torch.sum(A**2))
                    B_norm.append(torch.sum(B**2))

                    # Plot Jacobian errors for neural dynamics (d_z = d_x):
                    if params.plot_jac_err:
                        assert params.d_z == params.d_x
                        x0 = torch.tensor([0.,0.,0.,0.])
                        u0 = torch.tensor(0.)
                        dfdx = jacrev(_dxdt_torch, argnums=0)(x0, u0)
                        # `unsqueeze(-1)` makes shape (4, 1)
                        dfdu = jacrev(_dxdt_torch, argnums=1)(x0, u0).unsqueeze(-1)
                        A_err.append(torch.sum( (dfdx - A)**2 ))
                        B_err.append(torch.sum( (dfdu - B)**2 ))

                    # Compute eigenvalues of LQR-controlled system
                    P = A - B @ F
                    e = torch.linalg.eigvals(P).real
                    eigenvalues.append(e.detach().numpy())

            # Minibatch completed, increment counter
            counter += 1

            #############################
            ### END PROCESS MINIBATCH ###
            #############################

        # End of epoch, save stats
        l_ep /= Nmb
        lrec_ep /= Nmb
        lmstep_ep /= Nmb
        ljac_ep /= Nmb
        losses.append(l_ep.cpu())
        losses_rec.append(lrec_ep.cpu())
        losses_mstep.append(lmstep_ep.cpu())
        losses_jac.append(ljac_ep.cpu())
        losses_all.append(l_all)

        # Print stats every $test_inc iterations
        if it % params.test_inc == 0:
            print("ep {}: train {}    test {}".format(it, l_ep, ltest))
            print("rec {}    mstep {}    jac {}".format(lrec_ep, lmstep_ep, ljac_ep))

        #################
        ### END EPOCH ###
        #################

    # Plot losses over training 
    plt.cla()
    plt.title("training losses")
    n_losses = len(losses_all[0][0].items())
    n_batches = len(losses_all[0])
    loss_lists = [[[], pair[1]] for _, pair in losses_all[0][0].items()]
    for ep, l_all in enumerate(losses_all):
        batch_losses = [0. for i in range(n_losses)]
        for l_all_batch in l_all:
            for l_idx, pair in enumerate(l_all_batch.values()):
                batch_losses[l_idx] += pair[0].cpu()
        batch_losses = [L/n_batches for L in batch_losses]
        for i in range(n_losses):
            loss_lists[i][0].append(batch_losses[i])     
    for loss, label in loss_lists:
        plt.plot(loss[-50:], label=label)
    plt.legend()
    plt.show()

    # Plot eigenvalues of A - BF
    if params.plot_eigenvalues:
        eigenvaluesT = np.array(eigenvalues).T
        plt.title("Eigenvalues of A - BF over training")
        for i, ei_of_t in enumerate(eigenvaluesT):
            plt.plot(ei_of_t, 'b.')
        plt.show()

    # Plot Frobenius norms of A and B
    if params.plot_jac_norm:
        plt.title("Frob. norm of A & B")
        plt.plot(A_norm, label="|A|")
        plt.plot(B_norm, label="|B|")
        plt.legend()
        plt.show()

    # Plot error between true Jacobian at origin/0-control and Jacobian of fdyn
    if params.plot_jac_err:
        plt.title("||A - df/dx||^2")
        plt.plot(A_err, label="|A-df/dx|^2")
        plt.legend()
        plt.show()
        plt.title("||B - df/du||^2")
        plt.plot(B_err, label="|B-df/du|^2")
        plt.legend()
        plt.show()

    # Video of cart and pole trajectory reconstructions evolving over training
    if params.make_video:
        training_example_video(fname, X)

    if params.latent_space_training_video:
        rewards, completion_rate, gammas = latent_space_video(ae_list, fdyn_list, n_pts=100, T=250)

    return ae, fdyn, ae_opt, fdyn_opt, ae_list, fdyn_list, rewards, completion_rates, gammas


# weight-update step to be called from train()
def train_step(X_batch, U_batch, ae, ae_opt,
               fdyn, fdyn_opt, m, ep):

    if params.control_affine or params.linear_state_space:
        fdyn_drift, fdyn_cntrl = fdyn
        fdyn_drift_opt, fdyn_cntrl_opt = fdyn_opt

    ae.train()
    if params.control_affine or params.linear_state_space:
        if params.linear_state_space_offset:
            for f in fdyn_drift:
                f.train()
        else:
            fdyn_drift.train()
            fdyn_cntrl.train()
    elif fdyn_opt is not None:
        fdyn.train()

    ae_opt.zero_grad()
    if params.control_affine or params.linear_state_space:
        if params.linear_state_space_offset:
            for f_opt in fdyn_drift_opt:
                f_opt.zero_grad()
        else:
            fdyn_drift_opt.zero_grad()
            fdyn_cntrl_opt.zero_grad()
    elif fdyn_opt is not None:
        fdyn_opt.zero_grad()

    loss = total_loss(ae, fdyn, X_batch, U_batch, m, ep)
    loss[0].backward()
    if params.clip_gradient:
        nn.utils.clip_grad_norm_(ae.parameters(), max_norm=1)
        if params.control_affine or params.linear_state_space:
            if params.linear_state_space_offset:
                for f in fdyn_drift:
                    nn.utils.clip_grad_norm_(f.parameters(), max_norm=1)
            else:
                nn.utils.clip_grad_norm_(fdyn_drift.parameters(), max_norm=1)
            nn.utils.clip_grad_norm_(fdyn_cntrl.parameters(), max_norm=1)
        elif fdyn is not None:
            nn.utils.clip_grad_norm_(fdyn.parameters(), max_norm=1)

    ae_opt.step()
    if params.control_affine or params.linear_state_space:
        if params.linear_state_space_offset:
            for f_opt in fdyn_drift_opt:
                f_opt.step()
        else:
            fdyn_drift_opt.step()
            fdyn_cntrl_opt.step()
    elif fdyn_opt is not None:
        fdyn_opt.step()

    return loss

