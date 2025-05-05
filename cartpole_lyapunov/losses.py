import numpy as np
import torch
import torch.nn as nn
import cartpole
from torch.autograd.functional import jacobian #, grad
from torch.func import jacrev, vmap
#from torchdiffeq import odeint
from torchdiffeq import odeint_adjoint as odeint
from integration import _flow_diff
from cartpole import dxdt_torch, _dxdt_torch
from controls import LQR
from data_utils import mix_dataset

import params


##################
### Evaluation ###
##################

def gamma_forwards(x_traj, z_traj, u_traj, ae, fdyn):
    assert params.linear_state_space
    '''
    fdyn_drift, fdyn_cntrl = fdyn
    z_next = fdyn_drift(z_curr).reshape(params.d_z, params.d_z) @ z_curr.reshape(params.d_z, 1) +\
             fdyn_cntrl(z_curr).reshape(params.d_z, params.d_u) @ torch.tensor([u]).reshape(params.d_u, 1)
    gamma = torch.linalg.norm(z_next.squeeze() - ae.encode(torch.tensor(x_next).float()).squeeze())
    return gamma.cpu().detach().numpy()
    '''
    fdyn_drift, fdyn_cntrl = fdyn
    x_traj = torch.tensor(x_traj).float() # ARRIVES AS NP.ARRAY
    z_traj = torch.vstack(z_traj) # ARRIVES AS LIST OF TORCH TENSORS
    u_traj = torch.tensor(u_traj).float() # ARRIVES AS NP.ARRAY
    #print(x_traj.shape)
    #print(z_traj.shape)
    #print(u_traj.shape)
    z_next_pred = (fdyn_drift(z_traj).reshape(-1, params.d_z, params.d_z) @ z_traj.reshape(-1, params.d_z, 1) +\
                   fdyn_cntrl(z_traj).reshape(-1, params.d_z, params.d_u) @ u_traj.reshape(-1, params.d_u, 1)).squeeze()
    z_next_true = ae.encode(x_traj[1:])
    #print(z_next_pred.shape)
    #print(z_next_true.shape)
    #print((z_next_pred-z_next_true).shape)
    #print(torch.linalg.norm(z_next_pred - z_next_true, dim=1).shape)
    return torch.linalg.norm(z_next_pred - z_next_true, dim=1).cpu().detach().numpy()

def gamma_backwards(x_traj, z_traj, u_traj, ae, fdyn):
    assert params.linear_state_space
    '''
    fdyn_drift, fdyn_cntrl = fdyn
    z_next = fdyn_drift(z_curr).reshape(params.d_z, params.d_z) @ z_curr.reshape(params.d_z, 1) +\
             fdyn_cntrl(z_curr).reshape(params.d_z, params.d_u) @ torch.tensor([u]).reshape(params.d_u, 1)
    gamma = torch.linalg.norm(z_next.squeeze() - ae.encode(torch.tensor(x_next).float()).squeeze())
    return gamma.cpu().detach().numpy()
    '''
    fdyn_drift, fdyn_cntrl = fdyn
    x_traj = torch.tensor(x_traj).float() # ARRIVES AS NP.ARRAY
    z_traj = torch.vstack(z_traj) # ARRIVES AS LIST OF TORCH TENSORS
    u_traj = torch.tensor(u_traj).float() # ARRIVES AS NP.ARRAY
    #print(x_traj.shape)
    #print(z_traj.shape)
    #print(u_traj.shape)
    z_next_pred = (fdyn_drift(z_traj).reshape(-1, params.d_z, params.d_z) @ z_traj.reshape(-1, params.d_z, 1) +\
                   fdyn_cntrl(z_traj).reshape(-1, params.d_z, params.d_u) @ u_traj.reshape(-1, params.d_u, 1)).squeeze()
    #z_next_true = ae.encode(x_traj[1:])
    x_next_true = x_traj[1:]
    x_next_pred = ae.decode(z_next_pred)
    #print(z_next_pred.shape)
    #print(z_next_true.shape)
    #print((z_next_pred-z_next_true).shape)
    #print(torch.linalg.norm(z_next_pred - z_next_true, dim=1).shape)
    return torch.linalg.norm(x_next_pred - x_next_true, dim=1).cpu().detach().numpy()

    
    


############################
### Reward (for testing) ###
############################

def cartpole_reward(x_traj, u_traj, Q, R):
    #print(trajectory.shape)
    #reward = np.sum(np.e**(-np.linalg.norm(trajectory, axis=1)))
    cost = np.sum(x_traj[:,np.newaxis,:] @ (Q[0,0] * np.eye(params.d_x))[np.newaxis,...] @ x_traj[...,np.newaxis]) +\
           np.sum(u_traj[:,np.newaxis,:] @ (R[0,0] * np.eye(params.d_u))[np.newaxis,...] @ u_traj[...,np.newaxis])
    return cost


#################################
##  Training loop calls this:  ##
#################################################################################


def total_loss(ae, fdyn, X, U, m, ep=None):

    compute_drift_loss = params.linear_state_space or params.learn_drift

    if type(X) is list:
        X, Xdrift = X
        U, _ = U # U for drift dataset will be `None`

    #assert (len(X.shape) == 3) and (len(U.shape) == 3)
    if params.currently_active_learning and (ep > params.mix_start):
        X, U = mix_dataset(X, U, ae, fdyn, ep)

    # For plotting:
    losses = {}

    # (auto)-Encoder losses
    Lenc = torch.tensor(0.0, requires_grad=True)    

    # Pre-compute Z because most losses use it 
    N, T, _ = X.shape
    Ndrift, Tdrift, _ =  Xdrift.shape
    Z = ae.encode(X.reshape(-1, params.d_x)).reshape(N, T, params.d_z) 
    Zdrift = ae.encode(Xdrift.reshape(-1, params.d_x)).reshape(Ndrift, Tdrift, params.d_z)

    # Autoencoder state reconstruction
    if params.penalize_rec:
        if ae.vae:
            if params.debug:
                print("vae loss active")
            L_vae = vae_loss(ae, X.reshape(-1, params.d_x))
            losses["vae"] = (L_vae.detach(), "vae")
            Lenc = Lenc + vae
        else:
            if params.single_example_loss:
                if params.debug:
                    print("single-example rec loss active")
                L_rec_single_elem = vmap(rec_single_example,in_dims=(None, 0))(ae, X.reshape(-1, params.d_x)).sum()
                losses["rec single-ex."] = (L_rec_single_elem.detach(), "rec. single-ex.")
                Lenc = Lenc + L_rec_single_elem
            else:
                if params.debug:
                    print("rec loss active")
                L_rec = params.lam_rec * rec_loss(ae, X.reshape(-1, params.d_x), Z.reshape(-1, params.d_z))
                losses["rec"] = (L_rec.detach(), "rec.")
                Lenc = Lenc + L_rec

    if params.penalize_reproj:
        if params.debug:
            print("reprojection loss active")
        L_reproj = params.lam_reproj * reproj_loss(ae, X.reshape(-1, params.d_x), Z.reshape(-1, params.d_z))
        losses["reproj"] = (L_reproj.detach(), "reproj.")
        Lenc = Lenc + L_reproj

    # Encourage E(0) = 0
    if params.penalize_latent_origin_norm:
        if params.debug:
            print("encoded origin loss active")
        L_origin = params.lam_origin * latent_origin_norm_loss(ae, X)
        losses["origin"] = (L_origin.detach(), "||E(0)||")
        Lenc = Lenc + L_origin

    # Encourage isotropic latents
    if params.penalize_isotropic_latent:
        if params.debug:
            print("isotropic latent loss active")
        L_isotropic = params.lam_isotropic_latent * isotropic_latent_loss(ae, Z.reshape(-1, params.d_z))
        losses["isotropic"] = (L_isotropic.detach(), "isotropic")
        Lenc = Lenc + L_isotropic

    # Constrastive loss
    if params.contrastive_encoder_loss:
        if params.debug:
            print("contrastive encoder loss active")
        L_contrastive_rec = 1e-2*constrastive_rec_loss(ae, X)
        losses["contrastive"] = (L_contrastive_rec.detach(), "contrastive enc.")
        Lenc = Lenc + L_contrastive_rec

    # Dynamics Jacobian reconstruction
    if params.rec_jac:
        if params.debug:
            print("jacobian reconstruction loss active")
        L_rec_jac = rec_jac(fdyn, X[:,:-1].reshape(-1, params.d_x), U.reshape(-1, params.d_u))
        losses["jac. rec."] = (L_rec_jac.detach(), "jac. rec.")
        Ljac = L_rec_jac
    else:
        Ljac = torch.tensor(0.)

    # Derivative reconstruction (no dynamics model)
    if params.enc_true_dyn:
        if params.debug:
            print("derivative reconstruction (no latent dynamics) loss active")
        L_rec_deriv = rec_deriv(fdyn, X, U)
        losses["deriv. rec."] = (L_rec_deriv.detach(), "reriv. rec.")
        Lenc = Lenc + L_rec_deriv

    # Dynamics losses
    Ldyn = torch.tensor(0.0, requires_grad=True)

    # Dynamics multi-step prediction
    if params.predict_mstep:
        if params.debug:
            print("mstep prediction loss active")
        L_mstep = params.lam_mstep_backwards * mstep(ae, fdyn, X[1:], Z[1:], U[1:], m)
        losses["mstep"] = (L_mstep.detach(), "backwards mstep")
        Ldyn = Ldyn + L_mstep
        if compute_drift_loss:
            if params.debug:
                print("drift loss active")
            L_drift = params.lam_drift_backwards * drift_loss(ae, fdyn, Xdrift, Zdrift)
            losses["backwards drift"] = (L_drift.detach(), "backwards drift")
            Ldyn = Ldyn + L_drift

    '''
    if params.active_learning:
        if params.debug:
            print("active learning loss enabled")
        if params.currently_active_learning:
            L_active = params.lam_active * active_learning_loss(ae, fdyn)
            #print(L_active)
        else:
            L_active = torch.tensor(0.)
        losses["active"] = (L_active.detach(), "active learning")
        Ldyn = Ldyn + L_active
    '''

    if params.ode:
        if params.single_example_loss:
            if params.debug:
                print("single example ode loss active")
            L_ode_single_example = ode_single_example(ae, fdyn, X, U, m)
            losses["ode single-ex."] = (L_ode_single_example.detach(), "ode single-ex.")
            Ldyn = Ldyn + L_ode_single_example
        else:
            if params.debug:
                print("ode loss active")
            L_ode = ode_loss(ae, fdyn, X, U, m)
            losses["ode"] = (L_ode.detach(), "ode")
            Ldyn = Ldyn + L_ode

    if params.enc_true_dyn:
        if params.debug:
            print("mstep encode true dynamics loss active")
        L_enc_true_dyn = mstep_enc_dyn(ae, fdyn, X, U, m)
        losses["enc. true dyn."] = (L_enc_true_dyn.detach(), "enc. true dyn.")
        Ldyn = Ldyn + L_enc_true_dyn
    if params.penalize_encoder_diagram:
        if params.debug:
            print("encoder diagram loss active")
        L_enc_diagram = enc_diagram_loss(ae, fdyn, X, U)
        losses["forwards 1-step"] = (L_enc_diagram.detach(), "forwards 1-step")
        Ldyn = Ldyn + L_enc_diagram
        
    if compute_drift_loss and (params.penalize_encoder_diagram or\
                               params.penalize_encoder_diagram_mstep):
        if params.debug:
            print("encoder diagram drift loss active")
        L_enc_diagram_drift = params.lam_drift_forwards * enc_diagram_drift_loss(ae, fdyn, Xdrift, Zdrift)
        losses["forwards drift"] = (L_enc_diagram_drift.detach(), "forwards drift")
        Ldyn = Ldyn + L_enc_diagram_drift # THERE WILL BE AN ERROR HERE

    if params.penalize_encoder_diagram_mstep:
        if params.debug:
            print("encoder diagram mstep loss active")
        L_enc_diagram_mstep = params.lam_mstep_forwards * enc_diagram_loss_mstep(ae, fdyn, X, Z, U, m)
        losses["forwards mstep"] = (L_enc_diagram_mstep.detach(), "forwards mstep")
        Ldyn = Ldyn + L_enc_diagram_mstep

    if params.penalize_decoder_diagram:
        if params.debug:
            print("decoder diagram loss active")
        L_decoder_diagram = dec_diagram_loss(ae, fdyn, X, U)
        losses["dec. diagram"] = (L_decoder_diagram.detach(), "dec. diagram")
        Ldyn = Ldyn + L_decoder_diagram
        if compute_drift_loss:
            if params.debug:
                print("decoder diagram drift loss active")
            L_dec_diagram_drift = dec_diagram_drift_loss(ae, fdyn, X_drift)
            losses["dec. diagram drift"] = (L_dec_diagram_drift.detach(), "dec. diagram drift")
            Ldyn = Ldyn + L_dec_diagram_drift

    # Jacobian norm penalities
    if params.penalize_ae_jac_norm:
        if params.debug:
            print("ae jacobian norm loss active")
        L_jac_norm_ae =  jac_norm_ae_loss(ae, X, Z)
        losses["ae jac. norm"] = (L_jac_norm_ae.detach(), "ae jac. norm")
        Ljac = Ljac + L_jac_norm_ae

    if params.penalize_fdyn_jac_norm:
        if params.debug:
            print("fdyn jacobian norm loss active")
        L_jac_norm_fdyn = jac_norm_fdyn_loss(ae, fdyn, X, Z)
        losses["fdyn jac. norm"] = (L_jac_norm_fdyn.detach(), "fdyn jac. norm")
        Ljac = Ljac + L_jac_norm_fdyn


    #for name, loss in losses.items():
    #   print(name, loss)

    return Lenc + Ldyn + Ljac, (Lenc, Ldyn, Ljac), losses


#################################################################################



##############
##  Helper  ##
#################################################################################


class LatentDynamics(nn.Module):
    def __init__(self, fdyn, U):
        super().__init__()
        self.fdyn = fdyn
        self.U = U

    def forward(self, t, Z):
        if t >= params.traj_len * params.DT:
            return 0
        arg = torch.hstack( (Z, self.U[:,int(self.U.shape[1]*t/(params.traj_len*params.DT))]) )
        return self.fdyn(arg)


#################################################################################



##############################
##  Maintained (12/30/24)   ##
#################################################################################


def active_learning_loss(ae, fdyn):
    assert params.linear_state_space
    fdyn_drift, fdyn_cntrl = fdyn
    lqr = LQR(ae, fdyn)
    X0 = 2*params.x_range_active*(torch.rand(params.batch_size, params.d_x) - 0.5)
    Z0 = ae.encode(X0)
    U = lqr(Z0).T.to("cuda")
    X1 = _flow_diff(dxdt_torch, X0, U, cartpole.DT)[-1]
    Z1 = (fdyn_drift(Z0).reshape(-1, params.d_z, params.d_z) @ Z0.unsqueeze(-1) +\
          fdyn_cntrl(Z0).reshape(-1, params.d_z, params.d_u) @ U.unsqueeze(-1)).squeeze() 
    Z1_true = ae.encode(X1)
    l = params.active_batch_reduc(torch.sum((Z1_true - Z1)**2, axis=1), dim=0)
    return l
    #dxdt_torch()


def isotropic_latent_loss(ae, Z):
    #Z = ae.encode(X).reshape(-1, params.d_z)
    cov = torch.sum(torch.einsum('ij,ik->ijk', Z, Z), dim=0) / Z.shape[0]
    #l = torch.sum((cov - torch.eye(params.d_z))**2)
    l = torch.linalg.norm(cov - torch.eye(params.d_z))
    return l


def latent_origin_norm_loss(ae, X):
    N = X.shape[0]
    x_origin = torch.tensor([0.,0.,0.,0.])
    z_origin = ae.encode(x_origin)
    #return torch.sum(torch.sqrt(torch.abs(z_origin)))
    return torch.linalg.norm(z_origin)


def mstep(ae, fdyn, X, Z, U, m):
    # Expects data of form:
    # (N)umbe of trajectories x (T)ime horizon x state-dimension
    assert (len(X.shape) == 3) and (len(U.shape) == 3)

    # If using control-affine latent dynamics, fdyn will be tuple
    if params.control_affine or params.linear_state_space:
        fdyn_drift, fdyn_cntrl = fdyn
        if params.linear_state_space_offset:
            fdyn_drift, fdyn_offset = fdyn_drift

    N = X.shape[0]
    T = X.shape[1]
    L = 0.
    #Z = ae.encode(X.reshape(-1, params.d_x)).reshape(N, T, params.d_z)
    Zref = Z
    for t in range(m):
        Z = Z[:,:-1,:].reshape(-1, params.d_z)
        U = U.reshape(-1, params.d_u)
        if params.learn_residual or params.linear_state_space_offset:
            Z_tmp = Z
        if params.control_affine:
            Z = fdyn_drift(Z) + (fdyn_cntrl(Z).unsqueeze(-1) @ U.unsqueeze(-1)).squeeze()
        elif params.linear_state_space:
            #print("z shape", Z.shape)
            #print("u shape", U.shape)
            Z = (fdyn_drift(Z).reshape(-1, params.d_z, params.d_z) @ Z.unsqueeze(-1)).squeeze() +\
                (fdyn_cntrl(Z).reshape(-1, params.d_z, params.d_u) @ U.unsqueeze(-1)).squeeze()
        else:
            Z = fdyn(torch.hstack( (Z, U) ))
        if params.learn_residual:
            Z = Z_tmp + Z
        if params.linear_state_space_offset:
            # make sure to use previous Z (Z_tmp)
            Z = Z + fdyn_offset(Z_tmp)
        U = U.reshape(N, T-(t+1), params.d_u)
        U = U[:,1:,:]
        Xhat = ae.decode(Z)
        Z = Z.reshape(N,T-(t+1),params.d_z)
        Xt = X[:,t+1:,:].reshape(-1, params.d_x)
        #Zt = ae.encode(Xt)
        Zt = Zref[:,t+1:,:].reshape(-1, params.d_z)
        # try activating this later in training:
        #L += torch.sum(torch.sum((Zt - Z.reshape(-1, params.d_z))**2, dim=1), dim=0)
        #L += torch.sum(torch.sum((Xt - Xhat)**2, dim=1),dim=0)
        L += params.mstep_batch_reduc(torch.sum((Xt[:,params.symbols] - Xhat[:,params.symbols])**2, dim=1),dim=0)

    return L


def rec_loss(ae, X, Z):
    assert len(X.shape) == 2
    #Xhat = ae(X)
    Xhat = ae.decode(Z)
    '''
    print("X SHAPE", X.shape)
    if Xhat.isnan().any():
        print("AE BLEW UP")
    if X[:,params.symbols].isnan().any():
        print("TARGET BLEW UP")
    if torch.sum((Xhat[:,params.symbols] - X[:,params.symbols])**2, dim=1).isnan().any():
        print("DIFF BLEW UP")
    print("NUMEL", Xhat.numel())
    print("MAX MEAN ARG", torch.max(torch.sum((Xhat[:,params.symbols] - X[:,params.symbols])**2, dim=1)))
    if params.rec_batch_reduc(torch.sum((Xhat[:,params.symbols] - X[:,params.symbols])**2, dim=1),dim=0).isnan().any():
        print("REDUC BLEW UP")
    '''
    Lrec = params.rec_batch_reduc(torch.sum((Xhat[:,params.symbols] - X[:,params.symbols])**2, dim=1),dim=0)
    '''
    if Lrec.isnan().any():
        print("RESULT BLEW UP")
    print("Lrec", Lrec)
    '''
    return Lrec


def reproj_loss(ae, X, Z):
    eps = 0. #1e-3
    #Z = ae.encode(X)
    #Z = Z.reshape(-1, params.d_z)
    '''
    Zi_var = [torch.mean((Z[:,i] - torch.mean(Z[:,i]))**2) for i in range(params.d_z)]
    Z_pert = Z
    for i in range(params.d_z):
        Z_pert[:,i] += 2*Zi_var[i]*eps*(torch.rand(Z[:,i].shape) - 0.5)
    '''
    Z_pert = Z
    X = ae.decode(Z_pert)
    Z_pert_hat = ae.encode(X)
    L = params.reproj_batch_reduc(torch.sum(Z_pert_hat - Z_pert, dim=1)**2)
    return L


def constrastive_rec_loss(ae, X):
    ptb = torch.randn(size=[X.shape[0]*X.shape[1], len(params.ignored)])
    X_ptb = torch.clone(X).reshape(-1, params.d_x)
    X_ptb[:,params.ignored] += ptb
    Z = ae.encode(X.reshape(-1, params.d_x))
    Z_ptb = ae.encode(X_ptb)
    L = params.contrastive_rec_batch_reduc(torch.sum((Z - Z_ptb)**2, dim=1))
    return L


def rec_single_example(ae, x):
    xhat = ae(x) 
    return torch.sum((xhat - x)**2)


def rec_jac(fdyn, X, U):

    # reconstruct jacobian of perturbations around a point
    if params.ptb_jac:
        X = params.ptb_eps_x * torch.randn(size=X.shape) + params.ptb_x.unsqueeze(0)
        U = params.ptb_eps_u * torch.randn(size=U.shape) + params.ptb_u.unsqueeze(0)

    X.requires_grad = True
    U.requires_grad = True

    A = vmap(jacrev(_dxdt_torch, argnums=0))(X, U[:,0])
    B = vmap(jacrev(_dxdt_torch, argnums=1))(X, U[:,0]).unsqueeze(-1)
    #print(A.shape)
    #print(B.shape)
    jac_xu = vmap(jacrev(fdyn))(torch.cat( (X, U), dim=1))
    Ahat = jac_xu[:,:,:-1]
    Bhat = jac_xu[:,:,-1].unsqueeze(-1)
    #Bhat = vmap(jacrev(fdyn))(torch.cat( (X, U), dim=1))
    #print(Ahat.shape)
    #print(Bhat.shape)
    L = params.jac_batch_reduc(torch.sum( (A - Ahat)**2 , dim=(1, 2)))
    L += params.jac_batch_reduc(torch.sum( (B - Bhat)**2 , dim=(1, 2)))
    return L


def drift_loss(ae, fdyn, X, Z):
    if params.linear_state_space or params.control_affine:
        fdyn_drift, _ = fdyn
    if params.linear_state_space_offset:
        fdyn_drift, fdyn_offset = fdyn_drift

    #Z0 = ae.encode(X[:,0])
    Z0 = Z[:,0]
    if params.linear_state_space:
        if params.learn_residual:
            Zhat = Z0 + (fdyn_drift(Z0).reshape(-1, params.d_z, params.d_z) @ Z0.unsqueeze(-1)).squeeze()
        else:
            Zhat = (fdyn_drift(Z0).reshape(-1, params.d_z, params.d_z) @ Z0.unsqueeze(-1)).squeeze()
        if params.linear_state_space_offset:
            Zhat = Zhat + fdyn_offset(Z0)
    elif params.control_affine:
        Zhat = fdyn_drift(Z0).squeeze()
    else:
        Zhat = fdyn(torch.hstack([Z0, torch.zeros_like(Z0[:,:1])]))
    Xhat = ae.decode(Zhat)
    L = params.drift_batch_reduc(torch.sum((X[:,1,params.symbols] - Xhat[:,params.symbols])**2, dim=1))
    #print("A loss", L)
    return L


def jac_norm_fdyn_loss(ae, fdyn, X, Z):
    fdyn_drift, fdyn_cntrl = fdyn
    X = X.reshape(-1, params.d_x)
    Z = Z.reshape(-1, params.d_z)
    #Z = ae.encode(X)
    jac_fdyn_drift = vmap(jacrev(fdyn_drift))(Z).reshape(-1, params.d_z**3)
    jac_fdyn_cntrl = vmap(jacrev(fdyn_cntrl))(Z).reshape(-1, params.d_z*params.d_z*params.d_u)
    return params.jac_norm_fdyn_batch_reduc(torch.linalg.norm(jac_fdyn_drift, dim=1) +\
                                            torch.linalg.norm(jac_fdyn_cntrl, dim=1))


def jac_norm_ae_loss(ae, X, Z):
    X = X.reshape(-1, params.d_x)
    Z = Z.reshape(-1, params.d_z) 
    #Z = ae.encode(X)
    jac_encoder = vmap(jacrev(ae.encoder))(X).reshape(-1, params.d_z*params.d_x)
    jac_decoder = vmap(jacrev(ae.decoder))(Z).reshape(-1, params.d_x*params.d_z)
    return params.jac_norm_ae_batch_reduc(torch.linalg.norm(jac_encoder, dim=1) +\
                                          torch.linalg.norm(jac_decoder, dim=1))


def enc_diagram_loss(ae, fdyn, X, U):
    assert params.linear_state_space

    fdyn_drift, fdyn_cntrl = fdyn
    Z0 = ae.encode(X[:,:-1].reshape(-1, params.d_x)) # (N*T, d_x)
    Zhat = (fdyn_drift(Z0).reshape(-1, params.d_z, params.d_z) @ Z0.unsqueeze(-1)+\
            fdyn_cntrl(Z0).reshape(-1, params.d_z, params.d_u) @ U.reshape(-1, params.d_u, 1)).squeeze()
    Ztrue = ae.encode(X[:,1:].reshape(-1, params.d_x))
    return params.enc_diagram_batch_reduc(torch.sum((Ztrue - Zhat)**2, dim=1))


def enc_diagram_drift_loss(ae, fdyn, X, Z):
    #assert params.linear_state_space
    if params.linear_state_space or params.control_affine:
        fdyn_drift, _ = fdyn
    #Z0 = ae.encode(X[:,:-1].reshape(-1, params.d_x)) # (N*T, d_x)
    Z0 = Z[:,:-1].reshape(-1, params.d_z)
    if params.linear_state_space:
        Zhat = (fdyn_drift(Z0).reshape(-1, params.d_z, params.d_z) @ Z0.unsqueeze(-1)).squeeze()
    else:
        Zhat = fdyn(torch.hstack([Z0, torch.zeros_like(Z0[:,:1])]))
    #Ztrue = ae.encode(X[:,1:].reshape(-1, params.d_x))
    Ztrue = Z[:,1:].reshape(-1, params.d_z)
    return params.enc_diagram_batch_reduc(torch.sum((Ztrue - Zhat)**2, dim=1))


def enc_diagram_loss_mstep(ae, fdyn, X, Z, U, m):
    #assert params.linear_state_space
    N = X.shape[0]
    T = X.shape[1]

    L = torch.tensor(0.0)
    #Z = ae.encode(X[:,:-1].reshape(-1, params.d_x)) # (N*T, d_x)
    Zref = Z
    Z = Z[:,:-1].reshape(-1, params.d_z)
    for i in range(m):
        if params.linear_state_space:
            fdyn_drift, fdyn_cntrl = fdyn
            Z = (fdyn_drift(Z).reshape(-1, params.d_z, params.d_z) @ Z.unsqueeze(-1)+\
                 fdyn_cntrl(Z).reshape(-1, params.d_z, params.d_u) @ U.reshape(-1, params.d_u, 1)).squeeze()
        else:
            Z = fdyn(torch.hstack([Z, U.reshape(-1, params.d_u)]))
        #Ztrue = ae.encode(X[:,1+i:].reshape(-1, params.d_x))
        Ztrue = Zref[:,1+i:].reshape(-1, params.d_z)
        Li = params.enc_diagram_batch_reduc(torch.sum((Ztrue - Z)**2, dim=1))
        L += Li
        U = U.reshape(N, T-1-i, params.d_u)
        U = U[:,1:].reshape(-1, params.d_u)
        Z = Z.reshape(N, T-1-i, params.d_z)
        Z = Z[:,:-1].reshape(-1, params.d_z)
    return L


def dec_diagram_loss(ae, fdyn, X, U):
    assert params.linear_state_space

    fdyn_drift, fdyn_cntrl = fdyn
    Z0 = ae.encode(X[:,:-1].reshape(-1, params.d_x)) # (N*T, d_x)
    Zhat = (fdyn_drift(Z0).reshape(-1, params.d_z, params.d_z) @ Z0.unsqueeze(-1)+\
            fdyn_cntrl(Z0).reshape(-1, params.d_z, params.d_u) @ U.reshape(-1, params.d_u, 1)).squeeze()
    Xhat = ae.decode(Zhat)
    Xtrue = X[:,1:].reshape(-1, params.d_x)
    return params.dec_diagram_batch_reduc(torch.sum((Xtrue - Xhat)**2, dim=1))


def dec_diagram_drift_loss(ae, fdyn, X):
    assert params.linear_state_space

    fdyn_drift, fdyn_cntrl = fdyn
    Z0 = ae.encode(X[:,:-1].reshape(-1, params.d_x)) # (N*T, d_x)
    Zhat = (fdyn_drift(Z0).reshape(-1, params.d_z, params.d_z) @ Z0.unsqueeze(-1)).squeeze()
    Xhat = ae.decode(Zhat)
    Xtrue = X[:,1:].reshape(-1, params.d_x)
    return params.dec_diagram_batch_reduc(torch.sum((Xtrue - Xhat)**2, dim=1))


#def adversarial_perturbation(L, f, X, delta):
#    if L = rec_loss:
#        dX = 


#################################################################################



##############################
##  Need Fixing (12/25/24)  ##
#################################################################################

def ode_loss(ae, fdyn, X, U, m):
    N = X.shape[0]
    T = m + 1
    DT = params.DT
    d_x = params.d_x
    d_z = params.d_z

    T_eval = params.DT * torch.arange(m+1)
    X = X[:,:m+1]
    Z = ae.encode(X.reshape(N*T, d_x)).reshape(N, T, d_z)
    L = 0.
    #def fdyn_(t, Z):
    #    arg = torch.hstack( (Z, U[:,int(U.shape[1]*t/(params.traj_len*DT))]) )
    #    return fdyn(arg)
    fdyn_ = LatentDynamics(fdyn, U)
    Z_pred = odeint(fdyn_, Z[:,0,:], T_eval, method=params.ode_method)
    Z_pred = torch.einsum('ijk->jik', Z_pred)
    L += torch.sum((Z - Z_pred)**2)
    Xhat = ae.decode(Z_pred.reshape(-1, d_z)).reshape(N, m+1, d_x)
    L += torch.sum((X - Xhat)**2)
    return L


def ode_single_example(ae, fdyn, X, U, m):
    L = 0.
    X = X[:,:m+1]
    for x, u in zip(X, U):
        z = ae.encode(x.reshape(-1, params.d_x)).reshape(m+1, params.d_z)
        t_eval = params.DT * torch.arange(m + 1)
        fdyn_ = LatentDynamics(fdyn, u.unsqueeze(0))
        z_pred = odeint(fdyn_, z[0].unsqueeze(0), t_eval, method=params.ode_method)
        #print(z_pred.shape)
        xhat = ae.decode(z_pred.reshape(-1, params.d_z)).reshape(m+1, params.d_x)
        L += torch.sum((xhat - x)**2)
    return L


def mstep_enc_dyn(ae, fdyn, X, U, m):
    assert (len(X.shape) == 3) and (len(U.shape) == 3)
    d_x = params.d_x
    d_z = params.d_z
    d_u = params.d_u
    DT = params.DT
    N = X.shape[0]
    T = m + 1

    L = 0.
    # dxdt in fdyn is vmapped
    # THIS WILL ONLY WORK WHEN X IS FULL TRAJECTORY
    Z = ae.encode(X.reshape(-1, d_x)).reshape(N, T, d_z)
    Z = _flow_diff(fdyn, Z[:,:-1].reshape(-1, d_z),
                         U.reshape(-1, d_u), DT)[0]
    Xhat = ae.decode(Z).reshape(N, T-1, d_x)
    L += torch.sum((X[:,1:] - Xhat)**2)
    Z = Z.reshape(N, T-1, d_z)
    Z_true = ae.encode(X[:,1:].reshape(-1, d_x)).reshape(N, T-1, d_z)
    L += torch.mean(torch.sum((Z_true.reshape(-1, d_z) - Z.reshape(-1, d_z))**2, axis=1), 0)

    for t in range(m-1):
        # dz/dt = (dz/dx)(dx/dt)
        Z = _flow_diff(fdyn, Z[:,:-1].reshape(-1, d_z),
                             U[:,1+t:].reshape(-1, d_u), DT)[0].reshape(N, T-(t+2), d_z)
        Xhat = ae.decode(Z.reshape(-1, d_z)).reshape(N, T-(t+2), d_x)
        L += torch.mean(torch.sum((X[:,t+2:].reshape(-1, d_x) - Xhat.reshape(-1, d_x))**2, axis=1), axis=0)
        Z_true = ae.encode(X[:,:T-(t+2)]).reshape(N, T-(t+2), d_z)
        L += torch.mean(torch.sum((Z_true.reshape(-1, d_z) - Z.reshape(-1, d_z))**2, axis=1), axis=0)

    return L


def rec_deriv(fdyn, X, U):
    dXdt = dxdt_torch(X[:,:-1].reshape(-1, params.d_x), U.reshape(-1, params.d_u))
    dXdt_hat = fdyn.ae(dXdt)
    
    Lrec = torch.sum((dXdt - dXdt_hat)**2)
    return Lrec


#TODO
def vae_loss(ae, X):
    assert len(X.shape) == 2

    Xhat = ae(X)
    #Lrec = params.inner_reduction(torch.sum((Xhat - X)**2, dim=1),dim=0)
    Lrec = torch.sum(torch.mean((Xhat - X)**2, dim=1),dim=0)


    mu, logvar = torch.vmap(ae.get_latent_params)(X)
    std = torch.exp(logvar/2)
    kld = torch.sum(-0.5 * torch.sum(1 + logvar - mu**2 - torch.exp(logvar),dim=1),dim=0)

    return Lrec + kld

#################################################################################




