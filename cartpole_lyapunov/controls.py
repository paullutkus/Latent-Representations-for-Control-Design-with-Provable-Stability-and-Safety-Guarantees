import numpy as np
import cvxpy as cp
import torch
import torch.nn as nn
import params
import matplotlib
import matplotlib.pyplot as plt
from cartpole import _dxdt_torch
from matplotlib import ticker
from torch.func import jacrev, grad, vmap
from collections import OrderedDict
from lyapunov import LyapunovEllipse, LyapunovGeneral



def validate_mlp_lyapunov(Z, V, rho):
    Z = torch.tensor(Z).float().to("cuda")
    Zi = Z[:,:-1].reshape(-1, params.d_z)
    Zf = Z[:,1:].reshape(-1, params.d_z)
    Vf = V(Zf)
    Vi = V(Zi)
    n_pts = len(Zi)
    print(Vf - rho*Vi)
    print(torch.min(Vf-rho*Vi))
    return torch.sum(torch.sign(nn.functional.relu(Vf - rho*Vi))) / n_pts


def mlp_lyapunov_reparam(Z, epochs=500, lr=1e-4, plot=True, grid_dens=500, rho=0.5, z_eq=None):
    # 1) train lyapunov 
    # 2) plot levelsets using matplotlib 
    # 3) compute lipschitz constant by gridding

    use_pd_loss = False
    use_dyn_loss = True
    use_lb_loss = False
    use_grad_loss = False
    print("pd loss:", use_pd_loss)
    print("dyn loss:", use_dyn_loss)
    print("lb loss", use_lb_loss)
    print("grad loss", use_grad_loss)

    print("rho:", rho)

    c = 0.1
    print("c lb", c)

    l_dyn = 1 #10
    l_pd = 1
    l_lb = 1
    l_grad = 1e-6
    print("l_dyn", l_dyn)
    print("l_pd", l_pd)
    print("l_lb", l_lb)
    print("l_grad", l_grad)

    Z = torch.tensor(Z).float().to("cuda")

    Z_blob = Z.reshape(-1, params.d_z)
    Z1_max = torch.max(Z_blob[:,0]); Z1_max += 0.1*Z1_max
    Z1_min = torch.min(Z_blob[:,0]); Z1_min -= 0.1*torch.abs(Z1_min)
    Z2_max = torch.max(Z_blob[:,1]); Z2_max += 0.1*Z2_max
    Z2_min = torch.min(Z_blob[:,1]); Z2_min -= 0.1*torch.abs(Z2_min)
    Z1_pts = torch.linspace(Z1_min, Z1_max, grid_dens)
    Z2_pts = torch.linspace(Z2_min, Z2_max, grid_dens)
    ZZ, WW = torch.meshgrid(Z1_pts, Z2_pts)

    print("using grid data, grid_density:", grid_dens)

    grid_data = torch.dstack([ZZ, WW]).reshape(-1, params.d_z)

    features = 128 # suggested: 128
    print("features:", features)
    '''
    V  = nn.Sequential(OrderedDict([
            ('mlp1', nn.Linear(2, features)),
            ('act1', nn.Tanh()),
            ('mlp2', nn.Linear(features, features)),
            ('act2', nn.Tanh()),
            ('mlp3', nn.Linear(features, 4))
         ]))
    '''

    n_layers = 3 # suggested: 3
    V = LyapunovGeneral(n_layers, features, z_eq=z_eq) #4
    print(V)

    #V = Lyapunov()
    V_opt = torch.optim.AdamW(params=V.parameters(), lr=lr)

    
    def grad_loss(V, Z):
        Z = Z.reshape(-1, params.d_z)
        grads = vmap(jacrev(V))(Z)
        return torch.sum(torch.norm(grads, dim=1)**2)


    def dyn_loss(V, Z):
        Zi = Z[:,:-1].reshape(-1, params.d_z)
        Zf = Z[:,1:].reshape(-1, params.d_z)

        '''
        Li = V(Zi).reshape(-1, params.d_z, params.d_z)
        Lf = V(Zf).reshape(-1, params.d_z, params.d_z)
        Pi = Li @ torch.transpose(Li, 1, 2) + torch.eye(params.d_z).unsqueeze(0).expand(Zi.shape[0], -1, -1)
        Pf = Lf @ torch.transpose(Lf, 1, 2) + torch.eye(params.d_z).unsqueeze(0).expand(Zf.shape[0], -1, -1)

        Vf_lhs = torch.bmm(Pf, Zf.unsqueeze(-1))
        Vf = torch.bmm(Zf.unsqueeze(1), Vf_lhs).squeeze()

        Vi_lhs = torch.bmm(Pi, Zi.unsqueeze(-1))
        Vi = torch.bmm(Zi.unsqueeze(1), Vi_lhs).squeeze()
        '''
        Vi = V(Zi)
        Vf = V(Zf)

        return torch.sum(nn.functional.relu((Vf-rho*Vi))**2)

    def lb_loss(V, Z):
        return torch.sum(nn.functional.relu(c*torch.linalg.norm(Z, dim=1) - V(Z))**2)

    V.train()
    for i in range(epochs):
        V_opt.zero_grad()

        loss = torch.tensor([0.])
        if use_dyn_loss:
            loss += l_dyn*dyn_loss(V, Z) 
        if use_lb_loss:
            loss += l_lb * lb_loss(V, grid_data) 
        if use_grad_loss:
            loss += l_grad * grad_loss(V, Z)

        loss.backward()
        V_opt.step()

        if i % 1000 == 0:
            print("it {}:".format(i), loss)

    if plot:
        fig, ax = plt.subplots(1)
        fig.set_size_inches(10, 10)
        alpha = torch.max(V(Z_blob)).item()
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

    return V, rho, alpha, lip


def mlp_lyapunov(Z, epochs=500, lr=1e-4, plot=True, grid_dens=500, rho=0.5):
    # 1) train lyapunov 
    # 2) plot levelsets using matplotlib 
    # 3) compute lipschitz constant by gridding

    use_pd_loss = True
    use_dyn_loss = True
    use_lb_loss = True
    print("pd loss:", use_pd_loss)
    print("dyn loss:", use_dyn_loss)
    print("lb loss", use_lb_loss)

    print("rho:", rho)

    c = 0.02

    l_dyn = 10
    l_pd = 1
    l_lb = 1
    print("l_dyn", l_dyn)
    print("l_pd", l_pd)
    print("l_lb", l_lb)

    Z = torch.tensor(Z).float().to("cuda")

    Z_blob = Z.reshape(-1, params.d_z)
    Z1_max = torch.max(Z_blob[:,0])
    Z1_min = torch.min(Z_blob[:,0])
    Z2_max = torch.max(Z_blob[:,1])
    Z2_min = torch.min(Z_blob[:,1])
    Z1_pts = torch.linspace(Z1_min, Z1_max, grid_dens)
    Z2_pts = torch.linspace(Z2_min, Z2_max, grid_dens)
    ZZ, WW = torch.meshgrid(Z1_pts, Z2_pts)

    print("using grid data, grid_density:", grid_dens)

    grid_data = torch.dstack([ZZ, WW]).reshape(-1, params.d_z)

    features = 64
    print("features:", features)
    V  = nn.Sequential(OrderedDict([
            ('mlp1', nn.Linear(2, features)),
            ('act1', nn.GELU()),
            ('mlp2', nn.Linear(features, features)),
            ('act2', nn.GELU()),
            ('mlp3', nn.Linear(features, 1))
         ]))

    #V = Lyapunov()
    V_opt = torch.optim.AdamW(params=V.parameters(), lr=lr)

    def pd_loss(V, Z):
        Z = Z.reshape(-1, params.d_z)
        return torch.sum(nn.functional.relu(-V(Z))**2)

    def dyn_loss(V, Z):
        Zi = Z[:,:-1].reshape(-1, params.d_z)
        Zf = Z[:,1:].reshape(-1, params.d_z)
        Vi = V(Zi)
        Vf = V(Zf)
        return torch.sum(nn.functional.relu((Vf-rho*Vi))**2)

    def lb_loss(V, Z):
        return torch.sum(nn.functional.relu(c*torch.linalg.norm(Z, dim=1) - V(Z))**2)

    V.train()
    for i in range(epochs):
        V_opt.zero_grad()

        loss = torch.tensor([0.])
        if use_dyn_loss:
            loss += l_dyn*dyn_loss(V, Z) 
        if use_pd_loss:
            loss += l_pd * pd_loss(V, Z)
            loss += l_pd * pd_loss(V, grid_data) 
        if use_lb_loss:
            loss += l_lb * lb_loss(V, grid_data) 

        loss.backward()
        V_opt.step()

        if i % 1000 == 0:
            print("it {}:".format(i), loss)

    if plot:
        alpha = torch.max(V(Z_blob)).item()
        print("alpha", alpha)
        grads = vmap(jacrev(V),in_dims=0)(Z_blob)
        lip = torch.max(torch.linalg.norm(grads.squeeze(), dim=-1))
        print("Lipschitz constant", lip)
        VV = V(torch.dstack([ZZ, WW]).reshape(-1, params.d_z)).reshape(grid_dens, grid_dens)
        plt.title("Rho: {0:.2f}, Local Lipschitz Over Data: {1:.2f}".format(rho, lip.item()))
        plt.contourf(ZZ.cpu().detach().numpy(), WW.cpu().detach().numpy(), VV.cpu().detach().numpy(), levels=50)
        plt.colorbar()
        cntr = plt.contour(ZZ.cpu().detach().numpy(), WW.cpu().detach().numpy(), VV.cpu().detach().numpy(), [alpha], colors=['w'])
        plt.clabel(cntr, inline=True, colors=['w'], fontsize=14)
        for Zi in Z:
            plt.plot(Zi[:,0].cpu().detach().numpy(), Zi[:,1].cpu().detach().numpy(), alpha=0.25)
        plt.show()

    return V, rho, alpha, lip



def verify_lyapunov(Z, P, b, r):
    # r: radius outside of which to verify 
    N = Z.shape[0]
    dV = []
    for i in range(N):
        Z_bar = Z[i] - b.reshape(1, 2) # Center data
        norm_Z_bar = np.linalg.norm(Z_bar, axis=1)
        if (norm_Z_bar <= r).all() or\
            norm_Z_bar[0] <= r or norm_Z_bar[1] <= r:
            pass
        else:
            # Eliminate all pairs after first enters radius
            dZ_bar = np.linalg.norm(Z_bar, axis=1)
            if not (dZ_bar >= r).all():
                first_idx = [idx for (idx, val) in enumerate(norm_Z_bar <= r) if val][0]
                #print("first idx", first_idx)
                Z_bar = Z_bar[:first_idx]
                #Z_bar = Z_bar[:first_idx] if first_idx % 2 == 0 else Z_bar[:first_idx+1]
                #assert Z_bar.shape[0] % 2 == 0

            Z0_bar = Z_bar[:-1]
            Z1_bar = Z_bar[1:]
            Vf = (Z1_bar[:,np.newaxis,:] @ P[np.newaxis,...] @ Z1_bar[:,:,np.newaxis]).squeeze()
            Vi = (Z0_bar[:,np.newaxis,:] @ P[np.newaxis,...] @ Z0_bar[:,:,np.newaxis]).squeeze()
            if len((Vf - Vi).shape) == 0:
                dV.append(np.expand_dims((Vf - Vi), 0))
            else:
                dV.append(Vf - Vi)
    dV = np.concatenate(dV)
    return len(dV[dV > 0]) / len(dV)


def lyapunov_sdp(Z, hull_pts, b, r, rho, solver=None):
    N = Z.shape[0]
    P = cp.Variable((2, 2), symmetric=True)
    eps = cp.Variable((1, 1))
    #obj = -cp.log_det(P) + 1e4*eps ### MIN VOLUME
    obj = cp.trace(P)
    cons = []
    cons.append(P >> 0.001*np.eye(2))
    cons.append(eps == 0)

    ### MIN VOLUME
    #for i in range(hull_pts.shape[0]):
    #    cons.append(cp.sum_squares(P @ hull_pts[i].reshape(2, 1) - b.reshape(2, 1)) <= 1) # + or - ? 

    for i in range(N): 
        Z_bar = Z[i] - b.reshape(1, 2) # Center data

        # Eliminate all pairs after first enters radius
        norm_Z_bar = np.linalg.norm(Z_bar, axis=1)
        if (norm_Z_bar <= r).all() or\
            norm_Z_bar[0] <= r or norm_Z_bar[1] <= r:
            pass
        else:
            if not (norm_Z_bar >= r).all():
                first_idx = [idx for (idx, val) in enumerate(norm_Z_bar <= r) if val][0]
                Z_bar = Z_bar[:first_idx]
                #Z_bar = Z_bar[:first_idx] if first_idx % 2 == 0 else Z_bar[:first_idx+1]
                #assert Z_bar.shape[0] % 2 == 0

            Z0_bar = Z_bar[:-1]
            Z1_bar = Z_bar[1:]
            plt.plot(Z_bar[:,0], Z_bar[:,1])
            for (z0_bar, z1_bar) in zip(Z0_bar, Z1_bar):
                cons.append(cp.quad_form(z1_bar, P) - rho * cp.quad_form(z0_bar, P) - eps <= 0) # ADD CONSTANT TO FIRST TERM
    plt.show()
                    
        #cons.append(cp.trace((Z1_bar @ Z1_bar.reshape(1, 2) -\
        #                      Z0_bar @ Z0_bar.reshape(1, 2)) @ P) <= 0)

    ### PROBLEM IS THAT THE DATA CONTAINS PAIRS OF ZEROS!!! ###
    prob = cp.Problem(cp.Minimize(obj), cons)
    prob.solve(verbose=True, solver=solver)
    return P.value


def learn_lyapunov(Z, V0, b):
    N = Z.shape[0]
    T = Z.shape[1]
    d = Z.shape[2]

    Z = torch.tensor(Z).float()
    Z0 = Z[:,:-1]
    Z1 = Z[:,1:]

    eps = 5e-4

    def L(V):
        Z0re = Z0.reshape(-1, d)
        Z1re = Z1.reshape(-1, d)
        #print("tst1", (Z0re.unsqueeze(1) - b.reshape(1, 1, d)).shape)
        #print("tst2", (Z0re.unsqueeze(-1) - b.reshape(1, 1, d)).shape)
        VZ0 = (Z0re.unsqueeze(1) - b.reshape(1, 1, d)) @ torch.linalg.inv(V @ V).reshape(1, d, d) @ (Z0re.unsqueeze(-1) - b.reshape(1, d, 1))
        VZ1 = (Z1re.unsqueeze(1) - b.reshape(1, 1, d)) @ torch.linalg.inv(V @ V).reshape(1, d, d) @ (Z1re.unsqueeze(-1) - b.reshape(1, d, 1))
        #print("V diff shape", (VZ1 - VZ0).shape)
        return torch.sum(VZ1 - VZ0 - (VZ1 + VZ0))

    V = V0
    #print(V.shape)
    #print(DV.shape)
    for t in range(5000):
        if L(V) < -15:
            break
        if t % 100 == 0:
            print(L(V))
        DV = grad(L)(V)
        V = V - eps*DV
    return V

def get_LQR_params(ae, fdyn, ret_AB=False, original_system=False, u_cost=1):
    if not original_system:
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
        elif not (params.enc_true_dyn or params.reverse_ae):
            fdyn.eval()
    with torch.no_grad():
        if not original_system:
            x_0 = torch.unsqueeze(torch.tensor([0., 0., 0., 0.]), 0)
            z_0 = ae.encode(x_0)
            z_0 = torch.tensor(z_0.cpu().detach().numpy())
            u_0 = torch.tensor([[0.]])
        else:
            z_0 = torch.tensor([0.,0.,0.,0.,]).unsqueeze(0)

        if original_system:
            f = _dxdt_torch
            A = jacrev(f, argnums=0)(torch.tensor([0.,0.,0.,.0]), torch.tensor(0.))
            B = jacrev(f, argnums=1)(torch.tensor([0.,0.,0.,.0]), torch.tensor(0.)).unsqueeze(-1)
            A = torch.eye(params.d_x) + A
        elif params.control_affine:
            f_wrap = lambda z, u: fdyn(torch.hstack((z, u)))
            grad_drift = torch.autograd.functional.jacobian(fdyn_drift, z_0)
            grad_cntrl = fdyn_cntrl(z_0).T
            A = grad_drift.squeeze()
            B = grad_cntrl
        elif params.linear_state_space:
            #print("fdyn at eq.", z_0.squeeze()-(fdyn_drift(z_0).reshape(params.d_z, params.d_z) @ z_0.reshape(-1,1) +\
            #                     fdyn_cntrl(z_0).reshape(params.d_z, params.d_u) @ u_0).squeeze())
            A = fdyn_drift(z_0).reshape(params.d_z, params.d_z)
            B = fdyn_cntrl(z_0).reshape(params.d_z, params.d_u)
            #print(A)
            #print(B)
        else:
            #f_wrap = lambda z, u: fdyn(torch.hstack((z, u)))
            #gradf = torch.autograd.functional.jacobian(f_wrap, (z_0, u_0))
            #print("fdyn at eq.", z_0 - fdyn(torch.cat((z_0[0], u_0[0]), dim=0)))
            #print(torch.cat((z_0[0], u_0[0]), dim=0).shape)
            jac_xu = jacrev(fdyn)( torch.cat((z_0[0], u_0[0]), dim=0) )
            #print("jac_xu.shape", jac_xu.shape)
            A = jac_xu[:,:-1]
            B = jac_xu[:,-1].unsqueeze(-1)
            #print("A shape", A.shape)
            #print("B shape", B.shape)
            #A = gradf[0].squeeze()
            #B = torch.unsqueeze(gradf[1].squeeze(), -1)
            '''
            if not ret_AB:
                print("z_0 shape", z_0.shape)
                print("u_0 shape", u_0.shape)
                print("fdyn input shape", torch.hstack((z_0,u_0)).shape)
                print("fdyn output shape", fdyn(torch.hstack((z_0,u_0))).shape)
                print("gradf[0] shape", gradf[0].shape)
                print("gradf[1] shape", gradf[1].shape)
            '''
        if params.learn_residual and not original_system:
            A = torch.eye(params.d_z) + A

        if original_system:
            Q = 1*torch.eye(params.d_x)
            R = u_cost*torch.eye(params.d_u) #1e-3, # PREV USING THIS: 1
            N = 0*torch.ones((params.d_x, params.d_u))
        else:
            Q = 1*torch.eye(params.d_z)
            R = u_cost*torch.eye(params.d_u) #1e-3, # PREV USING THIS: 1
            N = 0*torch.ones((params.d_z, params.d_u))

        
        if not ret_AB:
            print("A shape", A.shape)
            print("B shape", B.shape)
            print("Q shape", Q.shape)
            print("R shape", R.shape)
            print("A\n", A)
            print("B\n", B)

        P_f = Q
        P_t = [P_f]
        for i in range(200): #1e4
            P_next = A.T@P_t[-1]@A - (A.T@P_t[-1]@B + N) @ torch.linalg.inv(R + B.T@P_t[-1]@B) @ (B.T@P_t[-1]@A + N.T) + Q
            P_t.append(P_next)

        P = P_t[-1]
        #print("P\n", P)
        F = torch.linalg.inv(R + B.T@P@B) @ (B.T@P@A + N.T)
        #print("F\n", F)

        #LQR = lambda z: -F @ (z-z_0).T
    if ret_AB:
        return F, z_0, A, B, Q, R
    P = A - B @ F
    e = torch.linalg.eigvals(P)
    print("eigenvalues:\n", e)
    return F, z_0, Q, R


class LQR(nn.Module):
    def __init__(self, ae, fdyn, original_system=False, u_cost=1):
        super().__init__()
        self.ae = ae
        self.fdyn = fdyn
        self.F, self.z_0, self.A, self.B, self.Q, self.R = get_LQR_params(ae, fdyn, ret_AB=True, original_system=original_system, u_cost=u_cost)
        #self.F.requires_grad = False
        ### THE USE OF DETACH HERE IS A PROBLEM
        self.F = self.F.cpu().detach()
        self.z_0.requires_grad = False
        self.u_cost = u_cost

    def forward(self, z):
        return -self.F.cpu() @ (z.cpu() - self.z_0.cpu()).T
        #return (z-self.z_0).T



'''
def get_LQR(ae, f_dyn):
    if params.control_affine:
        fdyn_drift, fdyn_cntrl = fdyn

    x_0 = torch.unsqueeze(torch.tensor([0., 0., 0., 0.]), 0)
    z_0 = ae.encode(x_0)
    u_0 = torch.tensor([[0.]])

    print("z_0 shape", z_0.shape)
    print("u_0 shape", u_0.shape)
    if params.control_affine:
        print("f_dyn input shape", torch.hstack((z_0,u_0)).shape)
        #print("f_dyn output shape", f_dyn(torch.hstack((z_0,u_0))).shape)
        f_wrap = lambda z, u: f_dyn(torch.hstack((z, u)))
        grad_drift = torch.autograd.functional.jacobian(fdyn_drift, z_0)
        grad_cntrl = fdyn_cntrl(z_0)
        print(grad_drift.shape)
        print(grad_cntrl.shape)
        #gradf = torch.autograd.functional.jacobian(f_wrap, (z_0, u_0))
        #print("gradf[0] shape", gradf[0].shape)
        #print("gradf[1] shape", gradf[1].shape)
        A = gradf[0].squeeze()
        B = torch.unsqueeze(gradf[1].squeeze(), -1)
    else:
        print("f_dyn input shape", torch.hstack((z_0,u_0)).shape)
        print("f_dyn output shape", f_dyn(torch.hstack((z_0,u_0))).shape)
        f_wrap = lambda z, u: f_dyn(torch.hstack((z, u)))
        gradf = torch.autograd.functional.jacobian(f_wrap, (z_0, u_0))
        print("gradf[0] shape", gradf[0].shape)
        print("gradf[1] shape", gradf[1].shape)
        A = gradf[0].squeeze()
        B = torch.unsqueeze(gradf[1].squeeze(), -1)

    Q = torch.eye(params.d_z)
    R = torch.zeros((params.d_u, params.d_u))

    print("A shape", A.shape)
    print("B shape", B.shape)
    print("Q shape", Q.shape)
    print("R shape", R.shape)

    P_f = Q
    P_t = [P_f]
    for i in range(100):
        P_next = A.T@P_t[-1]@A - (A.T@P_t[-1]@B) @ torch.linalg.inv(R + B.T@P_t[-1]@B) @ (B.T@P_t[-1]@A) + Q
        P_t.append(P_next)

    P = P_t[-1]
    F = torch.linalg.inv(R + B.T@P@B) @ (B.T@P@A)

    LQR = lambda z: -F @ (z-z_0).T
    return LQR
'''



