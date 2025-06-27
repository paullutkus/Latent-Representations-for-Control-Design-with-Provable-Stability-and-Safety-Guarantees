import numpy as np
import scipy
import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint
import params
from cartpole import dxdt
#from torchdiffeq import odeint



# for tracking gradients through f_z
class LatentDynamics(nn.Module):
    def __init__(self, fdyn, U):
        super().__init__()
        self.fdyn = fdyn
        self.U = U


    def forward(self, t, Z):
        return self.fdyn(Z, self.U)



# discretize continuous-time system by evolving over fixed interval,
# with zero-order-hold control input
def flow(x0, t_end, N, U, method='RK45'):
    t_eval = np.linspace(0, t_end, N)
    fxu = lambda t, x: dxdt(x, U[int((t/t_end)*N)])
    result = scipy.integrate.solve_ivp(fxu, (0, t_end), y0=x0, t_eval=t_eval, method=method)
    return result.y.T


# discrete continuous-time system by evolving over fixed interval,
# with constant control input
def _flow(x0, t_end, u):
    fxu = lambda t, x: dxdt(x, u)
    result = scipy.integrate.solve_ivp(fxu, (0,t_end), y0=x0)
    return result.y.T


# flow the system using differentiable odeint,
# fixed control input
def _flow_diff(f, x0, u, t_end, method=None):
    fxu = LatentDynamics(f, u)
    t_eval = torch.tensor([0,t_end])
    if method is not None:
        xt = odeint(fxu, x0, t_eval, method=method) 
    else:
        xt = odeint(fxu, x0, t_eval, method=params.ode_method)
    return xt
