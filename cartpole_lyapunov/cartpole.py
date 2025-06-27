import math
import scipy
import numpy as np
import torch
import pygame
from pygame import gfxdraw
import params



# simulation parameters
DT = 0.02
CART_MASS = 0.5 
POLE_MASS = 0.05 
POLE_LEN = 0.5
X_BOX = [4*[-params.x_range],
         4*[ params.x_range]]
U_BOX = [-params.u_range, params.u_range]


# continuous-time RHS of \dot{x}=f(x,u) for cartpole 
def dxdt(q, u):
    assert (type(u) == float) or (u.shape == ())

    dt = DT
    g = 9.8
    m_c = CART_MASS 
    m_p = POLE_MASS 
    l = POLE_LEN
    
    x         = q[0]
    x_dot     = q[1]
    theta     = q[2]
    theta_dot = q[3]

    F = u
    
    sintheta = np.sin(theta)
    costheta = np.cos(theta)

    a = (-F - m_p*l*theta_dot**2 * sintheta) / (m_c + m_p)  
    b = 4.0/3.0 - m_p*costheta**2 / (m_c + m_p)
    
    theta_acc = (g*sintheta + costheta*a) / (l*b)
    theta_dot = theta_dot + dt*theta_acc 

    ### COMPUTE THETA_ACC BEFORE C, N ###
    
    c = theta_dot**2 * sintheta - theta_acc*costheta 
    
    x_acc = (F + m_p*l*c) / (m_c + m_p) 
    x_dot = x_dot + dt*x_acc
  
    q = np.array([x_dot, x_acc, theta_dot, theta_acc])
    return q


# vectorized version of cartpole RHS (for tracking gradients)
def dxdt_torch(q, u):
    dt = DT
    g = 9.8
    m_c = CART_MASS 
    m_p = POLE_MASS 
    l = POLE_LEN 

    x         = q[:,0]
    x_dot     = q[:,1]
    theta     = q[:,2]
    theta_dot = q[:,3]

    F = u.squeeze()

    sintheta = torch.sin(theta)
    costheta = torch.cos(theta)

    a = (-F - m_p*l*theta_dot**2 * sintheta) / (m_c + m_p)
    b = 4.0/3.0 - m_p*costheta**2 / (m_c + m_p)
        
    theta_acc = (g*sintheta + costheta*a) / (l*b)
    theta_dot = theta_dot + dt*theta_acc

    ### COMPUTE THETA_ACC BEFORE C, N ###

    c = theta_dot**2 * sintheta - theta_acc*costheta

    x_acc = (F + m_p*l*c) / (m_c + m_p)
    x_dot = x_dot + dt*x_acc

    q = torch.stack([x_dot, x_acc, theta_dot, theta_acc]).T
    return q


# non-vectorized version of previous function
def _dxdt_torch(q, u):
    dt = DT
    g = 9.8
    m_c = CART_MASS
    m_p = POLE_MASS 
    l = POLE_LEN 

    x         = q[0]
    x_dot     = q[1]
    theta     = q[2]
    theta_dot = q[3]

    F = u

    sintheta = torch.sin(theta)
    costheta = torch.cos(theta)

    a = (-F - m_p*l*theta_dot**2 * sintheta) / (m_c + m_p)
    b = 4.0/3.0 - m_p*costheta**2 / (m_c + m_p)
        
    theta_acc = (g*sintheta + costheta*a) / (l*b)
    theta_dot = theta_dot + dt*theta_acc

    ### COMPUTE THETA_ACC BEFORE C, N ###

    c = theta_dot**2 * sintheta - theta_acc*costheta

    x_acc = (F + m_p*l*c) / (m_c + m_p)
    x_dot = x_dot + dt*x_acc

    q = torch.stack((x_dot, x_acc, theta_dot, theta_acc))
    return q



# class for rendering cartpole (not designed for use with server)
class CartpoleRenderer:
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(self, state):
        self.screen = None
        self.screen_width = 600
        self.screen_height = 400
        self.clock = None
        self.isopen = True

        self.gravity = 9.8

        ### NOT USED ###
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        ################

        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )

    def render(self):
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode(
                (self.screen_width, self.screen_height)
            )

        if self.clock is None:
            self.clock = pygame.time.Clock()

        #PROBLEM
        world_width = self.x_threshold * 2
        scale = self.screen_width / world_width
        polewidth = 10.0
        polelen = scale * (2 * self.length)

        cartwidth = 50.0
        cartheight = 30.0

        #if self.state is None:
            #return None

        x = self.state

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = x[0] * scale + self.screen_width / 2.0  # MIDDLE OF CART
        carty = 100  # TOP OF CART
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

        gfxdraw.aacircle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.filled_circle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )

        gfxdraw.hline(self.surf, 0, self.screen_width, carty, (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        pygame.event.pump()
        self.clock.tick(self.metadata["render_fps"])
        pygame.display.flip()

    def close(self):
        if self.screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.isopen = False

