from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.animation as animation
import seaborn as sns
from IPython.display import HTML, display
import params 
from utils import rollout_parallel_rk4
from controls import LQR

def solo_roa_plot(V, ae, roa=False, roa_per_axis=None, fdyn=None, lqr=None, save_traj=False, n_chunks=0, X_test=None):

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_title("Region of Attraction\n v, w in [-3, 3]")
    ax.set_ylabel("Theta")
    ax.set_xlabel("Position")

    roa_axes_x_chunks = np.linspace(-roa[0], roa[0], n_chunks+1)
    X_roa_flatten_vw_chunks = []
    X_roa_success_alpha_chunks = []
    for chunk_idx in range(n_chunks):
        roa_axes = [np.linspace(roa_axes_x_chunks[chunk_idx], roa_axes_x_chunks[chunk_idx+1], roa_per_axis[0] // n_chunks)]
        ### THIS IS THE PROBLEM ###
        for ii in range(1, params.d_x):
            roa_axes.append(np.linspace(-roa[ii], roa[ii], roa_per_axis[ii]))
        roa_pts = np.stack(np.meshgrid(*roa_axes, indexing='ij'), axis=-1)
        roa_og_shape = roa_pts.shape

        ### Rollout Trajectories ###
        roa_pts = roa_pts.reshape(-1, params.d_x)
        T_roa = 3500
        #X_roa, _, _, _ = rollout_trajectories(ae, fdyn, lqr, roa_pts, T=T_roa, plot=True)
        roa_pts = roa_pts.astype('float32')
        X_roa = rollout_parallel_rk4(ae, fdyn, lqr, roa_pts, T_roa, save=save_traj)
        if not save_traj:
            T_roa=1
        #for X_traj in X_roa:
        #    ax.plot(X_traj[:,0], X_traj[:,1])
        #print("comparison", np.sum((X_roa[:,0].reshape(roa_og_shape) - roa_pts.reshape(roa_og_shape))**2))

        ### Swap v and theta axes ###
        X_roa_xth_index = np.transpose(X_roa.reshape(roa_og_shape[:-1]+(-1, params.d_x)), (0, 2, 1, 3, 4, 5))
        X_roa_flatten_vw = X_roa_xth_index.reshape(X_roa_xth_index.shape[0], 
                                                   X_roa_xth_index.shape[1], 
                                                   X_roa_xth_index.shape[2]*X_roa_xth_index.shape[3],
                                                   T_roa+1,
                                                   params.d_x)
        X_roa_final_pos = X_roa_flatten_vw[:,:,:,-1,0]
        X_roa_init_pos  = X_roa_flatten_vw[:,:,:, 0,0]
        X_roa_final_angle= X_roa_flatten_vw[:,:,:,-1, 2]
        X_roa_final_w= X_roa_flatten_vw[:,:,:,-1, 3]

        X_roa_success_bool = np.logical_and((np.abs(X_roa_final_pos) <= roa[0]/2), (np.abs(X_roa_final_angle) <= 0.1)) #roa[0] / 2
        X_roa_success_bool = np.logical_and(X_roa_success_bool, (np.abs(X_roa_final_w) <= 0.1))
        X_roa_success_alpha = np.sum(X_roa_success_bool, axis=2) / X_roa_success_bool.shape[-1]

        X_roa_flatten_vw_chunks.append(X_roa_flatten_vw)
        X_roa_success_alpha_chunks.append(X_roa_success_alpha)

    for ii in range(n_chunks):
        X_roa_flatten_vw = X_roa_flatten_vw_chunks[ii]
        X_roa_success_alpha = X_roa_success_alpha_chunks[ii]
        ax.scatter(X_roa_flatten_vw[:,:,0,0].reshape(-1,4)[:,0], 
                   X_roa_flatten_vw[:,:,0,0].reshape(-1,4)[:,2],
                   color='b', alpha=X_roa_success_alpha.reshape(-1))
    if X_test is not None:
        X_test = X_test.cpu().detach().numpy()
        for X_test_traj in tqdm(X_test[:15000]):
            ax.plot(X_test_traj[:,0], X_test_traj[:,2])

    plt.show()


def animate_cartpole(trajectory, 
                     pole_length=1.0, 
                     cart_width=0.5, 
                     cart_height=0.25, 
                     view_width=10.0,
                     step=1,
                     fps=30):
    
    # [Function code is identical to previous response...]
    
    # 1. Data Preparation
    traj_tmp = trajectory[:,1]
    trajectory[:,1] = trajectory[:,2]
    trajectory[:,2] = traj_tmp
    trajectory_np = np.array(trajectory[::step, :2])
    num_frames = len(trajectory_np)
    
    # 2. Setup the Figure and Axes
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_ylim(-cart_height * 2, pole_length * 1.5)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.6)
    y_cart = 0.0

    # 3. Create Initial Artists
    x0, theta0 = trajectory_np[0]
    cart = Rectangle((x0 - cart_width / 2, y_cart - cart_height / 2),
                     cart_width, cart_height, fc='black')
    ax.add_patch(cart)
    x_pole_end_0 = x0 + pole_length * np.sin(theta0)
    y_pole_end_0 = y_cart + pole_length * np.cos(theta0)
    pole, = ax.plot([x0, x_pole_end_0], [y_cart, y_pole_end_0], 'brown', lw=4)
    pivot, = ax.plot([x0], [y_cart], 'bo', ms=8)
    track_line = ax.axhline(y_cart - cart_height / 2, color='gray', lw=2, zorder=-1)
    pos_text = ax.text(0.05, 0.95, '', 
                         transform=ax.transAxes, fontsize=12, 
                         verticalalignment='top')
    
    # 4. Define Animation Update Function
    def update(frame):
        x, theta = trajectory_np[frame]
        cart.set_x(x - cart_width / 2)
        x_pole_end = x + pole_length * np.sin(theta)
        y_pole_end = y_cart + pole_length * np.cos(theta)
        pole.set_data([x, x_pole_end], [y_cart, y_pole_end])
        pivot.set_data([x], [y_cart])
        #ax.set_xlim(x - view_width / 2, x + view_width / 2)
        ax.set_xlim(-4, 4)
        pos_text.set_text(f'Frame: {frame * step}\n'
                          f'x: {x:.2f} m\n'
                          f'θ: {np.degrees(theta):.2f}°')
        return cart, pole, pivot, pos_text

    # 5. Create and Display Animation
    ani = animation.FuncAnimation(fig, update, frames=num_frames,
                                  blit=False, interval=1000/fps)
    plt.close(fig)
    print("Generating animation... (This may take a moment)")
    return display(HTML(ani.to_html5_video()))
    #return display(HTML(ani.to_jshtml()))


def animate_dyn(V, ae, fdyn, phase_indices, phase_range, sweep_index, sweep_range, fix_index, fix_value=0., 
            n_per_axis=100, fps=30, encoder_idx=0, n_frames=500):

    lqr = LQR(ae, fdyn)
    X0_anim = torch.tensor([[0., 6., -1., -1.]])
    traj = rollout_parallel_rk4(ae, fdyn, lqr, X0_anim, save=True)[0]

    fig, ax = plt.subplots(figsize=(8, 8))
    y_ax = np.linspace(phase_range[0]+traj[0,phase_indices[0]], phase_range[1]+traj[0,phase_indices[0]], n_per_axis)
    x_ax = np.linspace(phase_range[2]+traj[0,phase_indices[1]], phase_range[3]+traj[0,phase_indices[1]], n_per_axis)
    XX, YY = np.meshgrid(x_ax, y_ax)

    n_frames = n_frames
    #sweep_vals = np.linspace(sweep_range[0], sweep_range[1], n_frames)

    axes = 4 * [None]
    for idx, grid in zip(phase_indices, [XX,YY]):
        axes[idx] = grid
    axes[sweep_index] = traj[0,sweep_index]*np.ones((n_per_axis, n_per_axis))
    axes[fix_index] = traj[0,fix_index]*np.ones((n_per_axis, n_per_axis))
    X = np.dstack(axes)
    #print(V(ae.encode(torch.tensor(X).float().reshape(n_per_axis*n_per_axis,-1))).shape)
    #VV = V(ae.encode(torch.tensor(X).float().reshape(n_per_axis*n_per_axis, -1))).reshape(n_per_axis, n_per_axis).cpu().detach().numpy()
    VV = ae.encode(torch.tensor(X).float().reshape(n_per_axis*n_per_axis, -1))[:,encoder_idx].reshape(n_per_axis, n_per_axis).cpu().detach().numpy()
    #cm = sns.color_palette("Set2")
    #contour = ax.contourf(XX, YY, VV, levels=[0, 1e-3], colors=['red'])
    contour = ax.contourf(XX, YY, VV, levels=[-10, -5, -1, -0.5, -0.1, -.05, -0.01, 
                                               0.0, 
                                               0.01, .05, 0.1, 0.5, 1, 5, 10], 
                          colors=['black', 'purple', 'blue', 'green', 'yellow', 'orange', 
                                  'red', 'red', 
                                  'orange', 'yellow', 'green', 'blue', 'purple'])#, 'black'])
    contourlines = ax.contour(XX, YY, VV, levels=contour.levels, colors='black', alpha=1., linewidths=0.)
    labels = ax.clabel(contourlines, inline=True, fontsize=10, fmt='%1.1f', colors='white')
    for label in labels:
        label.set_backgroundcolor('black')

    state_names = ['Position', 'Velocity', 'Theta', 'Angular Vel.']

    hline = ax.axhline(0, color='white', linewidth=1, zorder=5)
    vline = ax.axvline(0, color='white', linewidth=1, zorder=5)
    ax.set_title("E(x)[{0}]\n{1}: {2:.2f}, {3}: {4:.2f}".format(encoder_idx, state_names[sweep_index], traj[0,sweep_index], state_names[fix_index], traj[0,fix_index]))
    ax.set_ylabel(state_names[phase_indices[1]], fontsize=12)
    ax.set_xlabel(state_names[phase_indices[0]], fontsize=12)
    ax.set_ylim([traj[0,phase_indices[1]]+phase_range[2], traj[0,phase_indices[1]]+phase_range[3]])
    ax.set_xlim([traj[0,phase_indices[0]]+phase_range[0], traj[0,phase_indices[0]]+phase_range[1]])
    traj_plot = ax.plot(traj[0, phase_indices[0]], traj[0, phase_indices[1]], color='white')



    def update(frame):
        nonlocal contour
        nonlocal contourlines
        nonlocal hline
        nonlocal vline
        nonlocal traj_plot
        axes = 4 * [None]
        y_ax = np.linspace(phase_range[0]+traj[frame,phase_indices[0]], phase_range[1]+traj[frame,phase_indices[0]], n_per_axis)
        x_ax = np.linspace(phase_range[2]+traj[frame,phase_indices[1]], phase_range[3]+traj[frame,phase_indices[1]], n_per_axis)
        XX, YY = np.meshgrid(y_ax, x_ax)

        for idx, grid in zip(phase_indices, [XX,YY]):
            axes[idx] = grid

        axes[sweep_index] = traj[frame,sweep_index]*np.ones((n_per_axis, n_per_axis))
        axes[fix_index] = traj[frame,fix_index]*np.ones((n_per_axis, n_per_axis))

        X = np.dstack(axes)
        #VV = V(ae.encode(torch.tensor(X).float().reshape(n_per_axis*n_per_axis, -1))).reshape(n_per_axis, n_per_axis).cpu().detach().numpy()
        VV = ae.encode(torch.tensor(X).float().reshape(n_per_axis*n_per_axis, -1))[:,encoder_idx].reshape(n_per_axis, n_per_axis).cpu().detach().numpy()
        #cm = sns.color_palette("Set1")

        contour.remove()
         
        #contour = ax.contourf(XX, YY, VV, levels=[0, 0.1, 100, 1000], colors=['red', 'blue', 'green'])
        contour = ax.contourf(XX, YY, VV, levels=[-10, -5, -1.0, -0.5, -0.1, -.05, -0.01, 
                                                   0.0, 
                                                   0.01, .05, 0.1, 0.5, 1.0, 5, 10], 
                              colors=['black', 'purple', 'blue', 'green', 'yellow', 'orange', 
                                      'red', 'red', 
                                      'orange', 'yellow', 'green', 'blue', 'purple'])#, 'black'])
        contourlines.remove()
        contourlines = ax.contour(XX, YY, VV, levels=contour.levels, colors='black', alpha=1., linewidths=0.)
        labels = ax.clabel(contourlines, inline=True, fontsize=10, fmt='%1.1f', colors='white')
        for label in labels:
            label.set_backgroundcolor('black')
        ax.set_title("E(x)[{0}]\n{1}: {2:.3f}, {3}: {4:.3f}".format(encoder_idx, state_names[sweep_index], traj[frame,sweep_index], state_names[fix_index], traj[frame,fix_index]))
        ax.set_ylim([traj[frame,phase_indices[1]]+phase_range[2], traj[frame,phase_indices[1]]+phase_range[3]])
        ax.set_xlim([traj[frame,phase_indices[0]]+phase_range[0], traj[frame,phase_indices[0]]+phase_range[1]])

        traj_plot = ax.plot(traj[:frame, phase_indices[0]], traj[:frame,phase_indices[1]], color='white')



        vline.remove()
        hline.remove()
        hline = ax.axhline(traj[frame,phase_indices[1]], color='black', linewidth=1, zorder=5)
        vline = ax.axvline(traj[frame,phase_indices[0]], color='black', linewidth=1, zorder=5)


        return contour, contourlines, hline, vline

    ani = animation.FuncAnimation(fig, update, frames=n_frames,
                                  blit=False, interval=1000/fps)
    plt.close(fig)
    print("Generating animation...")
    #return display(HTML(ani.to_jshtml()))
    #return display(HTML(ani.to_jshtml()))
    #anim.save('animation.gif', writer=PillowWriter(fps=15))
    return display(HTML(ani.to_html5_video()))


def animate(V, ae, phase_indices, phase_range, sweep_index, sweep_range, fix_index, fix_value=0., 
            n_per_axis=100, fps=30, encoder_idx=0):
    fig, ax = plt.subplots(figsize=(8, 8))
    y_ax = np.linspace(phase_range[0], phase_range[1], n_per_axis)
    x_ax = np.linspace(phase_range[2], phase_range[3], n_per_axis)
    XX, YY = np.meshgrid(x_ax, y_ax)

    n_frames = n_per_axis
    sweep_vals = np.linspace(sweep_range[0], sweep_range[1], n_frames)

    axes = 4 * [None]
    for idx, grid in zip(phase_indices, [XX,YY]):
        axes[idx] = grid
    axes[sweep_index] = sweep_vals[0]*np.ones((n_per_axis, n_per_axis))
    axes[fix_index] = fix_value*np.ones((n_per_axis, n_per_axis))
    X = np.dstack(axes)
    #print(V(ae.encode(torch.tensor(X).float().reshape(n_per_axis*n_per_axis,-1))).shape)
    #VV = V(ae.encode(torch.tensor(X).float().reshape(n_per_axis*n_per_axis, -1))).reshape(n_per_axis, n_per_axis).cpu().detach().numpy()
    VV = ae.encode(torch.tensor(X).float().reshape(n_per_axis*n_per_axis, -1))[:,encoder_idx].reshape(n_per_axis, n_per_axis).cpu().detach().numpy()
    #cm = sns.color_palette("Set2")
    #contour = ax.contourf(XX, YY, VV, levels=[0, 1e-3], colors=['red'])
    contour = ax.contourf(XX, YY, VV, levels=[-10, -5, -1, -0.5, -0.1, -.05, -0.01, 
                                               0.0, 
                                               0.01, .05, 0.1, 0.5, 1, 5, 10], 
                          colors=['black', 'purple', 'blue', 'green', 'yellow', 'orange', 
                                  'red', 'red', 
                                  'orange', 'yellow', 'green', 'blue', 'purple'])#, 'black'])
    contourlines = ax.contour(XX, YY, VV, levels=contour.levels, colors='black', alpha=1., linewidths=0.)
    labels = ax.clabel(contourlines, inline=True, fontsize=10, fmt='%1.1f', colors='white')
    for label in labels:
        label.set_backgroundcolor('black')

    state_names = ['Position', 'Velocity', 'Theta', 'Angular Vel.']

    ax.axhline(0, color='white', linewidth=1, zorder=5)
    ax.axvline(0, color='white', linewidth=1, zorder=5)
    ax.set_title("E(x)[{0}]\n{1}: {2:.2f}, {3}: {4:.2f}".format(encoder_idx, state_names[sweep_index], sweep_vals[0], state_names[fix_index], fix_value))
    ax.set_ylabel(state_names[phase_indices[1]], fontsize=12)
    ax.set_xlabel(state_names[phase_indices[0]], fontsize=12)


    def update(frame):
        nonlocal contour
        nonlocal contourlines
        axes = 4 * [None]
        for idx, grid in zip(phase_indices, [XX,YY]):
            axes[idx] = grid
        axes[sweep_index] = sweep_vals[frame]*np.ones((n_per_axis, n_per_axis))
        axes[fix_index] = fix_value*np.ones((n_per_axis, n_per_axis))
        X = np.dstack(axes)
        #VV = V(ae.encode(torch.tensor(X).float().reshape(n_per_axis*n_per_axis, -1))).reshape(n_per_axis, n_per_axis).cpu().detach().numpy()
        VV = ae.encode(torch.tensor(X).float().reshape(n_per_axis*n_per_axis, -1))[:,encoder_idx].reshape(n_per_axis, n_per_axis).cpu().detach().numpy()
        #cm = sns.color_palette("Set1")

        contour.remove()
         
        #contour = ax.contourf(XX, YY, VV, levels=[0, 0.1, 100, 1000], colors=['red', 'blue', 'green'])
        contour = ax.contourf(XX, YY, VV, levels=[-10, -5, -1.0, -0.5, -0.1, -.05, -0.01, 
                                                   0.0, 
                                                   0.01, .05, 0.1, 0.5, 1.0, 5, 10], 
                              colors=['black', 'purple', 'blue', 'green', 'yellow', 'orange', 
                                      'red', 'red', 
                                      'orange', 'yellow', 'green', 'blue', 'purple'])#, 'black'])
        contourlines.remove()
        contourlines = ax.contour(XX, YY, VV, levels=contour.levels, colors='black', alpha=1., linewidths=0.)
        labels = ax.clabel(contourlines, inline=True, fontsize=10, fmt='%1.1f', colors='white')
        for label in labels:
            label.set_backgroundcolor('black')
        ax.set_title("E(x)[{0}]\n{1}: {2:.2f}, {3}: {4:.2f}".format(encoder_idx, state_names[sweep_index], sweep_vals[frame], state_names[fix_index], fix_value))

        return contour, contourlines

    ani = animation.FuncAnimation(fig, update, frames=n_frames,
                                  blit=False, interval=1000/fps)
    plt.close(fig)
    print("Generating animation...")
    #return display(HTML(ani.to_jshtml()))
    #return display(HTML(ani.to_jshtml()))
    #anim.save('animation.gif', writer=PillowWriter(fps=15))
    return display(HTML(ani.to_html5_video()))
