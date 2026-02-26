import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
from IPython.display import HTML, display

def animate(V, ae, phase_indices, phase_range, sweep_index, sweep_range, fix_index, fix_value=0., 
            n_per_axis=100, fps=30):
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
    VV = V(ae.encode(torch.tensor(X).float().reshape(n_per_axis*n_per_axis, -1))).reshape(n_per_axis, n_per_axis).cpu().detach().numpy()
    cm = sns.color_palette("Set2")
    contour = ax.contourf(XX, YY, VV, levels=[0, 1e-3], colors=['red'])


    def update(frame):
        nonlocal contour
        axes = 4 * [None]
        for idx, grid in zip(phase_indices, [XX,YY]):
            axes[idx] = grid
        axes[sweep_index] = sweep_vals[frame]*np.ones((n_per_axis, n_per_axis))
        axes[fix_index] = fix_value*np.ones((n_per_axis, n_per_axis))
        X = np.dstack(axes)
        VV = V(ae.encode(torch.tensor(X).float().reshape(n_per_axis*n_per_axis, -1))).reshape(n_per_axis, n_per_axis).cpu().detach().numpy()
        cm = sns.color_palette("Set1")

        contour.remove()
         
        #contour = ax.contourf(XX, YY, VV, levels=[0, 0.1, 100, 1000], colors=['red', 'blue', 'green'])
        contour = ax.contourf(XX, YY, VV, levels=[0, 0.01, .05, 0.1, 1, 10, 100, 1000], 
                              colors=['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'black'])

        return contour

    ani = animation.FuncAnimation(fig, update, frames=n_frames,
                                  blit=False, interval=1000/fps)
    plt.close(fig)
    print("Generating animation...")
    return display(HTML(ani.to_jshtml()))
