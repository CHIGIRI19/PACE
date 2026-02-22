import os, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.color_palette("bright")
from .constants import IMGS_DIR
from matplotlib.lines import Line2D


def plot_comparision(
    df, generated, trajectories,
    palette = 'viridis', df_time_key='samples',
    save=False, path=IMGS_DIR, file='comparision.png',
    x='d1', y='d2', z='d3', is_3d=False
):
    if not os.path.isdir(path):
        os.makedirs(path)

    if not is_3d:
        return new_plot_comparisions(
            df, generated, trajectories,
            palette=palette, df_time_key=df_time_key,
            x=x, y=y, z=z, is_3d=is_3d,            
            groups=None,
            save=save, path=path, file=file,
        )

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import torch

def new_plot_comparisions(
    df, generated, trajectories,
    palette = 'viridis',
    df_time_key='samples',
    x='x1', y='x2', z='x3', 
    groups=None,
    save=False, path=IMGS_DIR, file='comparision.png',
    is_3d=False
):
    if groups is None:
        groups = sorted(df[df_time_key].unique())
    cmap = plt.cm.viridis
    sns.set_palette(palette)
    plt.rcParams.update({
        'axes.prop_cycle': plt.cycler(color=cmap(np.linspace(0, 1, len(groups) + 1))),
        'axes.axisbelow': False,
        'axes.edgecolor': 'lightgrey',
        'axes.facecolor': 'None',
        'axes.grid': False,
        'axes.labelcolor': 'dimgrey',
        'axes.spines.right': False,
        'axes.spines.top': False,
        'figure.facecolor': 'white',
        'lines.solid_capstyle': 'round',
        'patch.edgecolor': 'w',
        'patch.force_edgecolor': True,
        'text.color': 'dimgrey',
        'xtick.bottom': False,
        'xtick.color': 'dimgrey',
        'xtick.direction': 'out',
        'xtick.top': False,
        'ytick.color': 'dimgrey',
        'ytick.direction': 'out',
        'ytick.left': False,
        'ytick.right': False, 
        'font.size':12, 
        'axes.titlesize':10,
        'axes.labelsize':12
    })

    n_cols = 1
    n_rols = 1

    grid_figsize = [12, 8]
    dpi = 300
    grid_figsize = (grid_figsize[0] * n_cols, grid_figsize[1] * n_rols)
    fig = plt.figure(None, grid_figsize, dpi=dpi)

    hspace = 0.3
    wspace = None
    gspec = plt.GridSpec(n_rols, n_cols, fig, hspace=hspace, wspace=wspace)

    outline_width = (0.3, 0.05)
    size = 300
    bg_width, gap_width = outline_width
    point = np.sqrt(size)

    gap_size = (point + (point * gap_width) * 2) ** 2
    bg_size = (np.sqrt(gap_size) + (point * bg_width) * 2) ** 2

#    plt.legend(frameon=False)

    is_3d = False
    
    # if is_3d:        
    #     ax = fig.add_subplot(1,1,1,projection='3d')
    # else:
    #     ax = fig.add_subplot(1,1,1)
    
    axs = []
    for i, gs in enumerate(gspec):        
        ax = plt.subplot(gs)
        
        
        n = 0.3   
        ax.scatter(
                df[x], df[y],
                c=df[df_time_key],
                s=size,
                alpha=0.7 * n,
                marker='X',
                linewidths=0,
                edgecolors=None,
                cmap=cmap
            )
        
        for trajectory in np.transpose(trajectories, axes=(1,0,2)):
                plt.plot(trajectory[:, 0], trajectory[:, 1], alpha=0.3, color='Black');
        
        states = sorted(df[df_time_key].unique())
        points = np.concatenate(generated, axis=0)
        n_gen = int(points.shape[0] / len(states))
        colors = [state for state in states for i in range(n_gen)]
        n = 1
        o = '.'
        ax.scatter(
                points[:, 0], points[:, 1],
                c='black',
                s=bg_size,
                alpha=1 * n,
                marker=o,
                linewidths=0,
                edgecolors=None
            )
        ax.scatter(
                points[:, 0], points[:, 1],
                c='white',
                s=gap_size,
                alpha=1 * n,
                marker=o,
                linewidths=0,
                edgecolors=None
            )
        pnts = ax.scatter(
                points[:, 0], points[:, 1],
                c=colors,
                s=size,
                alpha=0.7 * n,
                marker=o,
                linewidths=0,
                edgecolors=None,
                cmap=cmap
            )
                
        legend_elements = [        
            Line2D(
                [0], [0], marker='o', 
                color=cmap((i) / (len(states)-1)), label=f'T{state}', 
                markerfacecolor=cmap((i) / (len(states)-1)), markersize=15,
            )
            for i, state in enumerate(states)
        ]
        
        leg = plt.legend(handles=legend_elements, loc='upper left')
        ax.add_artist(leg)
        
        legend_elements = [        
            Line2D(
                [0], [0], marker='X', color='w', 
                label='Ground Truth', markerfacecolor=cmap(0), markersize=15, alpha=0.3
            ),
            Line2D([0], [0], marker='o', color='w', label='Predicted', markerfacecolor=cmap(.999), markersize=15),
            Line2D([0], [0], color='black', lw=2, label='Trajectory')
            
        ]
        leg = plt.legend(handles=legend_elements, loc='upper right')
        ax.add_artist(leg)
        
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.get_xaxis().get_major_formatter().set_scientific(False)
        ax.get_yaxis().get_major_formatter().set_scientific(False)
        kwargs = dict(bottom=False, left=False, labelbottom=False, labelleft=False)
        ax.tick_params(which="both", **kwargs)
        ax.set_frame_on(False)
        ax.patch.set_alpha(0)
        

        axs.append(ax)

    if save:
        # NOTE: savefig complains image is too large but saves it anyway. 
        try:
            fig.savefig(os.path.expanduser(os.path.join(path, file)))
        except ValueError:
            pass 
    plt.close()
    return fig

