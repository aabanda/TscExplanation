from matplotlib.collections import LineCollection
import numpy as np
import matplotlib.pyplot as plt

def plot_colormap(ts, weights):

    x = range(0,len(ts)-1)
    y = ts[1:]
    weights = weights[1:]
    weights = (weights- np.min(weights))/(np.max(weights)-np.min(weights))

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    fig, axs = plt.subplots(figsize=(6,4))
    norm = plt.Normalize(weights.min(), weights.max())
    lc = LineCollection(segments, cmap='jet', norm=norm)
    lc.set_array(weights)
    lc.set_linewidth(2)
    line = axs.add_collection(lc)

    cb = fig.colorbar(line, ax=axs)
    cb.ax.tick_params(labelsize=15)
    axs.set_xlim(-0.5, len(x)+0.5)
    axs.set_ylim(np.min(ts)-0.5,np.max(ts)+0.5)

    plt.show()
    plt.tight_layout()


