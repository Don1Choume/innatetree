from matplotlib import pyplot as plt
import numpy as np

def plot_Lorenz(srs_list, labels=[], savename=None):
    fig = plt.figure(figsize=(10, 8), constrained_layout=True)
    gs = fig.add_gridspec(4, 3)
    ax0 = []
    ax0.append(fig.add_subplot(gs[0, 0]))
    for srs in srs_list:
        ax0[-1].plot(srs[:, 0], srs[:, 1])
    ax0[-1].set_title('x vs y')
    ax0[-1].grid(linestyle=':')
    ax0
    ax0.append(fig.add_subplot(gs[0, 1]))
    for srs in srs_list:
        ax0[-1].plot(srs[:, 1], srs[:, 2])
    ax0[-1].set_title('y vs z')
    ax0[-1].grid(linestyle=':')
    ax0.append(fig.add_subplot(gs[0, 2]))
    for srs in srs_list:
        ax0[-1].plot(srs[:, 2], srs[:, 0])
    ax0[-1].set_title('z vs x')
    ax0[-1].grid(linestyle=':')
    
    ax1 = fig.add_subplot(gs[1, :])
    for srs in srs_list:
        ax1.plot(np.arange(len(srs[:, 0])), srs[:, 0])
    ax1.set_title('x time series')
    ax1.grid(linestyle=':')
    
    ax2 = fig.add_subplot(gs[2, :])
    for srs in srs_list:
        ax2.plot(np.arange(len(srs[:, 1])), srs[:, 1])
    ax2.set_title('y time series')
    ax2.grid(linestyle=':')
    
    ax3 = fig.add_subplot(gs[3, :])
    if len(labels):
        for srs, lbl in zip(srs_list, labels):
            ax3.plot(np.arange(len(srs[:, 2])), srs[:, 2], label=lbl)
    else:
        for srs in srs_list:
            ax3.plot(np.arange(len(srs[:, 2])), srs[:, 2])
    ax3.set_title('z time series')
    ax3.grid(linestyle=':')
    ax3.set_xlabel('step')
    if len(labels):
        ax3.legend(loc='best')
    
    if savename is not None:
        plt.savefig(savename)
        plt.clf()
        plt.close()
    else:
        plt.show()