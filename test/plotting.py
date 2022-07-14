import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec #as gs
import pandas as pd

def make_plot_full_blend2_compact_rescale(deblended, residual, seq_remain, seq_pred, seq_true, true_x, true_y, savedir, batch):
    """
    Plots paneled figure of preblended, blended and deblended galaxies.
    """
    for i in range(true_x.shape[0]):
        fig = plt.Figure()
        x = len(residual)
        gs = GridSpec(3, x)

        for idx in range(2):
            ax = fig.add_subplot(gs[2, idx])
            ax.set_xticks([])
            ax.set_yticks([])
            if idx == 0:
                ax.imshow(true_x[0]/true_x[0].max())
                ax.text(3., 10., '%.3f'%true_x[0].max(), size=8, color='w')
                ax.text(72., 75., str(seq_true[0]), size=8, color='w')
                ax.text(0.5, -0.1, 'Preblended 1', size=8, ha='center', transform=ax.transAxes)
            if idx == 1:
                ax.imshow(true_y[0]/true_y[0].max())
                ax.text(3., 10., '%.3f'%true_y[0].max(), size=8, color='w')
                ax.text(72., 75., str(seq_true[1]), size=8, color='w')
                ax.text(0.5, -0.1, 'Preblended 2', size=8, ha='center', transform=ax.transAxes)

        for idx in range(x):
            ax1 = fig.add_subplot(gs[0, idx])
            if idx != (x-1):
                ax2 = fig.add_subplot(gs[1, idx])

            if idx == (x-1):
                ax1.imshow(residual[idx]/residual[idx].max())
                ax1.text(3., 10., '%.3f'%residual[idx].max(), size=8, color='w')
                ax1.text(0.5, -0.1, 'Residual '+ str(int(idx)), size=8, ha='center', transform=ax1.transAxes)
                ax1.text(72., 75., str(seq_remain[idx]), size=8, color='w')
                ax1.set_xticks([])
                ax1.set_yticks([])
            else:
                ax1.imshow(residual[idx]/residual[idx].max())
                ax1.text(3., 10., '%.3f'%residual[idx].max(), size=8, color='w')
                ax1.text(72., 75., str(seq_remain[idx]), size=8, color='w')
                if idx == 0:
                    ax1.text(0.5, -0.1, 'Blended', size=8, ha='center', transform=ax1.transAxes)
                else:
                    ax1.text(0.5, -0.1, 'Residual '+ str(int(idx)), size=8, ha='center', transform=ax1.transAxes)
                ax2.imshow(deblended[idx]/deblended[idx].max())
                ax2.text(3., 10., '%.3f'%deblended[idx].max(), size=8, color='w')
                ax2.text(72., 75., str(seq_pred[idx]), size=8, color='w')
                ax2.text(0.5, -0.1, 'Deblended '+ str(int(idx+1)), size=8, ha='center', transform=ax2.transAxes)
                ax1.set_xticks([])
                ax1.set_yticks([])
                ax2.set_xticks([])
                ax2.set_yticks([])

        fig.tight_layout()
        fig.subplots_adjust(wspace=-0.55,hspace=0.15)
        if not os.path.isdir(savedir):
            os.makedirs(savedir)
        filename = os.path.join(savedir, 'test-{}-{}.png'.format(batch, i))
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)

def make_plot_full_blend3_compact_rescale(deblended, residual, seq_remain, seq_pred, seq_true, true_x, true_y, true_z, savedir, batch):
    for i in range(true_x.shape[0]):
        fig = plt.Figure()
        x = len(residual)
        gs = GridSpec(3, x)

        for idx in range(3):
            ax = fig.add_subplot(gs[2, idx])
            ax.set_xticks([])
            ax.set_yticks([])
            if idx == 0:
                ax.imshow(true_x[0]/true_x[0].max())
                ax.text(3., 10., '%.3f'%true_x[0].max(), size=8, color='w')
                ax.text(72., 75., str(seq_true[0]), size=8, color='w')
                ax.text(0.5, -0.1, 'Preblended 1', size=8, ha='center', transform=ax.transAxes)
            if idx == 1:
                ax.imshow(true_y[0]/true_y[0].max())
                ax.text(3., 10., '%.3f'%true_y[0].max(), size=8, color='w')
                ax.text(72., 75., str(seq_true[1]), size=8, color='w')
                ax.text(0.5, -0.1, 'Preblended 2', size=8, ha='center', transform=ax.transAxes)
            if idx == 2:
                ax.imshow(true_z[0]/true_z[0].max())
                ax.text(3., 10., '%.3f'%true_z[0].max(), size=8, color='w')
                ax.text(72., 75., str(seq_true[2]), size=8, color='w')
                ax.text(0.5, -0.1, 'Preblended 3', size=8, ha='center', transform=ax.transAxes)

        for idx in range(x):
            ax1 = fig.add_subplot(gs[0, idx])
            if idx != (x-1):
                ax2 = fig.add_subplot(gs[1, idx])

            if idx == (x-1):
                ax1.imshow(residual[idx]/residual[idx].max())
                ax1.text(3., 10., '%.3f'%residual[idx].max(), size=8, color='w')
                ax1.text(0.5, -0.1, 'Residual '+ str(int(idx)), size=8, ha='center', transform=ax1.transAxes)
                ax1.text(72., 75., str(seq_remain[idx]), size=8, color='w')
                ax1.set_xticks([])
                ax1.set_yticks([])
            else:
                ax1.imshow(residual[idx]/residual[idx].max())
                ax1.text(3., 10., '%.3f'%residual[idx].max(), size=8, color='w')
                ax1.text(72., 75., str(seq_remain[idx]), size=8, color='w')
                if idx == 0:
                    ax1.text(0.5, -0.1, 'Blended', size=8, ha='center', transform=ax1.transAxes)
                else:
                    ax1.text(0.5, -0.1, 'Residual '+ str(int(idx)), size=8, ha='center', transform=ax1.transAxes)
                ax2.imshow(deblended[idx]/deblended[idx].max())
                ax2.text(3., 10., '%.3f'%deblended[idx].max(), size=8, color='w')
                ax2.text(72., 75., str(seq_pred[idx]), size=8, color='w')
                ax2.text(0.5, -0.1, 'Deblended '+ str(int(idx+1)), size=8, ha='center', transform=ax2.transAxes)
                ax1.set_xticks([])
                ax1.set_yticks([])
                ax2.set_xticks([])
                ax2.set_yticks([])

        fig.tight_layout()
        fig.subplots_adjust(wspace=-0.15,hspace=0.15)
        if not os.path.isdir(savedir):
            os.makedirs(savedir)
        filename = os.path.join(savedir, 'test-{}-{}.png'.format(batch, i))
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
