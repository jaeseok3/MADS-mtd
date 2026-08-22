import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

fm.fontManager.addfont('/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf')
fm.fontManager.addfont('/usr/share/fonts/truetype/liberation2/LiberationSerif-Bold.ttf')
fm.fontManager.addfont('/usr/share/fonts/truetype/liberation2/LiberationSerif-Italic.ttf')

plt.rcParams.update({
    'font.family': 'Liberation Serif',
    'mathtext.fontset': 'stix',
    'font.size': 15,
    'axes.labelsize': 17,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 13,
    'axes.linewidth': 1.0,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
})
LETTER_Y = -0.28  # panel letter+name below x-axis label (subcaption style)

archs = {
    'ResNet-18': ('kd_epoch_metrics_resnet.json', '#1a3a5c', '-', 'o'),
    'VGG-16-BN': ('kd_epoch_metrics_vgg.json', '#b2182b', '--', 's'),
    'AlexNet': ('kd_epoch_metrics_alexnet.json', '#1b7837', '-.', '^'),
}

fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))

for name, (fname, color, ls, marker) in archs.items():
    with open(fname) as f:
        data = json.load(f)
    epochs = [d['epoch'] for d in data]
    agreement = [d['pairwise_agreement_mean'] for d in data]
    unanimous = [d['unanimous_agreement'] for d in data]
    score_std = [d['score_std_mean'] for d in data]

    axes[0].plot(epochs, agreement, color=color, linestyle=ls, marker=marker,
                 markersize=4, linewidth=1.8, label=name, markevery=2)
    axes[1].plot(epochs, unanimous, color=color, linestyle=ls, marker=marker,
                 markersize=4, linewidth=1.8, label=name, markevery=2)
    axes[2].plot(epochs, score_std, color=color, linestyle=ls, marker=marker,
                 markersize=4, linewidth=1.8, label=name, markevery=2)

axes[0].set_ylabel('Pairwise Agreement (%)'); axes[0].set_ylim(35, 90)
axes[1].set_ylabel('Unanimous Agreement (%)'); axes[1].set_ylim(0, 70)
axes[2].set_ylabel('Softmax Score Std'); axes[2].set_ylim(0.05, 0.30)

for ax in axes:
    ax.set_xlabel('Epoch')
    ax.set_xlim(1, 20)
    ax.legend(frameon=True, fancybox=False, edgecolor='black', borderpad=0.4)
    ax.grid(True, alpha=0.25, linewidth=0.5, linestyle='--')
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

plt.tight_layout()
fig.canvas.draw()
# save each panel as a standalone file (LaTeX subfigure treatment; letters added in .tex)
for i, letter in enumerate(['a', 'b', 'c']):
    ext = axes[i].get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
    for e in ('pdf', 'png'):
        fig.savefig(f'../figures/kd_agree_{letter}.{e}', dpi=200, bbox_inches=ext.expanded(1.04, 1.06))
print('Saved kd_agree_{a,b,c}')
