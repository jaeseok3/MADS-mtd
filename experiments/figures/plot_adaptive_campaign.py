"""figure*: 3 subfigures (ResNet / VGG / AlexNet) side by side. Each plots, vs
shield strength alpha, the adaptive attack success of a switch-aware RESTART
attacker and a TRANSFER-reuse attacker (left y-axis, ASR %), plus benign accuracy
(right y-axis). Shows the shield collapses the restart attacker at zero accuracy
cost while divergence already holds the transfer attacker low."""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

for f in ['Regular', 'Bold', 'Italic']:
    p = f'/usr/share/fonts/truetype/liberation2/LiberationSerif-{f}.ttf'
    if os.path.exists(p): fm.fontManager.addfont(p)
plt.rcParams.update({'font.family': 'Liberation Serif', 'mathtext.fontset': 'stix',
                     'font.size': 9, 'axes.labelsize': 9.5, 'xtick.labelsize': 8,
                     'ytick.labelsize': 8, 'legend.fontsize': 7, 'axes.linewidth': 0.8,
                     'xtick.direction': 'in', 'ytick.direction': 'in'})  # shared style w/ fig_decision_budget
DIR = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.join(os.path.dirname(DIR), 'figures')
ARCHS = [('resnet', 'ResNet-18'), ('vgg', 'VGG-16-BN'), ('alexnet', 'AlexNet')]

# single taller panel: all three architectures overlaid. Restart attacker (solid,
# left axis) collapses as shield strength grows; benign accuracy (dotted, right
# axis) stays flat. Transfer-reuse attacker dropped.
from matplotlib.lines import Line2D
# top-tier scheme: ColorBrewer Dark2 + distinct marker & line style per series (B&W-safe)
STY = {'resnet': ('#1b9e77', 'o', '-'), 'vgg': ('#d95f02', 's', '--'), 'alexnet': ('#7570b3', '^', '-.')}
fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.05))  # match fig_decision_budget
ax2 = ax.twinx()
for a, name in ARCHS:
    col, mk, ls = STY[a]
    d = json.load(open(os.path.join(DIR, f'c10_adaptive_campaign_{a}.json')))
    xs = sorted(float(k) for k in d)
    restart = [d[f'{x}']['restart_asr'] for x in xs]
    acc = [d[f'{x}']['acc'] for x in xs]
    ax.plot(xs, restart, marker=mk, ls=ls, color=col, lw=1.6, ms=4, mec='white', mew=0.5, zorder=4)
    ax2.plot(xs, acc, ls=':', color=col, lw=1.2, alpha=0.75, zorder=3)
ax.set_xlabel(r'Obfuscator strength $\alpha_{\max}$', labelpad=2); ax.set_ylim(0, 100)
ax.set_ylabel('Restart-attacker ASR (%)', labelpad=2)
ax.grid(True, alpha=0.22, ls='--', lw=0.4); ax.set_axisbelow(True)
ax2.set_ylim(0, 100); ax2.set_ylabel('Benign accuracy (%)', color='#4d4d4d', labelpad=2)
# single compact legend: 3 archs (solid = restart ASR, left) + benign-acc key (dotted, right)
leg_h = [Line2D([0], [0], color=STY[a][0], lw=1.7, marker=STY[a][1], ls=STY[a][2], ms=4.5,
                mec='white', mew=0.5, label=name) for a, name in ARCHS]
leg_h.append(Line2D([0], [0], color='#888', lw=1.2, ls=':', label='Benign acc. (right)'))
leg = ax.legend(handles=leg_h, loc='upper right', frameon=True, edgecolor='#bbb',
                framealpha=0.95, labelspacing=0.25, handlelength=1.5, borderpad=0.35, handletextpad=0.45,
                bbox_to_anchor=(0.995, 0.985))
leg.get_frame().set_linewidth(0.6)
plt.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(FIG, f'fig_adaptive_shield.{ext}'), dpi=200, bbox_inches='tight')
print('saved fig_adaptive_shield (3-arch, taller, no transfer-reuse)')
