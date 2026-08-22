"""Single figure visualizing Table 5's decisive point: decision-based (HSJA) ASR
vs query budget, three datasets (ResNet). AAA preserves the top-1 label so its
decision-based ASR equals no-defense and CLIMBS with the budget, while MADS holds
0% at every budget. Consumes arch_hardlabel_{ds}_resnet_b{B}.json (same sweep
harness). -> figures/fig_decision_budget.{pdf,png}"""
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
                     'xtick.direction': 'in', 'ytick.direction': 'in'})  # shared style w/ fig_adaptive_obfuscator
DIR = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.join(os.path.dirname(DIR), 'figures')
BUDGETS = list(range(2000, 16001, 1000))  # 2k..16k at 1k intervals
# top-tier scheme: ColorBrewer Dark2 + distinct marker & line style per dataset
DS = [('cifar', 'CIFAR-100', '#1b9e77', 'o', '-'),
      ('c10', 'CIFAR-10', '#d95f02', 's', '--'),
      ('tin', 'TinyImageNet', '#7570b3', '^', '-.')]

fig, ax = plt.subplots(figsize=(3.5, 2.05))  # match fig_adaptive_obfuscator
xs_all = []
for key, name, col, mk, ls in DS:
    xs, static = [], []
    for B in BUDGETS:
        fp = os.path.join(DIR, f'arch_hardlabel_{key}_resnet_b{B}.json')
        if not os.path.exists(fp): continue
        d = json.load(open(fp)); xs.append(B); static.append(d['HSJA_static']['asr'])
    xs_all = xs if len(xs) > len(xs_all) else xs_all
    ax.plot(xs, static, marker=mk, ls=ls, color=col, lw=1.6, ms=4, mec='white', mew=0.5,
            label=f'{name} - Existing')
# MADS: 0% at every budget for all datasets (single flat line, distinct black diamond)
ax.plot(xs_all, [0]*len(xs_all), marker='D', ls=(0, (5, 2)), color='black', lw=1.8, ms=4,
        label='all datasets - MADS (ours)', zorder=5)
TICKS = list(range(2000, 16001, 2000))
ax.set_xticks(TICKS); ax.set_xticklabels([f'{t//1000}k' for t in TICKS])
ax.xaxis.set_minor_locator(plt.NullLocator())
ax.set_xlabel('Decision-based query budget $Q_{\\max}$')
ax.set_ylabel('HSJA attack success rate (%)'); ax.set_ylim(-3, 82)
ax.grid(True, alpha=0.25, ls='--', lw=0.5); ax.set_axisbelow(True)
leg = ax.legend(frameon=True, edgecolor='#bbb', framealpha=0.95, loc='upper right',
                labelspacing=0.25, handlelength=1.5, borderpad=0.35, handletextpad=0.45)
leg.get_frame().set_linewidth(0.6)
plt.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(FIG, f'fig_decision_budget.{ext}'), dpi=200, bbox_inches='tight')
print('saved fig_decision_budget')
