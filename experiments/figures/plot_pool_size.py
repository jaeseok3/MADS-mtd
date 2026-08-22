"""Pool-size (K) ablation figure (CIFAR-100 ResNet). Direct mechanism test vs the
EXISTING MTD baselines: periodic and random switching pick the replacement blindly
(random), so their pure-attack ASR is flat in K -- more models do not help them.
MADS picks the replacement by greedy max-margin, so its ASR falls steadily as K
grows (more per-input options). Hence a larger pool helps only when selection is
input-aware. -> figures/fig_pool_size.{pdf,png}"""
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
                     'font.size': 15, 'axes.labelsize': 16, 'legend.fontsize': 11.5,
                     'xtick.direction': 'in', 'ytick.direction': 'in'})
DIR = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.join(os.path.dirname(DIR), 'figures')
d = json.load(open(os.path.join(DIR, 'pool_size_ablation.json')))
Ks = sorted(int(k) for k in d)
mads = [d[str(k)]['mads_asr'] for k in Ks]
periodic = [d[str(k)]['periodic_asr'] for k in Ks]
rand = [d[str(k)]['random_asr'] for k in Ks]
STATIC = 91.0  # CIFAR-100 static pure-attack Avg ASR (Table 3)

fig, ax = plt.subplots(figsize=(6.2, 4.4))
ax.axhline(STATIC, color='#8d8d8d', ls=':', lw=1.8, label='Static (no switching)')
ax.plot(Ks, rand, 'D--', color='#1565c0', lw=2.0, ms=6, label='Random switch (existing MTD)')
ax.plot(Ks, periodic, 's--', color='#6a1b9a', lw=2.0, ms=6, label='Periodic switch (existing MTD)')
ax.plot(Ks, mads, 'o-', color='#c62828', lw=2.7, ms=8.5, label='MADS: greedy selection (ours)')
ax.set_xlabel('Pool size $K$ (models in rotation)')
ax.set_ylabel('Pure-attack ASR (%)')
ax.set_xticks(Ks); ax.set_ylim(0, 100)
ax.grid(True, alpha=0.25, ls='--', lw=0.5); ax.set_axisbelow(True)
ax.legend(loc='lower left', frameon=True, edgecolor='black')
plt.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(FIG, f'fig_pool_size.{ext}'), dpi=200, bbox_inches='tight')
print('saved fig_pool_size')
print('  K:', Ks, 'MADS:', mads, 'periodic:', periodic, 'random:', rand)
