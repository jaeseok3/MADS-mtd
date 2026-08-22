"""ASR <-> steady-state throughput trade-off (CIFAR-100), all three architectures
overlaid. Each arch: a periodic-switching frontier (throughput collapses as the
period shortens, ASR stuck near the transfer floor), a Static point (max
throughput, broken), and MADS (ours) in the good corner: low ASR at near-max
throughput. Per-arch latencies from throughput_results.json (RTX 2070S); MADS's
per-query monitor overhead (~7%, measured on ResNet) is applied uniformly.
-> figures/fig_throughput_tradeoff.{pdf,png}"""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
for f in ['Regular', 'Bold', 'Italic']:
    p = f'/usr/share/fonts/truetype/liberation2/LiberationSerif-{f}.ttf'
    if os.path.exists(p): fm.fontManager.addfont(p)
plt.rcParams.update({'font.family': 'Liberation Serif', 'mathtext.fontset': 'stix',
                     'font.size': 9.5, 'axes.labelsize': 10.5, 'legend.fontsize': 7.5,
                     'xtick.labelsize': 9, 'ytick.labelsize': 9, 'axes.linewidth': 0.8,
                     'xtick.direction': 'in', 'ytick.direction': 'in'})
DIR = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.join(os.path.dirname(DIR), 'figures')
lat = json.load(open(os.path.join(DIR, 'throughput_results.json')))
MON = 3316.6 / 3103.5   # MADS per-query monitor overhead ratio (measured, ResNet); MADS tput = 100/MON
# top-tier scheme: ColorBrewer Dark2 + distinct line style per architecture
ARCHS = [('ResNet-18', '#1b9e77', '-',  'throughput_tradeoff.json'),
         ('VGG-16-BN', '#d95f02', '--', 'throughput_tradeoff_vgg.json'),
         ('AlexNet',   '#7570b3', '-.', 'throughput_tradeoff_alexnet.json')]
PERIODS = [10, 25, 50, 100, 250, 1000]

fig, ax = plt.subplots(figsize=(5.0, 2.95))
ax.add_patch(Rectangle((14, 80), 34, 30, color='#4d4d4d', alpha=0.055, zorder=0, lw=0))
for name, col, ls, fn in ARCHS:
    d = json.load(open(os.path.join(DIR, fn)))
    Lc = lat[name]; T_INF = Lc['inference_latency_mean_ms']*1000; T_SW = Lc['switch_overhead_mean_ms']*1000
    px = [d[f'periodic_{P}']['asr'] for P in PERIODS]
    py = [T_INF/(T_INF + (1.0/P)*T_SW)*100 for P in PERIODS]
    ax.plot(px, py, ls=ls, color=col, lw=1.6, alpha=0.9, zorder=3)
    ax.plot(px, py, 'o', color='white', mec=col, mew=1.3, ms=5, zorder=3.5)
    ax.scatter([d['static']['asr']], [100], s=72, color=col, marker='D', edgecolor='white', linewidth=0.7, zorder=4)
    my = 100.0/MON
    ax.scatter([d['mads']['asr']], [my], s=290, color=col, marker='*', edgecolor='black', linewidth=0.9, zorder=6)

ax.set_xlim(14, 96); ax.set_ylim(0, 109)
ax.set_xlabel('Pure-attack ASR (%)  -  lower is better'); ax.set_ylabel('Throughput (% of max)')
ax.grid(True, alpha=0.18, ls='-', lw=0.5, color='#b0b0b0'); ax.set_axisbelow(True)
ax.annotate('better', xy=(23.0, 91.5), xytext=(33, 73), fontsize=10, color='#333333',
            style='italic', fontweight='bold', ha='left', va='center',
            arrowprops=dict(arrowstyle='-|>', color='#333333', lw=1.5, connectionstyle='arc3,rad=-0.30'))
ax.text(33, 65, '(low ASR,\nhigh throughput)', fontsize=7.5, color='#555555', va='top')

arch_h = [Line2D([0], [0], color=c, lw=2.0, ls=ls, label=n) for n, c, ls, _ in ARCHS]
pol_h = [Line2D([0], [0], marker='*', color='w', mfc='#8a8a8a', mec='k', ms=13, ls='', label='MADS (ours)'),
         Line2D([0], [0], marker='D', color='w', mfc='#8a8a8a', ms=7, ls='', label='Static (no switch)'),
         Line2D([0], [0], marker='o', color='w', mfc='w', mec='#8a8a8a', mew=1.3, ms=6, ls='', label='Periodic ($P{=}10{-}1000$)')]
l1 = ax.legend(handles=arch_h, loc='lower left', frameon=True, edgecolor='#cccccc', framealpha=0.95,
               title='Architecture', title_fontsize=7.5, borderpad=0.5, labelspacing=0.3,
               bbox_to_anchor=(0.005, 0.02))
l1.get_frame().set_linewidth(0.7); ax.add_artist(l1)
l2 = ax.legend(handles=pol_h, loc='upper center', frameon=True, edgecolor='#cccccc', framealpha=0.95,
               borderpad=0.5, labelspacing=0.4, handletextpad=0.4, bbox_to_anchor=(0.5, 0.99))
l2.get_frame().set_linewidth(0.7)
plt.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(FIG, f'fig_throughput_tradeoff.{ext}'), dpi=220, bbox_inches='tight')
print('saved fig_throughput_tradeoff (3-arch)')
for name, col, fn in ARCHS:
    d = json.load(open(os.path.join(DIR, fn)))
    print(f'  {name}: MADS asr={d["mads"]["asr"]} static={d["static"]["asr"]} periodic={[d[f"periodic_{P}"]["asr"] for P in PERIODS]}')
