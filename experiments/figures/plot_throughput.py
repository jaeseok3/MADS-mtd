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
    'font.size': 17,
    'axes.labelsize': 20,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 15,
    'axes.linewidth': 1.0,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
})

with open('throughput_results.json') as f:
    data = json.load(f)

archs = {
    'ResNet-18': ('#1a3a5c', '-', 'o'),
    'VGG-16-BN': ('#b2182b', '--', 's'),
    'AlexNet': ('#1b7837', '-.', '^'),
}

fig, ax = plt.subplots(figsize=(4.7, 3.4))

for name, (color, ls, marker) in archs.items():
    d = data[name]
    qmax = d['q_max_values']
    tp = d['throughputs']

    ax.plot(qmax, tp, color=color, linestyle=ls, marker=marker,
            markersize=5, linewidth=1.8, label=name)
    ax.axhline(y=d['baseline_throughput'], color=color, linestyle=':', linewidth=0.8, alpha=0.5)

ax.set_xscale('log')
ax.set_xlabel(r'$Q_{\max}$')
ax.set_ylabel('Throughput (queries/sec)')
ax.legend(frameon=True, fancybox=False, edgecolor='black', borderpad=0.4)
ax.grid(True, alpha=0.25, linewidth=0.5, linestyle='--')
ax.set_axisbelow(True)

for spine in ax.spines.values():
    spine.set_linewidth(1.0)

plt.tight_layout()
plt.savefig('throughput_vs_qmax.pdf', dpi=300, bbox_inches='tight')
print('Saved throughput_vs_qmax.pdf')
