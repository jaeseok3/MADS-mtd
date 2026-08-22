import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

fm.fontManager.addfont('/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf')
fm.fontManager.addfont('/usr/share/fonts/truetype/liberation2/LiberationSerif-Bold.ttf')
fm.fontManager.addfont('/usr/share/fonts/truetype/liberation2/LiberationSerif-Italic.ttf')
fm.fontManager.addfont('/usr/share/fonts/truetype/liberation2/LiberationSerif-BoldItalic.ttf')

plt.rcParams.update({
    'font.family': 'Liberation Serif',
    'mathtext.fontset': 'stix',
    'font.size': 17,
    'axes.labelsize': 20,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 11,
    'axes.linewidth': 1.0,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
})

df = pd.read_csv('per_image_queries.csv')

attacks = ['SimBA-DCT', 'NES', 'SignHunter', 'SquareAttack']
colors = {
    'SimBA-DCT':    '#b35806',
    'NES':          '#1a3a5c',
    'SignHunter':   '#b2182b',
    'SquareAttack': '#1b7837',
    'DifAttack':    '#5e3c99',
    'DifAttack++':  '#01665e',
}
linestyles = {
    'SimBA-DCT':    '--',
    'NES':          '-',
    'SignHunter':   '-.',
    'SquareAttack': ':',
    'DifAttack':    '-',
    'DifAttack++':  '-',
}
linewidths = {
    'SimBA-DCT':    1.5,
    'NES':          1.5,
    'SignHunter':   2.0,
    'SquareAttack': 1.5,
    'DifAttack':    1.5,
    'DifAttack++':  2.0,
}

num_images = 1000
q_range = np.arange(1, 1001)

fig, ax = plt.subplots(figsize=(4.7, 3.4))

for atk in attacks:
    sub = df[df['attack'] == atk]
    queries = sub['queries'].values
    success = sub['success'].values
    effective_q = np.where(success == 1, queries, 9999)

    asr = np.array([(effective_q <= q).sum() / num_images * 100 for q in q_range])

    ax.plot(q_range, asr, color=colors[atk], linestyle=linestyles[atk],
            linewidth=linewidths[atk], label=atk)

ax.set_xlabel(r'Query Budget ($Q_{\max}$)')
ax.set_ylabel('Attack Success Rate (%)')
ax.set_xlim(0, 1000)
ax.set_ylim(0, 100)
ax.legend(frameon=True, fancybox=False, edgecolor='black',
          loc='lower right', borderpad=0.3, handlelength=1.4,
          handletextpad=0.4, labelspacing=0.25, framealpha=0.9)
ax.grid(True, alpha=0.25, linewidth=0.5, linestyle='--')
ax.set_axisbelow(True)

for spine in ax.spines.values():
    spine.set_linewidth(1.0)

plt.tight_layout()
plt.savefig('asr_vs_budget.pdf', dpi=300, bbox_inches='tight')
plt.savefig('asr_vs_budget.png', dpi=150, bbox_inches='tight')
print('Saved asr_vs_budget.pdf')
