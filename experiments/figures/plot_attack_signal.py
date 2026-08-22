"""
Plot for 5.2.3 Attack-Potential Signal Validity.
(a) Margin trajectories: attack queries converge toward margin=0 (decision boundary)
(b) Margin distribution: clean images have high margin, adversarial queries cluster near 0
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.lines import Line2D

fm.fontManager.addfont('/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf')
fm.fontManager.addfont('/usr/share/fonts/truetype/liberation2/LiberationSerif-Bold.ttf')
fm.fontManager.addfont('/usr/share/fonts/truetype/liberation2/LiberationSerif-Italic.ttf')

plt.rcParams.update({
    'font.family': 'Liberation Serif',
    'mathtext.fontset': 'stix',
    'font.size': 16,
    'axes.labelsize': 17,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'legend.fontsize': 14,
    'axes.linewidth': 1.0,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
})
LETTER_Y = -0.40  # panel letter+name below x-axis label (subcaption style)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'figures')

with open(os.path.join(SCRIPT_DIR, 'attack_signal_data.json')) as f:
    data = json.load(f)

all_margins = data['all_margins']
all_success = data['all_success']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.7))

# --- (a) Margin trajectories ---
success_margins = [m for m, s in zip(all_margins, all_success) if s]
fail_margins = [m for m, s in zip(all_margins, all_success) if not s]

for m in success_margins:
    ax1.plot(range(len(m)), m, color='#c62828', alpha=0.15, linewidth=0.8)
for m in fail_margins:
    ax1.plot(range(len(m)), m, color='#1565c0', alpha=0.3, linewidth=0.8)

max_len = max(len(m) for m in success_margins)
avg_margin = []
for q in range(max_len):
    vals = [m[q] for m in success_margins if q < len(m)]
    if vals:
        avg_margin.append(np.mean(vals))
ax1.plot(range(len(avg_margin)), avg_margin, color='#b71c1c', linewidth=2.5,
         label='Avg. (successful)', zorder=5)

ax1.axhline(y=0, color='black', linestyle='--', linewidth=1.2, alpha=0.7)
ax1.text(max_len * 0.55, 0.8, 'Decision Boundary (margin = 0)',
         fontsize=12, color='#333333', style='italic')

ax1.set_xlabel('Query Index')
ax1.set_ylabel('Confidence Margin\n(true class logit $-$ runner-up logit)')
ax1.set_xlim(0, 70)
ax1.set_ylim(-5, 25)

legend_elements = [
    Line2D([0], [0], color='#c62828', alpha=0.4, linewidth=1, label='Successful attacks'),
    Line2D([0], [0], color='#b71c1c', linewidth=2.5, label='Avg. (successful)'),
    Line2D([0], [0], color='#1565c0', alpha=0.6, linewidth=1, label='Failed attacks'),
    Line2D([0], [0], color='black', linestyle='--', linewidth=1.2, label='Decision boundary'),
]
ax1.legend(handles=legend_elements, frameon=True, fancybox=False,
           edgecolor='black', borderpad=0.4, loc='upper right')
ax1.grid(True, alpha=0.2, linewidth=0.5, linestyle='--')
ax1.set_axisbelow(True)

# --- (b) Margin distribution histogram ---
# Collect margins at different stages
clean_m = [m[0] for m in all_margins]  # query 0: clean image margin
mid_m = [m[len(m)//2] for m in all_margins]  # midpoint of attack
last_m = [m[-1] for m in all_margins]  # final query margin

bins = np.linspace(-8, 25, 35)

ax2.hist(clean_m, bins=bins, alpha=0.7, color='#1565c0', edgecolor='white', linewidth=0.5,
         label='Before attack (query 0)', zorder=3)
ax2.hist(mid_m, bins=bins, alpha=0.6, color='#ff8f00', edgecolor='white', linewidth=0.5,
         label='Mid-attack', zorder=2)
ax2.hist(last_m, bins=bins, alpha=0.7, color='#c62828', edgecolor='white', linewidth=0.5,
         label='Final query', zorder=4)

ax2.axvline(x=0, color='black', linestyle='--', linewidth=1.5, zorder=5)
ax2.text(0.3, ax2.get_ylim()[1] * 0.85 if ax2.get_ylim()[1] > 0 else 12,
         'Decision\nBoundary', fontsize=12, color='#333333', style='italic', va='top')

# Add annotation arrow showing the shift
ax2.annotate('', xy=(np.mean(last_m), 8), xytext=(np.mean(clean_m), 8),
             arrowprops=dict(arrowstyle='->', color='#333333', lw=2.0))
ax2.text((np.mean(clean_m) + np.mean(last_m)) / 2, 9.0, 'Attack\nprogress',
         ha='center', va='bottom', fontsize=12, color='#333333', fontweight='bold')

ax2.set_xlabel('Confidence Margin')
ax2.set_ylabel('Number of Images')
ax2.legend(frameon=True, fancybox=False, edgecolor='black',
           borderpad=0.4, loc='upper right')
ax2.grid(True, alpha=0.2, linewidth=0.5, linestyle='--')
ax2.set_axisbelow(True)

for ax in [ax1, ax2]:
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

plt.tight_layout()
fig.canvas.draw()
for ax, letter in [(ax1, 'a'), (ax2, 'b')]:
    ext = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
    for e in ('pdf', 'png'):
        fig.savefig(os.path.join(FIG_DIR, f'attack_signal_{letter}.{e}'), dpi=200, bbox_inches=ext.expanded(1.04, 1.06))
print('Saved attack_signal_{a,b}')
