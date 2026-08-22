"""MOCKUP: pairwise cross-ASR as ONE directed graph, all methods overlaid.
5 nodes = the K=5 students (S0..S4). Every method draws its own directed arrows
i->j (width proportional to that method's ASR of i-crafted examples on j) in its
own colour, each method offset to a different arc curvature so the six colours
fan out on each edge. One big canvas so the lines stay separable.
Uses tin_heatmap_matrices.json (ResNet-18, eps=2/255). EI-MTD has no stored 5x5
matrix, so for this feel-check it borrows Std KD's (placeholder, labelled)."""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle
from matplotlib.lines import Line2D

DIR = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(os.path.dirname(DIR), 'figures')
data = json.load(open(os.path.join(DIR, 'tin_heatmap_matrices.json')))['resnet']

# (key, legend label, colour, matrix)
METHODS = [
    ('Independent',    'Independent (No KD)',        '#7B1FA2', data['Independent']['transfer_matrix']),
    ('Std KD',         'Standard KD',                '#F57C00', data['Std KD']['transfer_matrix']),
    ('Morphence',      'Morphence',                  '#2E7D32', data['Morphence']['transfer_matrix']),
    ('DeepMTD',        'DeepMTD',                    '#D32F2F', data['DeepMTD']['transfer_matrix']),
    ('EI-MTD',         'EI-MTD (placeholder)',       '#795548', data['Std KD']['transfer_matrix']),
    ('Div KD + Noise', 'Div KD + Noise (Proposed)',  '#1976D2', data['Div KD + Noise']['transfer_matrix']),
]
K = 5
ang = np.pi/2 + np.arange(K) * 2*np.pi/K
POS = np.stack([np.cos(ang), np.sin(ang)], 1) * 2.4   # bigger pentagon

fig, ax = plt.subplots(figsize=(15, 15))
# per-method arc curvature so the six colours separate on every edge
RADS = np.linspace(0.06, 0.42, len(METHODS))
for m, (key, label, col, M) in enumerate(METHODS):
    M = np.array(M); rad = RADS[m]
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            asr = M[i][j]
            lw = 0.15 + ((asr/100.0)**2.2)*11.0   # steep so thin edges stay thin, thick edges pop
            a = FancyArrowPatch(POS[i], POS[j], connectionstyle=f'arc3,rad={rad}',
                                arrowstyle='-|>', mutation_scale=9,
                                lw=lw, color=col, alpha=0.8,
                                shrinkA=42, shrinkB=42, zorder=2)
            ax.add_patch(a)
# nodes: white fill, black outline (larger)
for i in range(K):
    ax.add_patch(Circle(POS[i], 0.42, facecolor='white', edgecolor='black', lw=2.6, zorder=5))
    ax.text(POS[i][0], POS[i][1], f'S{i}', ha='center', va='center',
            fontsize=22, fontweight='bold', color='black', zorder=6)

legend = [Line2D([0], [0], color=c, lw=4, label=l) for _, l, c, _ in METHODS]
ax.legend(handles=legend, loc='upper center', bbox_to_anchor=(0.5, 1.06),
          ncol=3, fontsize=13, frameon=False)
ax.set_title('Cross-model adversarial transfer, all methods overlaid '
             '(ResNet-18, TinyImageNet, $\\epsilon{=}2/255$; arrow $i{\\to}j$ width $\\propto$ ASR)',
             fontsize=14, fontweight='bold', pad=42)
ax.set_xlim(-3.2, 3.2); ax.set_ylim(-3.2, 3.2); ax.set_aspect('equal'); ax.axis('off')
plt.savefig(os.path.join(FIGDIR, 'fig_cross_asr_graph.png'), dpi=140, bbox_inches='tight')
plt.savefig(os.path.join(FIGDIR, 'fig_cross_asr_graph.pdf'), bbox_inches='tight')
print('[*] saved fig_cross_asr_graph (single overlaid graph)')
