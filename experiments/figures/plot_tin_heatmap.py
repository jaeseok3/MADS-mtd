"""3x5 transfer heatmap on TinyImageNet (eps=2/255): 3 architectures x 5 methods
Rows: ResNet-18, VGG-16-BN, AlexNet.  Cols: Independent, Std KD, Morphence,
DeepMTD, Div KD + Noise (Proposed).  Reads tin_heatmap_matrices.json."""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DIR = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(os.path.dirname(DIR), 'figures')
data = json.load(open(os.path.join(DIR, 'tin_heatmap_matrices.json')))

ARCHS = [('ResNet-18', 'resnet'), ('VGG-16-BN', 'vgg'), ('AlexNet', 'alexnet')]
METHODS = [
    ('Independent', 'Independent\n(No KD)'),
    ('Std KD', 'Standard KD'),
    ('Morphence', 'Morphence\n(Std KD + Noise)'),
    ('DeepMTD', 'DeepMTD\n(single base)'),
    ('Div KD + Noise', 'Div KD + Noise\n(Proposed)'),
]
nr, nc = len(ARCHS), len(METHODS)
fig, axes = plt.subplots(nr, nc, figsize=(21, 9.8), gridspec_kw={'hspace': 0.34, 'wspace': 0.10})

for r, (aname, akey) in enumerate(ARCHS):
    for c, (mkey, title) in enumerate(METHODS):
        ax = axes[r, c]
        M = np.array(data[akey][mkey]['transfer_matrix']); cross = data[akey][mkey]['cross']; K = M.shape[0]
        im = ax.imshow(M, cmap='YlOrRd', vmin=15, vmax=100, aspect='equal')
        for i in range(K):
            for j in range(K):
                v = M[i, j]
                ax.text(j, i, f'{v:.0f}', ha='center', va='center', fontsize=9,
                        color='white' if v > 80 else 'black', fontweight='bold' if i == j else 'normal')
        ax.set_xticks(range(K)); ax.set_yticks(range(K))
        ax.set_xticklabels([f'S{i}' for i in range(K)], fontsize=9)
        ax.set_yticklabels([f'S{i}' for i in range(K)], fontsize=9)
        if c == 0: ax.set_ylabel(f'{aname}\nSource', fontsize=13, fontweight='bold')
        if r == nr-1: ax.set_xlabel('Target', fontsize=12)
        if r == 0:
            ours = (mkey == 'Div KD + Noise')
            ax.set_title(title, fontsize=13, fontweight='bold', color='#AD1457' if ours else '#37474F', pad=10)
        col = '#AD1457' if mkey == 'Div KD + Noise' else '#D32F2F'
        ax.text(K/2-0.5, K+0.5, f'cross-ASR: {cross:.1f}%', ha='center', va='top', fontsize=12, color=col, fontweight='bold')

# colorbar removed: every cell is annotated with its ASR, so the bar was redundant and left right-side whitespace
# suptitle removed: the caption already states dataset/budget/K
# no tight_layout: it would override the gridspec hspace that keeps the cross-ASR labels clear
plt.savefig(os.path.join(FIGDIR, 'fig_3x4_heatmap.pdf'), dpi=150, bbox_inches='tight')
plt.savefig(os.path.join(FIGDIR, 'fig_3x4_heatmap.png'), dpi=150, bbox_inches='tight')
print('[*] saved fig_3x4_heatmap (TIN 3x5)')
