"""Pairwise cross-ASR heatmap: Standard KD vs Divergent KD."""
import json, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = "experiments"

with open(f'{OUT_DIR}/divergent_kd_results_standard_baseline.json') as f:
    std = json.load(f)
with open(f'{OUT_DIR}/divergent_kd_results_v5b_pool.json') as f:
    div = json.load(f)

mat_std = np.array(std['transfer']['transfer_matrix'])
mat_div = np.array(div['transfer']['transfer_matrix'])

labels = [f'S{i}' for i in range(5)]

# --- Figure 1: Side-by-side heatmaps ---
fig, axes = plt.subplots(1, 3, figsize=(14, 5),
                         gridspec_kw={'width_ratios': [1, 1, 0.05]})

vmin, vmax = 65, 100

for ax, mat, title in [(axes[0], mat_std, 'Standard KD'),
                        (axes[1], mat_div, 'Divergent KD (All-Pairs)')]:
    im = ax.imshow(mat, cmap='YlOrRd', vmin=vmin, vmax=vmax, aspect='equal')
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel('Target Student', fontsize=12)
    ax.set_ylabel('Source Student', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)

    for i in range(5):
        for j in range(5):
            val = mat[i, j]
            color = 'white' if val > 92 else 'black'
            if i == j:
                ax.text(j, i, f'{val:.0f}%', ha='center', va='center',
                        fontsize=10, color=color, fontweight='bold')
            else:
                ax.text(j, i, f'{val:.1f}%', ha='center', va='center',
                        fontsize=10, color=color)

    cross = [mat[i, j] for i in range(5) for j in range(5) if i != j]
    avg = np.mean(cross)
    ax.text(2, 5.5, f'Avg Cross-ASR: {avg:.1f}%', ha='center',
            fontsize=11, fontweight='bold', color='#D32F2F')

cb = fig.colorbar(im, cax=axes[2], label='ASR (%)')
cb.ax.tick_params(labelsize=10)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig_transfer_heatmap.pdf', dpi=150, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig_transfer_heatmap.png', dpi=150, bbox_inches='tight')
print("[*] Saved fig_transfer_heatmap")


# --- Figure 2: Pairwise bar comparison ---
fig2, ax2 = plt.subplots(figsize=(10, 4.5))

cross_std = [mat_std[i, j] for i in range(5) for j in range(5) if i != j]
cross_div = [mat_div[i, j] for i in range(5) for j in range(5) if i != j]

pairs = [f'S{i}->S{j}' for i in range(5) for j in range(5) if i != j]
x = np.arange(len(pairs))
w = 0.35

bars1 = ax2.bar(x - w/2, cross_std, w, label='Standard KD', color='#9E9E9E', alpha=0.85)
bars2 = ax2.bar(x + w/2, cross_div, w, label='Divergent KD', color='#E91E63', alpha=0.85)

ax2.axhline(y=np.mean(cross_std), color='#757575', linestyle='--', alpha=0.7, linewidth=1.5)
ax2.axhline(y=np.mean(cross_div), color='#C2185B', linestyle='--', alpha=0.7, linewidth=1.5)

ax2.text(len(pairs)-0.5, np.mean(cross_std)+0.8, f'avg {np.mean(cross_std):.1f}%',
         fontsize=9, color='#616161', ha='right')
ax2.text(len(pairs)-0.5, np.mean(cross_div)-1.8, f'avg {np.mean(cross_div):.1f}%',
         fontsize=9, color='#C2185B', ha='right')

ax2.set_ylabel('Cross-ASR (%)', fontsize=12)
ax2.set_xlabel('Student Pair (Source -> Target)', fontsize=12)
ax2.set_xticks(x)
ax2.set_xticklabels(pairs, rotation=45, ha='right', fontsize=8)
ax2.legend(fontsize=10, loc='upper right')
ax2.set_ylim(65, 95)
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig_pairwise_transfer.pdf', dpi=150, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig_pairwise_transfer.png', dpi=150, bbox_inches='tight')
print("[*] Saved fig_pairwise_transfer")
