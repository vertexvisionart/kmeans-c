"""
Render plots from the CSVs produced by `kmeans_demo`:

  output/kmeans_results.png  - true vs predicted clusters, elbow curve, errors
  output/silhouette.png      - per-sample silhouette plot

Silhouette is computed manually (no sklearn).
"""

import csv
from itertools import permutations
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

Path('output').mkdir(exist_ok=True)


def load_csv(path):
    """Read a CSV with a header row into a dict of numpy arrays."""
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return {key: np.array([float(r[key]) for r in rows]) for key in rows[0]}


dataset = load_csv('data/dataset.csv')
centroids = load_csv('data/centroids.csv')
elbow = load_csv('data/elbow.csv')

X = np.column_stack([dataset['x0'], dataset['x1']])
true_labels = dataset['true_label'].astype(int)
pred_labels = dataset['pred_label'].astype(int)
C = np.column_stack([centroids['x0'], centroids['x1']])
cent_ids = centroids['cluster'].astype(int)
k = len(cent_ids)

PALETTE = ['#E74C3C', '#3498DB', '#2ECC71', '#9B59B6', '#F39C12',
           '#1ABC9C', '#E67E22', '#34495E', '#E91E63', '#00BCD4']


def silhouette_sample(X, labels, idx):
    """Silhouette score for a single point."""
    c = labels[idx]
    same = X[labels == c]
    if len(same) <= 1:
        return 0.0
    a = np.mean(np.linalg.norm(same - X[idx], axis=1))
    other = np.unique(labels[labels != c])
    if len(other) == 0:
        return 0.0
    b = min(np.mean(np.linalg.norm(X[labels == oc] - X[idx], axis=1))
            for oc in other)
    return (b - a) / max(a, b)


print('computing silhouette...')
np.random.seed(0)
sample_idx = np.random.choice(len(X), size=min(200, len(X)), replace=False)
sil_scores = np.array([silhouette_sample(X, pred_labels, i) for i in sample_idx])
mean_sil = float(np.mean(sil_scores))
print(f'  mean silhouette (200-sample): {mean_sil:.4f}')


def best_label_mapping(true, pred, k):
    """Try every permutation of cluster ids; return best accuracy + mapped preds."""
    best_acc = 0.0
    best_perm = tuple(range(k))
    for perm in permutations(range(k)):
        mapped = np.array([perm[p] for p in pred])
        acc = float(np.mean(mapped == true))
        if acc > best_acc:
            best_acc = acc
            best_perm = perm
    return best_acc, np.array([best_perm[p] for p in pred])


print('aligning predicted labels with ground truth...')
accuracy, mapped_pred = best_label_mapping(true_labels, pred_labels, k)
print(f'  accuracy after alignment: {accuracy * 100:.2f}%')

# Main 2x2 figure ------------------------------------------------------------
fig = plt.figure(figsize=(16, 12))
fig.patch.set_facecolor('#F8F9FA')
gs = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

STYLE = dict(alpha=0.72, s=32, linewidths=0.3, edgecolors='white')

# 1: true labels
ax1.set_facecolor('#FAFAFA')
for c in range(k):
    mask = true_labels == c
    ax1.scatter(X[mask, 0], X[mask, 1], color=PALETTE[c],
                label=f'cluster {c}', **STYLE)
ax1.set_title('Ground-truth labels', fontsize=13, fontweight='bold', pad=10)
ax1.set_xlabel('x0')
ax1.set_ylabel('x1')
ax1.legend(fontsize=9, framealpha=0.85, loc='upper right')
ax1.grid(True, alpha=0.3, linestyle='--')
for s in ('top', 'right'):
    ax1.spines[s].set_visible(False)

# 2: predicted labels + centroids
ax2.set_facecolor('#FAFAFA')
for c in range(k):
    mask = pred_labels == c
    ax2.scatter(X[mask, 0], X[mask, 1], color=PALETTE[c],
                label=f'cluster {c}', **STYLE)
ax2.scatter(C[:, 0], C[:, 1], marker='*', s=380, c='white',
            edgecolors='black', linewidths=1.5, zorder=10, label='centroids')
for i, cid in enumerate(cent_ids):
    ax2.scatter(C[i, 0], C[i, 1], marker='*', s=220, c=[PALETTE[cid]],
                edgecolors='black', linewidths=0.8, zorder=11)

inertia_val = float(np.sum([
    np.sum((X[pred_labels == c] - C[cent_ids == c]) ** 2) for c in range(k)
]))
ax2.set_title(f'k-means result (k={k})\n'
              f'inertia: {inertia_val:.1f}   silhouette: {mean_sil:.3f}',
              fontsize=13, fontweight='bold', pad=10)
ax2.set_xlabel('x0')
ax2.set_ylabel('x1')
ax2.legend(fontsize=9, framealpha=0.85, loc='upper right')
ax2.grid(True, alpha=0.3, linestyle='--')
for s in ('top', 'right'):
    ax2.spines[s].set_visible(False)

# 3: elbow curve
k_vals = elbow['k'].astype(int)
inertias = elbow['inertia']

ax3.set_facecolor('#FAFAFA')
ax3.plot(k_vals, inertias, 'o-', color='#3498DB', lw=2.5,
         markersize=8, markerfacecolor='white', markeredgewidth=2,
         markeredgecolor='#3498DB', zorder=5)
ax3.fill_between(k_vals, inertias, alpha=0.1, color='#3498DB')

diffs2 = np.diff(np.diff(inertias))
elbow_idx = int(np.argmax(np.abs(diffs2))) + 1
opt_k = int(k_vals[elbow_idx])
ax3.axvline(opt_k, color='#E74C3C', linestyle='--', lw=1.5, alpha=0.8)
ax3.scatter([opt_k], [inertias[elbow_idx]], s=120, color='#E74C3C',
            zorder=6, label=f'elbow at k={opt_k}')

ax3.set_title('Elbow method', fontsize=13, fontweight='bold', pad=10)
ax3.set_xlabel('number of clusters k')
ax3.set_ylabel('inertia (WCSS)')
ax3.legend(fontsize=10, framealpha=0.85)
ax3.set_xticks(k_vals)
ax3.grid(True, alpha=0.3, linestyle='--')
for s in ('top', 'right'):
    ax3.spines[s].set_visible(False)
for kv, iv in zip(k_vals, inertias):
    ax3.annotate(f'{iv:.0f}', (kv, iv), textcoords='offset points',
                 xytext=(0, 10), ha='center', fontsize=8, color='#555555')

# 4: correct vs misclassified
correct = mapped_pred == true_labels
incorrect = ~correct

ax4.set_facecolor('#FAFAFA')
ax4.scatter(X[correct, 0], X[correct, 1], c='#2ECC71', alpha=0.6, s=28,
            linewidths=0.2, edgecolors='white',
            label=f'correct ({int(correct.sum())})')
ax4.scatter(X[incorrect, 0], X[incorrect, 1], c='#E74C3C', s=80, marker='x',
            linewidths=1.5, zorder=5,
            label=f'misclassified ({int(incorrect.sum())})')
ax4.set_title(f'Comparison with ground truth\naccuracy: {accuracy * 100:.1f}%',
              fontsize=13, fontweight='bold', pad=10)
ax4.set_xlabel('x0')
ax4.set_ylabel('x1')
ax4.legend(fontsize=10, framealpha=0.85)
ax4.grid(True, alpha=0.3, linestyle='--')
for s in ('top', 'right'):
    ax4.spines[s].set_visible(False)

fig.suptitle('K-means clustering (C99)', fontsize=16, fontweight='bold',
             y=0.99, color='#2C3E50')

plt.savefig('output/kmeans_results.png', dpi=150,
            bbox_inches='tight', facecolor=fig.get_facecolor())
print('saved: output/kmeans_results.png')
plt.close()

# Silhouette plot ------------------------------------------------------------
fig2, ax = plt.subplots(figsize=(10, 5))
fig2.patch.set_facecolor('#F8F9FA')
ax.set_facecolor('#FAFAFA')

y_lower = 10
for c in range(k):
    mask = pred_labels[sample_idx] == c
    sil_c = np.sort(sil_scores[mask])
    size_c = len(sil_c)
    y_upper = y_lower + size_c
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, sil_c,
                     facecolor=PALETTE[c], alpha=0.85)
    ax.text(-0.05, y_lower + size_c / 2, str(c), fontsize=11,
            va='center', color=PALETTE[c], fontweight='bold')
    y_lower = y_upper + 5

ax.axvline(mean_sil, color='#E74C3C', linestyle='--', lw=1.8,
           label=f'mean silhouette = {mean_sil:.3f}')
ax.set_title('Silhouette analysis\n'
             '(values > 0 mean the point fits its cluster well)',
             fontsize=13, fontweight='bold')
ax.set_xlabel('silhouette coefficient')
ax.set_ylabel('cluster')
ax.set_xlim(-0.2, 1.0)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, linestyle='--', axis='x')
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)

plt.tight_layout()
plt.savefig('output/silhouette.png', dpi=150,
            bbox_inches='tight', facecolor=fig2.get_facecolor())
print('saved: output/silhouette.png')
plt.close()

print(f'\naccuracy:           {accuracy * 100:.2f}%')
print(f'mean silhouette:    {mean_sil:.4f}')
