import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.semi_supervised import LabelPropagation
import json

sns.set_theme(style='whitegrid')

# 1. Plots for Wine and Breast Cancer (PCA and t-SNE with L, U and Inferred labels)
for name, load_fn in [('wine', load_wine), ('breast_cancer', load_breast_cancer)]:
    data = load_fn()
    X, y = data.data, data.target
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.20, random_state=73, stratify=y)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    
    np.random.seed(73)
    y_semi = np.full_like(y_train, -1)
    for c in np.unique(y_train):
        c_idx = np.where(y_train == c)[0]
        sel = np.random.choice(c_idx, size=max(1, int(np.round(0.10 * len(c_idx)))), replace=False)
        y_semi[sel] = c
        
    lp = LabelPropagation(kernel='knn', n_neighbors=5)
    lp.fit(X_train, y_semi)
    inferred_labels = lp.transduction_
    
    pca = PCA(n_components=2, random_state=73)
    X_pca = pca.fit_transform(X_train)
    
    tsne = TSNE(n_components=2, random_state=73, perplexity=min(30, len(X_train)//4))
    X_tsne = tsne.fit_transform(X_train)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # PCA Plot
    scatter = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=inferred_labels, cmap='tab10', alpha=0.6, s=40, label='U (Inferidos)')
    l_mask = y_semi != -1
    axes[0].scatter(X_pca[l_mask, 0], X_pca[l_mask, 1], c=y_semi[l_mask], cmap='tab10', edgecolors='black', linewidths=2.0, s=120, label='L (Sementes)')
    display_title = "Wine" if name == "wine" else "Breast Cancer"
    axes[0].set_title(f'{display_title} — Projeção PCA (2D) [L vs U Inferidos]', fontsize=12)
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    axes[0].legend()
    
    # t-SNE Plot
    axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=inferred_labels, cmap='tab10', alpha=0.6, s=40, label='U (Inferidos)')
    axes[1].scatter(X_tsne[l_mask, 0], X_tsne[l_mask, 1], c=y_semi[l_mask], cmap='tab10', edgecolors='black', linewidths=2.0, s=120, label='L (Sementes)')
    axes[1].set_title(f'{display_title} — Projeção t-SNE (2D) [L vs U Inferidos]', fontsize=12)
    axes[1].set_xlabel('Dimensão 1 t-SNE')
    axes[1].set_ylabel('Dimensão 2 t-SNE')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(f'{name}_pca_tsne.png', dpi=150)
    plt.close()
    print(f'Salvo {name}_pca_tsne.png')

# 2. Stress Test Curve: F1-score vs % rótulos
with open('experimento_pratico_resultados.json', encoding='utf-8') as f:
    res = json.load(f)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fractions = [5, 10, 20]

for idx, ds in enumerate(['wine', 'breast_cancer']):
    ax = axes[idx]
    stress = res[ds]['step7_stress']['multi_seed_fractions']
    display_title = "Wine" if ds == "wine" else "Breast Cancer"
    
    for model_name, label, color, marker in [
        ('baseline', 'Supervisionado (Baseline L)', '#d9534f', 'o'),
        ('self_training', 'Self-Training (tau=0.75)', '#0275d8', 's'),
        ('label_spreading', 'Label Spreading (a=0.2)', '#5cb85c', '^')
    ]:
        means = [stress[f'{f}%'][model_name]['mean'] for f in fractions]
        stds = [stress[f'{f}%'][model_name]['std'] for f in fractions]
        ax.errorbar(fractions, means, yerr=stds, label=label, color=color, marker=marker, capsize=4, linewidth=2)
        
    ax.set_title(f'{display_title} — Curva F1-score x % Rótulos', fontsize=12)
    ax.set_xlabel('Percentual de Rótulos Revelados em Treino (%)')
    ax.set_ylabel('F1-score Médio no Teste')
    ax.set_xticks(fractions)
    ax.set_ylim([0.7, 1.05])
    ax.legend()

plt.tight_layout()
plt.savefig('f1_vs_labels_curve.png', dpi=150)
plt.close()
print('Salvo f1_vs_labels_curve.png')
