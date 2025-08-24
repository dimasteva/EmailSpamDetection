import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare, pearsonr, spearmanr
import scikit_posthocs as sp
import numpy as np
import itertools

def significant_pairs(r_mat, p_mat, alpha=0.05, name=""):
    print(f"\n=== Signifikantne korelacije ({name}, p < {alpha}) ===")
    cols = r_mat.columns
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            p_val = p_mat.iloc[i, j]
            if p_val < alpha:
                r_val = r_mat.iloc[i, j]
                print(f"{cols[i]} - {cols[j]}: r = {r_val:.4f}, p = {p_val:.4f}")

df = pd.read_csv('novicheckpoint1.csv')

f1_pivot = df.pivot(index='Fold', columns='Model', values='F1 Score')
k_pivot = df.pivot(index='Fold', columns='Model', values='K Words')

# === Friedman test ===
stat_f1, p_f1 = friedmanchisquare(*[f1_pivot[model] for model in f1_pivot.columns])
stat_k, p_k = friedmanchisquare(*[k_pivot[model] for model in k_pivot.columns])

print("=== Friedman test ===")
print(f"F1-score: statistika = {stat_f1:.4f}, p = {p_f1:.4f}")
print(f"K Words: statistika = {stat_k:.4f}, p = {p_k:.4f}")

if p_f1 < 0.05:
    nemenyi_f1 = sp.posthoc_nemenyi_friedman(f1_pivot)
    print("\n=== Nemenyi test za F1-score ===")
    print(nemenyi_f1)

if p_k < 0.05:
    nemenyi_k = sp.posthoc_nemenyi_friedman(k_pivot)
    print("\n=== Nemenyi test za broj kljucnih reci (k) ===")
    print(nemenyi_k)

# === Korelacije izmedju modela po foldovima ===
print("\n=== Korelacije izmedju modela (po foldovima) ===")

# helper za r i p matrice
def corr_pvalues(df, method="pearson"):
    cols = df.columns
    r_mat = pd.DataFrame(np.zeros((len(cols), len(cols))), columns=cols, index=cols)
    p_mat = pd.DataFrame(np.zeros((len(cols), len(cols))), columns=cols, index=cols)

    for i, j in itertools.combinations(range(len(cols)), 2):
        x, y = df.iloc[:, i], df.iloc[:, j]
        if method == "pearson":
            r, p = pearsonr(x, y)
        else:
            r, p = spearmanr(x, y)
        r_mat.iloc[i, j] = r
        r_mat.iloc[j, i] = r
        p_mat.iloc[i, j] = p
        p_mat.iloc[j, i] = p

    np.fill_diagonal(r_mat.values, 1.0)
    np.fill_diagonal(p_mat.values, 0.0)
    return r_mat, p_mat

# Pearson
f1_corr_pearson, f1_pvals_pearson = corr_pvalues(f1_pivot, method="pearson")
k_corr_pearson, k_pvals_pearson   = corr_pvalues(k_pivot, method="pearson")

print("\nPearson korelacija F1-score:\n", f1_corr_pearson)
print("\nP-vrednosti (Pearson, F1-score):\n", f1_pvals_pearson)

print("\nPearson korelacija K Words:\n", k_corr_pearson)
print("\nP-vrednosti (Pearson, K Words):\n", k_pvals_pearson)

# Spearman
f1_corr_spear, f1_pvals_spear = corr_pvalues(f1_pivot, method="spearman")
k_corr_spear, k_pvals_spear   = corr_pvalues(k_pivot, method="spearman")

print("\nSpearman korelacija F1-score:\n", f1_corr_spear)
print("\nP-vrednosti (Spearman, F1-score):\n", f1_pvals_spear)

print("\nSpearman korelacija K Words:\n", k_corr_spear)
print("\nP-vrednosti (Spearman, K Words):\n", k_pvals_spear)

significant_pairs(f1_corr_pearson, f1_pvals_pearson, name="Pearson F1-score")
significant_pairs(k_corr_pearson,  k_pvals_pearson,  name="Pearson K Words")

significant_pairs(f1_corr_spear, f1_pvals_spear, name="Spearman F1-score")
significant_pairs(k_corr_spear,  k_pvals_spear,  name="Spearman K Words")

# === Vizualizacije Pearson i Spearman r matrica ===
plt.figure(figsize=(8,6))
sns.heatmap(f1_corr_pearson, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Pearson korelacija F1-score izmedju modela")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,6))
sns.heatmap(k_corr_pearson, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Pearson korelacija K Words izmedju modela")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,6))
sns.heatmap(f1_corr_spear, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Spearman korelacija F1-score izmedju modela")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,6))
sns.heatmap(k_corr_spear, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Spearman korelacija K Words izmedju modela")
plt.tight_layout()
plt.show()

# === Prosecne metrike po modelima ===
avg_metrics = df.groupby('Model')[['F1 Score', 'K Words']].mean().reset_index()

pearson_corr, pearson_p = pearsonr(avg_metrics['F1 Score'], avg_metrics['K Words'])
spearman_corr, spearman_p = spearmanr(avg_metrics['F1 Score'], avg_metrics['K Words'])

print("\n=== Korelacije (po modelima) ===")
print(f"Pearson: r = {pearson_corr:.4f}, p = {pearson_p:.4f}")
print(f"Spearman: ρ = {spearman_corr:.4f}, p = {spearman_p:.4f}")

# === Vizualizacije distribucija ===
sns.boxplot(data=f1_pivot)
plt.title("F1-score po modelima")
plt.ylabel("F1 Score")
plt.xlabel("Model")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

sns.boxplot(data=k_pivot)
plt.title("Broj kljucnih reci (k) po modelima")
plt.ylabel("K Words")
plt.xlabel("Model")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

sns.scatterplot(data=avg_metrics, x='K Words', y='F1 Score', hue='Model', s=100)
plt.title("Korelacija izmedju interpretabilnosti i tacnosti")
plt.xlabel("Prosecan broj kljucnih reci (k)")
plt.ylabel("Prosecan F1-score")
plt.tight_layout()
plt.show()
