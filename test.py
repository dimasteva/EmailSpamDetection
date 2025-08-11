import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare, pearsonr, spearmanr
import scikit_posthocs as sp

df = pd.read_csv('results_summary.csv')

f1_pivot = df.pivot(index='Fold', columns='Model', values='F1 Score')
k_pivot = df.pivot(index='Fold', columns='Model', values='K Words')

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

avg_metrics = df.groupby('Model')[['F1 Score', 'K Words']].mean().reset_index()

pearson_corr, pearson_p = pearsonr(avg_metrics['F1 Score'], avg_metrics['K Words'])
spearman_corr, spearman_p = spearmanr(avg_metrics['F1 Score'], avg_metrics['K Words'])

print("\n=== Korelacije (po modelima) ===")
print(f"Pearson: r = {pearson_corr:.4f}, p = {pearson_p:.4f}")
print(f"Spearman: ρ = {spearman_corr:.4f}, p = {spearman_p:.4f}")

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