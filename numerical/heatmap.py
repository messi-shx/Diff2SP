import pandas as pd
import matplotlib.pyplot as plt

path = "sample_1_denorm.csv"


n = 500
# Many generated files are saved without a header, so header=None is recommended.
df = pd.read_csv(path, header=None).head(n)

# If there are non-feature columns (for example timestamps or labels), drop them first:
# df = df.drop(columns=[0])     # drop the first column
# df = df.iloc[:, :-1]          # drop the last column if it is a label

corr = df.corr(method="pearson")

# Save the correlation matrix.
# corr.to_csv("correlation_matrix.csv", index=True)

# Plot the heatmap with matplotlib instead of seaborn.
plt.figure(figsize=(8, 6))
# plt.imshow(corr.values, vmin=-1, vmax=1, aspect="equal")
plt.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
plt.axis("off")

ax = plt.gca() 
for sp in ax.spines.values():
    sp.set_visible(False)

cbar = plt.colorbar()
cbar.outline.set_visible(False)

# plt.colorbar()
plt.xticks(range(corr.shape[0]), corr.columns, rotation=45, ha="right")
plt.yticks(range(corr.shape[0]), corr.index)
# plt.title("Correlation Heatmap (Pearson)")
plt.tight_layout()
plt.savefig(f"correlation_heatmap_{n}.pdf", dpi=200)
plt.show()
