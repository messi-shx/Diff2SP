import numpy as np
import pandas as pd

def distance(cor1, cor2):
    A = np.asarray(cor1, dtype=float)
    B = np.asarray(cor2, dtype=float)
    if A.shape != B.shape:
        raise ValueError(f"Shape mismatch: {A.shape} vs {B.shape}")

    A = 0.5 * (A + A.T)
    B = 0.5 * (B + B.T)

    mask = np.triu(np.ones(A.shape, dtype=bool), k=1)
    d = (A - B)[mask]
    d = d[~np.isnan(d)]
    if d.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(d * d)))

path = "gan_sample_1.csv"

df_real = pd.read_csv("/home/sun1321/src/diff2sp_new/data/data.csv")
corr_real = df_real.corr(method="pearson")

# Load all generated samples at once.
df_all = pd.read_csv(path, header=None)

num = [3, 5, 10, 30, 70, 100, 200, 1000]
for n in num:
    n_eff = min(n, len(df_all))  # Prevent n from exceeding the total number of rows.
    df = df_all.sample(n=n_eff, replace=False, random_state=42)  # Random sampling without replacement.
    corr = df.corr(method="pearson")
    print(f"{n_eff} random samples with correlation distance {distance(corr, corr_real)}")
