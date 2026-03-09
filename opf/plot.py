import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Data
# -----------------------------
classes = [f"Class {i}" for i in range(1, 17)]

gan_mean = np.array([
    1.8617, 0.7159, 0.7520, 1.1585,
    0.9322, 0.4052, 1.0044, 3.9781,
    0.4037, 0.5744, 0.4027, 4.3256,
    1.1684, 0.6416, 0.6794, 2.0182
])
gan_std = np.array([
    0.7521, 0.2131, 0.3585, 0.6737,
    0.6410, 0.0019, 0.7201, 0.0917,
    0.0017, 0.0062, 0.0009, 0.1449,
    0.1502, 0.0731, 0.1236, 0.7992
])

noise_mean = np.array([
    0.6418, 0.5837, 0.8485, 0.4094,
    0.4077, 0.4047, 0.4127, 1.3252,
    0.4057, 0.6015, 0.4056, 2.0337,
    0.5313, 0.5256, 0.5832, 0.8872
])
noise_std = np.array([
    0.4725, 0.0143, 0.3331, 0.0025,
    0.0042, 0.0012, 0.0144, 0.0459,
    0.0003, 0.0428, 0.0021, 0.4812,
    0.0468, 0.0857, 0.0100, 0.5899
])

noopt_mean = np.array([
    0.6574, 1.0448, 1.3153, 0.9414,
    0.8961, 0.5585, 0.3788, 1.4934,
    1.4805, 0.7788, 1.3263, 3.5829,
    0.7950, 0.5040, 0.6253, 1.9330
])
noopt_std = np.array([
    0.5593, 0.1555, 0.0771, 0.6897,
    0.6390, 0.3570, 0.0013, 0.5739,
    0.6149, 0.1844, 0.7714, 0.1463,
    0.1602, 0.1132, 0.1608, 0.3200
])

full_mean = np.array([
    0.1163, 0.2947, 0.3060, 0.0850,
    0.0764, 0.1179, 0.1159, 1.0179,
    0.1081, 0.2863, 0.1233, 1.7312,
    0.2585, 0.3235, 0.3030, 0.1375
])
full_std = np.array([
    0.0217, 0.0086, 0.0128, 0.0175,
    0.0153, 0.0081, 0.0072, 0.0504,
    0.0261, 0.0094, 0.0059, 0.0568,
    0.0230, 0.0129, 0.0225, 0.0079
])

# -----------------------------
# Plot
# -----------------------------
x = np.arange(len(classes))
width = 0.18

fig, ax = plt.subplots(figsize=(12, 4.8))
error_kw = dict(elinewidth=1, capsize=2, capthick=1)

ax.bar(
    x - 1.5 * width, gan_mean, width,
    yerr=gan_std, label='GAN',
    color='tab:blue', error_kw=error_kw
)
ax.bar(
    x - 0.5 * width, noise_mean, width,
    yerr=noise_std, label='Vanilla Diffusion',
    color='tab:orange', error_kw=error_kw
)
ax.bar(
    x + 0.5 * width, noopt_mean, width,
    yerr=noopt_std, label='Diffusion w/o opt loss',
    color='tab:green', error_kw=error_kw
)
ax.bar(
    x + 1.5 * width, full_mean, width,
    yerr=full_std, label='Full DiT2SP',
    color='tab:red', error_kw=error_kw
)

# -----------------------------
# Style
# -----------------------------
ax.set_xticks(x)
ax.set_xticklabels(classes, rotation=30, ha='right', fontsize=12)  # Increase x-axis tick label size.
ax.tick_params(axis='y', labelsize=12)  # Increase y-axis tick label size.

# If you want axis titles, uncomment the next two lines.
# ax.set_xlabel("Classes", fontsize=14)
# ax.set_ylabel("Value", fontsize=14)

ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, 1.18),
    ncol=4,
    frameon=False,
    fontsize=12,          # Increase legend font size.
    handlelength=1.4,
    columnspacing=1.5
)

ax.yaxis.grid(True, linestyle='--', linewidth=0.8, alpha=0.35)
ax.set_axisbelow(True)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ymax = max(
    np.max(gan_mean + gan_std),
    np.max(noise_mean + noise_std),
    np.max(noopt_mean + noopt_std),
    np.max(full_mean + full_std)
)
ax.set_ylim(0, ymax * 1.12)

plt.tight_layout()

plt.savefig("err.pdf", dpi=300, bbox_inches="tight")
plt.show()
