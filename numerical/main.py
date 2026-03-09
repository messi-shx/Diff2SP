import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from typing import Optional

from diffusion import DDPM
from options import Options


# =========================
# Fixed config (no CLI args except --mode)
# =========================
DATA_FILE = "/home/sun1321/src/diff2sp_new/data/data.csv"
LABEL_FILE = "/home/sun1321/src/diff2sp_new/data/label.csv"

BATCH_SIZE = 32
SHUFFLE = True
STRIDE = 1

# Keep train behavior unchanged
MAX_ROWS_TRAIN = 200000

# Sample config (you can edit here)
CKPT_PATH = "/home/sun1321/src/diff2sp_new/output_model/epoch200.pt"
N_SAMPLES = 1000
OUT_DIR = "/home/sun1321/src/diff2sp_new"
FILTER_ALL_ZERO_ROWS = True

# Normalization stats (min/max) will be saved next to the checkpoint by default.
NORM_STATS_PATH = os.path.join(os.path.dirname(CKPT_PATH), "norm_stats.npz")

# If you have a *raw* (de-normalized) version of the data CSV, set this to its path
# so we can compute min/max in original units and automatically recover them at sampling.
# If left as None, stats will be computed from DATA_FILE.
RAW_DATA_FILE_FOR_STATS = None


class loadData(Dataset):
    """
    Long-format time series -> sliding windows
    - data.csv is T x (1 + D): first col is timestamp/index, drop it
    - label.csv is T x 1: numeric labels per timestep
    Output:
      features: (N, seq_len, input_dim)
      labels:   (N, num_classes) one-hot
    """
    def __init__(
        self,
        data_file: str,
        label_file: str,
        opt: Options,
        max_rows: Optional[int] = None,
        stride: int = 1
    ):
        df_x = pd.read_csv(data_file)
        df_y = pd.read_csv(label_file)

        if max_rows is not None:
            df_x = df_x.iloc[:max_rows, :]
            df_y = df_y.iloc[:max_rows, :]

        df_x = df_x.apply(pd.to_numeric, errors="coerce")
        y_raw = pd.to_numeric(df_y.iloc[:, 0], errors="coerce")

        # drop invalid rows
        valid = (~y_raw.isna()) & (~df_x.isna().any(axis=1))
        df_x = df_x.loc[valid].reset_index(drop=True)
        y_raw = y_raw.loc[valid].reset_index(drop=True)

        X = df_x.values.astype(np.float32)  # (T, D)
        T, D = X.shape

        # =========================
        # Optional min-max normalization to [0,1] (and stats save/load)
        # =========================
        # This lets you automatically recover de-normalized generated values
        # during sampling, without manually copy-pasting min/max.
        need_stats = (
            getattr(opt, "load_norm_stats", False)
            or getattr(opt, "normalize", "none") == "minmax"
        )
        if need_stats:
            stats_path = getattr(opt, "norm_stats_path", None)
            data_min = data_max = None

            # 1) Try loading saved stats first
            if stats_path is not None and os.path.exists(stats_path):
                stats = np.load(stats_path)
                data_min = stats["data_min"].astype(np.float32)
                data_max = stats["data_max"].astype(np.float32)
            else:
                # 2) Compute stats from a raw CSV (if provided) or from this CSV
                stats_source = getattr(opt, "raw_data_file_for_stats", None) or data_file
                df_stats = pd.read_csv(stats_source).apply(pd.to_numeric, errors="coerce")
                df_stats = df_stats.dropna(axis=0, how="any")
                Xs = df_stats.values.astype(np.float32)
                data_min = np.min(Xs, axis=0)
                data_max = np.max(Xs, axis=0)

                if stats_path is not None:
                    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
                    np.savez(stats_path, data_min=data_min, data_max=data_max)
                    print(f"[NormStats] saved: {stats_path}")

            if data_min.shape[0] != D or data_max.shape[0] != D:
                raise ValueError(
                    f"norm_stats feature dim mismatch: stats has {data_min.shape[0]} cols, "
                    f"but data has {D} cols. If you have a timestamp/index column, "
                    "make sure you drop it consistently in both the data and the stats source."
                )

            # Expose stats on opt so DDPM.sample can de-normalize
            opt.data_min = data_min
            opt.data_max = data_max

            # Only apply normalization to X if requested
            if getattr(opt, "normalize", "none") == "minmax":
                denom = np.maximum(data_max - data_min, getattr(opt, "norm_eps", 1e-8)).astype(np.float32)
                X = (X - data_min) / denom

        # auto-set input_dim from CSV
        opt.input_dim = D

        # infer classes
        unique_vals = np.sort(y_raw.unique())
        self.class_values = unique_vals.tolist()
        self.num_classes = len(self.class_values)
        val_to_idx = {v: i for i, v in enumerate(unique_vals)}
        y_idx = np.array([val_to_idx[v] for v in y_raw.to_numpy()], dtype=np.int64)

        # sliding windows: (N, seq_len, input_dim)
        L = opt.seq_len
        if T < L:
            raise ValueError(f"Not enough rows: T={T} but seq_len={L}")

        windows = []
        win_labels = []
        for start in range(0, T - L + 1, stride):
            end = start + L
            windows.append(X[start:end, :])      # (L, D)
            win_labels.append(y_idx[end - 1])    # label of last timestep in window

        self.features = torch.tensor(np.stack(windows, axis=0), dtype=torch.float32)  # (N, L, D)
        self.labels = F.one_hot(torch.tensor(win_labels), num_classes=self.num_classes).float()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {"features": self.features[idx], "labels": self.labels[idx]}


def build_dataloader(
    data_file: str,
    label_file: str,
    opt: Options,
    batch_size: int,
    shuffle: bool,
    max_rows: Optional[int] = None,
    stride: int = 1
):
    dataset = loadData(data_file, label_file, opt, max_rows=max_rows, stride=stride)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def run_train(opt: Options, dataloader: DataLoader):
    model = DDPM(opt, dataloader)
    model.train()


def run_sample(
    opt: Options,
    dataloader: DataLoader,
    ckpt: str,
    n_samples: int,
    condition_idx: int,
    out_dir: str,
    out_file: str,
    filter_all_zero_rows: bool = True,
):
    model = DDPM(opt, dataloader)
    os.makedirs(out_dir, exist_ok=True)

    # condition vector length must match opt.cond_dim
    condition = np.zeros(opt.cond_dim, dtype=int)
    if not (0 <= condition_idx < opt.cond_dim):
        raise ValueError(f"condition_idx must be in [0, {opt.cond_dim - 1}], got {condition_idx}")
    condition[condition_idx] = 1

    samples_out = model.sample(ckpt, n_samples=n_samples, condition=condition)

    # sample() may return either a list (legacy) or a dict with both normalized
    # and de-normalized outputs.
    if isinstance(samples_out, dict):
        samples_norm = samples_out.get("norm", [])
        samples_denorm = samples_out.get("denorm", None)
    else:
        samples_norm = samples_out
        samples_denorm = None

    def _combine_and_filter(sample_list):
        combined = torch.cat(
            [torch.tensor(s) if isinstance(s, np.ndarray) else s for s in sample_list],
            dim=0,
        )
        if filter_all_zero_rows:
            combined = combined[~torch.all(combined == 0, dim=1)]
        return combined

    # Save normalized output (what the model actually generates, typically in [0,1])
    norm_path = os.path.join(out_dir, out_file.replace(".csv", "_norm.csv"))
    norm_combined = _combine_and_filter(samples_norm)
    np.savetxt(norm_path, norm_combined.cpu().numpy(), delimiter=",", fmt="%.6f")
    print(f"Saved normalized samples to: {norm_path}")

    # Save de-normalized output (original units) if stats exist
    if samples_denorm is not None:
        denorm_path = os.path.join(out_dir, out_file.replace(".csv", "_denorm.csv"))
        denorm_combined = _combine_and_filter(samples_denorm)
        np.savetxt(denorm_path, denorm_combined.cpu().numpy(), delimiter=",", fmt="%.6f")
        print(f"Saved de-normalized samples to: {denorm_path}")
    else:
        print("[Note] No de-normalized output produced (missing min/max stats or disabled).")


def parse_args():
    p = argparse.ArgumentParser("Diffusion training/sampling")
    p.add_argument("--mode", choices=["train", "sample"], required=True)

    p.add_argument("--sample_class", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    isTrain = (args.mode == "train")

    # Options: keep exactly the same behavior as before
    opt = Options("diffusion", isTrain)

    # Normalization stats configuration (used for automatic de-normalization when sampling)
    opt.norm_stats_path = NORM_STATS_PATH
    opt.raw_data_file_for_stats = RAW_DATA_FILE_FOR_STATS

    # If you want the code to handle normalization+automatic de-normalization,
    # set normalize="minmax". Otherwise keep "none".
    # NOTE: if DATA_FILE is already normalized to [0,1], leaving this as "none" is OK.
    # If you want automatic recovery to original units, set RAW_DATA_FILE_FOR_STATS to your raw CSV
    # and set opt.normalize = "minmax".
    opt.norm_stats_path = NORM_STATS_PATH
    opt.raw_data_file_for_stats = RAW_DATA_FILE_FOR_STATS

    if args.mode == "train":
        dataloader = build_dataloader(
            DATA_FILE, LABEL_FILE, opt,
            batch_size=BATCH_SIZE,
            shuffle=SHUFFLE,
            max_rows=MAX_ROWS_TRAIN,
            stride=STRIDE
        )

        # set cond_dim from dataset (same logic as before)
        opt.cond_dim = dataloader.dataset.num_classes
        print("Detected classes:", dataloader.dataset.class_values)
        print("cond_dim set to:", opt.cond_dim)

        run_train(opt, dataloader)
        return

    # sample
    dataloader = build_dataloader(
        DATA_FILE, LABEL_FILE, opt,
        batch_size=BATCH_SIZE,
        shuffle=SHUFFLE,
        max_rows=None,
        stride=STRIDE
    )

    # set cond_dim for sampling too (important!)
    opt.cond_dim = dataloader.dataset.num_classes
    class_values = dataloader.dataset.class_values
    print("Detected classes:", class_values)
    print("cond_dim set to:", opt.cond_dim)

    if args.sample_class is None:
        # default: use first class
        condition_idx = 0
        class_value = class_values[0]
    else:
        sc = args.sample_class
        if sc in class_values:
            # user provided original label value (recommended)
            class_value = sc
            condition_idx = class_values.index(sc)
        elif 0 <= sc < opt.cond_dim:
            # user provided an index
            condition_idx = sc
            class_value = class_values[sc]
        else:
            raise ValueError(
                f"--sample_class={sc} is neither a valid class value {class_values} "
                f"nor a valid index [0, {opt.cond_dim - 1}]."
            )

    out_file = f"sample_{class_value}.csv"
    print(f"[Sampling] condition_idx={condition_idx}, class_value={class_value}, out_file={out_file}")

    run_sample(
        opt=opt,
        dataloader=dataloader,
        ckpt=CKPT_PATH,
        n_samples=N_SAMPLES,
        condition_idx=condition_idx,
        out_dir=OUT_DIR,
        out_file=out_file,
        filter_all_zero_rows=FILTER_ALL_ZERO_ROWS,
    )


if __name__ == "__main__":
    main()
