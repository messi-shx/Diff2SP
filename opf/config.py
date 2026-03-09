# config.py
from dataclasses import dataclass

@dataclass
class TrainConfig:
    # data
    data_dir: str = "./data"
    merged_csv: str = "merged_data.csv"
    label_csv: str = "label_one.csv"
    n_rows: int = 105120
    num_classes: int = 16

    # train
    seed: int = 42
    device: str = "cpu"
    batch_size: int = 256
    num_workers: int = 0
    lr: float = 2e-4
    weight_decay: float = 1e-4
    epochs: int = 50
    grad_clip: float = 1.0
    log_every: int = 100
    save_every_epochs: int = 5

    # diffusion
    x_dim: int = 18
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2

    # model (transformer)
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    dropout: float = 0.1

    # losses (ablation)
    lambda_noise: float = 1.0
    lambda_recon: float = 1.0
    lambda_opt: float = 30.0
    warmup_epochs: int = 5
    opt_loss_t_frac: float = 0.2   # Compute the optimization loss only at smaller t for better stability.
    opt_min_scenarios: int = 4     # Require at least this many same-class samples per batch for the Section 5.2 style fixed first-stage OPF.

    # OPF (soft-DCOPF)
    baseMVA: float = 100.0
    opf_dtype: str = "float64"     # qpth is more stable in float64.
    eps_q: float = 1e-6            # Make Q strictly positive definite.
    Ms: float = 1e3                # load shedding penalty
    Mr: float = 5e2                # curtailment penalty
    Mu: float = 1e2                # line slack penalty

    # A matrix
    a_seed: int = 123
    a_init_mode: str = "aligned"         # aligned | random | biased
    a_dirichlet_alpha0: float = 200.0  # Larger values keep A closer to the base load distribution.
    a_random_alpha: float = 2.0        # Dirichlet concentration used in random/biased modes.
    a_bias_mix: float = 0.15           # Amount of base_pd weight retained in biased mode.
    lr_A: float = 5e-4

    # GAN
    gan_z_dim: int = 32
    gan_g_hidden: int = 128
    gan_d_hidden: int = 128
    gan_lr_d: float = 2e-4

    # output
    out_dir: str = "./output_model"
    ckpt_name: str = "model.ckpt"
    a_name: str = "A.npy"
    scaler_name: str = "scaler.json"
