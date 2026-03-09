# main.py
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import TrainConfig
from data import load_csv_data, Diff2SPDataset
from utils import (
    set_seed, ensure_dir,
    StandardScaler, save_json,
    EMA, save_checkpoint, load_checkpoint
)

from models.transformer import TransformerDenoiser
from models.diffusion import DDPM, diffusion_losses
from models.gan import ConditionalGenerator, ConditionalDiscriminator
from opf.dc_opf_qpth import SoftDCOPF_QPTH


# ---------------------------
# Naming helpers
# ---------------------------
def normalize_ablation(ablation: str) -> str:
    """
    normalize synonyms:
      full -> full
      noopt/no_opt -> no_opt
      noise/noise_only -> noise_only
      gan -> gan
      norecon/no_recon -> no_recon
    """
    a = (ablation or "full").strip().lower()
    if a in ["full", "model_full"]:
        return "full"
    if a in ["noopt", "no_opt", "model_noopt"]:
        return "no_opt"
    if a in ["noise", "noise_only", "model_noise"]:
        return "noise_only"
    if a in ["gan", "model_gan"]:
        return "gan"
    if a in ["norecon", "no_recon", "model_norecon"]:
        return "no_recon"
    raise ValueError(f"Unknown ablation: {ablation}")


def ablation_to_tag(ablation_norm: str) -> str:
    """
    tag used in filenames
    """
    if ablation_norm == "full":
        return "full"
    if ablation_norm == "no_opt":
        return "noopt"
    if ablation_norm == "noise_only":
        return "noise"
    if ablation_norm == "gan":
        return "gan"
    if ablation_norm == "no_recon":
        return "norecon"
    return ablation_norm


def ckpt_filename(tag: str) -> str:
    return f"model_{tag}.ckpt"

def a_init_filename(tag: str) -> str:
    return f"A_{tag}_init.npy"

def sample_filename(tag: str, class_num_1based: int) -> str:
    return f"sample_{tag}_pd{class_num_1based}.csv"


def a_filename(tag: str) -> str:
    return f"A_{tag}.npy"


# ---------------------------
# A matrix initializer (A0)
# ---------------------------
def build_A_30x18(
    base_pd_30: np.ndarray,
    x_train_18: np.ndarray,
    seed: int = 123,
    alpha0: float = 200.0,
    mode: str = "aligned",
    random_alpha: float = 2.0,
    bias_mix: float = 0.15,
):
    """
    Build an initial A0: (30,18), nonneg, each column sums to 'scale' (same across columns).
    pd = x @ A^T  (x:18)
    mode:
      - aligned: close to base_pd distribution
      - random: ignore base_pd, use near-uniform random columns
      - biased: mostly random, keep a small amount of base_pd bias
    """
    rng = np.random.default_rng(seed)
    base_pd_30 = np.maximum(base_pd_30.astype(np.float64), 1e-6)
    w_base = base_pd_30 / base_pd_30.sum()  # bus weight

    mean_total_x = float(np.mean(np.sum(x_train_18.astype(np.float64), axis=1)))
    target_total = float(np.sum(base_pd_30))
    scale = target_total / max(mean_total_x, 1e-6)

    A = np.zeros((30, 18), dtype=np.float64)
    mode = (mode or "aligned").strip().lower()

    if mode == "aligned":
        alpha = np.maximum(w_base * alpha0, 1e-3)
    elif mode == "random":
        alpha = np.full_like(w_base, max(float(random_alpha), 1e-3))
    elif mode == "biased":
        rand_w = rng.random(w_base.shape[0]).astype(np.float64)
        rand_w /= rand_w.sum()
        mix = float(np.clip(bias_mix, 0.0, 1.0))
        w_mix = mix * w_base + (1.0 - mix) * rand_w
        alpha = np.maximum(w_mix * alpha0, max(float(random_alpha) * 0.1, 1e-3))
    else:
        raise ValueError(f"Unknown a_init_mode: {mode}")

    for j in range(18):
        col = rng.dirichlet(alpha)          # sums to 1
        A[:, j] = col * scale               # sums to scale

    return A.astype(np.float32)


def get_device(cfg: TrainConfig):
    if cfg.device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------
# Train
# ---------------------------
def train(cfg: TrainConfig, ablation: str):
    ablation_norm = normalize_ablation(ablation)
    tag = ablation_to_tag(ablation_norm)

    set_seed(cfg.seed)
    device = get_device(cfg)
    ensure_dir(cfg.out_dir)

    # load data
    x, y = load_csv_data(cfg.data_dir, cfg.merged_csv, cfg.label_csv, cfg.n_rows)
    print(f"[INFO] data loaded: x={x.shape}, y={y.shape}, y_min={y.min()}, y_max={y.max()}", flush=True)

    # scaler
    scaler = StandardScaler.fit(x)
    x_norm = scaler.transform(x)

    # OPF solver (double is more stable)
    dtype = torch.float64 if getattr(cfg, "opf_dtype", "float64") == "float64" else torch.float32
    print("[INFO] building OPF solver ...", flush=True)
    opf = SoftDCOPF_QPTH(
        baseMVA=cfg.baseMVA,
        eps_q=cfg.eps_q,
        Ms=cfg.Ms, Mr=cfg.Mr, Mu=cfg.Mu,
        dtype=dtype,
        device=str(device),
    )
    print("[INFO] OPF solver ready.", flush=True)

    # initial A0 from base pd and real data statistics
    base_pd = opf.get_base_pd_mw()  # (30,)
    A0 = build_A_30x18(
        base_pd,
        x,
        seed=cfg.a_seed,
        alpha0=cfg.a_dirichlet_alpha0,
        mode=cfg.a_init_mode,
        random_alpha=cfg.a_random_alpha,
        bias_mix=cfg.a_bias_mix,
    )  # (30,18)
    print(f"[INFO] A init mode: {cfg.a_init_mode}", flush=True)
    # Fix scale to prevent trivial scaling cheats
    scale = float(A0[:, 0].sum())
    prob0 = A0 / (A0.sum(axis=0, keepdims=True) + 1e-12)  # each column sums to 1

    if tag == "full":
        # Trainable A via logits (softmax over buses per column)
        A_logits = nn.Parameter(torch.log(torch.tensor(prob0, device=device, dtype=torch.float32) + 1e-12))
        A_scale = torch.tensor(scale, device=device, dtype=torch.float32)

        def A_current():
            # (30,18), each column sums to scale, nonneg
            return torch.softmax(A_logits, dim=0) * A_scale
    else:
        A_logits = None
        A_scale = None
        A_fixed = torch.tensor(A0, device=device, dtype=torch.float32)

        def A_current():
            return A_fixed

    # Save initial A snapshot only for full, where A is actually optimized.
    A_init = A_current().detach().cpu().numpy()
    if tag == "full":
        init_path = os.path.join(cfg.out_dir, a_init_filename(tag))
        if not os.path.exists(init_path):
            np.save(init_path, A_init)

    # Current/final A used by this ablation.
    np.save(os.path.join(cfg.out_dir, a_filename(tag)), A_init)

    # dataloader
    ds = Diff2SPDataset(x_norm, y)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, drop_last=True)

    # model
    net = TransformerDenoiser(
        x_dim=cfg.x_dim,
        num_classes=cfg.num_classes,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        dropout=cfg.dropout,
    ).to(device)

    ddpm = DDPM(net, cfg.timesteps, cfg.beta_start, cfg.beta_end).to(device)

    # optimizer: diffusion params always; A_logits only for full
    lr_A = float(getattr(cfg, "lr_A", 5e-4))  # if you didn't add lr_A to config.py, fallback here
    param_groups = [{"params": ddpm.parameters(), "lr": cfg.lr}]
    if tag == "full":
        param_groups.append({"params": [A_logits], "lr": lr_A})
    opt = torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)

    ema = EMA(ddpm, decay=0.999)

    # ablation loss weights (training objective)
    lambda_noise = cfg.lambda_noise
    lambda_recon = cfg.lambda_recon
    lambda_opt_base = cfg.lambda_opt

    if ablation_norm == "full":
        pass
    elif ablation_norm == "no_opt":
        lambda_opt_base = 0.0
    elif ablation_norm == "no_recon":
        lambda_recon = 0.0
    elif ablation_norm == "noise_only":
        lambda_recon = 0.0
        lambda_opt_base = 0.0
    else:
        raise ValueError(f"Unknown ablation_norm: {ablation_norm}")

    # scaler inverse
    def scaler_inv_fn(x_t: torch.Tensor):
        mean = torch.tensor(scaler.mean, device=x_t.device, dtype=x_t.dtype)
        std = torch.tensor(scaler.std, device=x_t.device, dtype=x_t.dtype)
        return x_t * (std + 1e-8) + mean

    # ---- warmup schedule for FULL (recommended) ----
    warmup_epochs = int(getattr(cfg, "warmup_epochs", 5))
    lambda_opt_final = float(getattr(cfg, "lambda_opt", 1.0))
    # Use small t for opt branch (more stable)
    opt_t_max = 10

    global_step = 0
    for epoch in range(1, cfg.epochs + 1):
        ddpm.train()

        if tag == "full":
            cur_lambda_opt = 0.0 if epoch <= warmup_epochs else lambda_opt_final
        else:
            cur_lambda_opt = lambda_opt_base

        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)

            A_step = A_current()  # (30,18) may require grad if full

            # IMPORTANT: opt_mode for full uses fixed_pg to align formula(20),
            # other tags can still compute metrics but won't backprop (lambda_opt=0 anyway).
            loss_dict = diffusion_losses(
                ddpm=ddpm,
                x0=xb,
                y=yb,
                scaler_inv_fn=scaler_inv_fn,
                A_30x18=A_step,
                opf_solver=opf,
                lambda_noise=lambda_noise,
                lambda_recon=lambda_recon,
                lambda_opt=cur_lambda_opt,
                opt_t_max=opt_t_max,
                opt_mode=("fixed_pg" if tag == "full" else "optimal_gap"),
                opt_scenarios=32,
                opt_min_scenarios=cfg.opt_min_scenarios,
                pha_rho=1.0,
                pha_max_iter=10,
                pha_tol=1e-2,
                qp_chunk=128,
            )

            loss = loss_dict["total"]

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ddpm.parameters(), cfg.grad_clip)
            opt.step()
            ema.update(ddpm)

            if global_step % cfg.log_every == 0:
                print(
                    f"[{tag}] [epoch {epoch:03d} step {global_step:06d}] "
                    f"loss={loss.item():.6f} "
                    f"noise={loss_dict['L_noise'].item():.6f} "
                    f"recon={loss_dict['L_recon'].item():.6f} "
                    f"opt={loss_dict['L_opt'].item():.6f} "
                    f"lambda_opt={cur_lambda_opt:.4f}",
                    flush=True
                )
            global_step += 1

        # save periodically
        if epoch % cfg.save_every_epochs == 0 or epoch == cfg.epochs:
            # save current A for this tag
            np.save(os.path.join(cfg.out_dir, a_filename(tag)), A_current().detach().cpu().numpy())

            ckpt_path = os.path.join(cfg.out_dir, ckpt_filename(tag))
            payload = {
                "cfg": cfg.__dict__,
                "ablation": ablation_norm,
                "tag": tag,
                "model": ddpm.state_dict(),
                "ema": ema.shadow,
                "scaler": scaler.to_json(),
                "A_tag": tag,
                "A_scale": float(scale),
            }
            if tag == "full":
                payload["A_logits"] = A_logits.detach().cpu()

            save_checkpoint(ckpt_path, payload)
            save_json(scaler.to_json(), os.path.join(cfg.out_dir, cfg.scaler_name))
            print(f"[SAVE] {ckpt_path}", flush=True)


def train_gan(cfg: TrainConfig, ablation: str = "gan"):
    ablation_norm = normalize_ablation(ablation)
    if ablation_norm != "gan":
        raise ValueError(f"train_gan only supports gan ablation, got {ablation}")
    tag = ablation_to_tag(ablation_norm)

    set_seed(cfg.seed)
    device = get_device(cfg)
    ensure_dir(cfg.out_dir)

    # load data
    x, y = load_csv_data(cfg.data_dir, cfg.merged_csv, cfg.label_csv, cfg.n_rows)
    print(f"[INFO] data loaded: x={x.shape}, y={y.shape}, y_min={y.min()}, y_max={y.max()}", flush=True)

    # scaler
    scaler = StandardScaler.fit(x)
    x_norm = scaler.transform(x)

    # Build fixed A exactly like non-full/noise path.
    dtype = torch.float64 if getattr(cfg, "opf_dtype", "float64") == "float64" else torch.float32
    print("[INFO] building OPF solver (for A init) ...", flush=True)
    opf = SoftDCOPF_QPTH(
        baseMVA=cfg.baseMVA,
        eps_q=cfg.eps_q,
        Ms=cfg.Ms, Mr=cfg.Mr, Mu=cfg.Mu,
        dtype=dtype,
        device=str(device),
    )
    print("[INFO] OPF solver ready.", flush=True)

    base_pd = opf.get_base_pd_mw()
    A0 = build_A_30x18(
        base_pd,
        x,
        seed=cfg.a_seed,
        alpha0=cfg.a_dirichlet_alpha0,
        mode=cfg.a_init_mode,
        random_alpha=cfg.a_random_alpha,
        bias_mix=cfg.a_bias_mix,
    )
    np.save(os.path.join(cfg.out_dir, a_filename(tag)), A0)
    print(f"[INFO] A init mode: {cfg.a_init_mode}", flush=True)

    # dataloader
    ds = Diff2SPDataset(x_norm, y)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, drop_last=True)

    # lightweight conditional GAN (intentionally simple, typically weaker than diffusion baseline)
    z_dim = int(getattr(cfg, "gan_z_dim", 32))
    g_hidden = int(getattr(cfg, "gan_g_hidden", 128))
    d_hidden = int(getattr(cfg, "gan_d_hidden", 128))
    lr_d = float(getattr(cfg, "gan_lr_d", cfg.lr))

    G = ConditionalGenerator(z_dim=z_dim, x_dim=cfg.x_dim, num_classes=cfg.num_classes, hidden_dim=g_hidden).to(device)
    D = ConditionalDiscriminator(x_dim=cfg.x_dim, num_classes=cfg.num_classes, hidden_dim=d_hidden).to(device)

    opt_g = torch.optim.AdamW(G.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    opt_d = torch.optim.AdamW(D.parameters(), lr=lr_d, weight_decay=cfg.weight_decay)
    bce = nn.BCEWithLogitsLoss()

    global_step = 0
    for epoch in range(1, cfg.epochs + 1):
        G.train()
        D.train()

        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            bsz = xb.size(0)

            real_target = torch.full((bsz,), 0.9, device=device)  # mild smoothing
            fake_target = torch.zeros((bsz,), device=device)

            # D step
            z = torch.randn((bsz, z_dim), device=device)
            x_fake = G(z, yb).detach()
            d_real = D(xb, yb)
            d_fake = D(x_fake, yb)
            d_loss = bce(d_real, real_target) + bce(d_fake, fake_target)

            opt_d.zero_grad(set_to_none=True)
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(D.parameters(), cfg.grad_clip)
            opt_d.step()

            # G step
            z2 = torch.randn((bsz, z_dim), device=device)
            x_gen = G(z2, yb)
            g_adv = bce(D(x_gen, yb), torch.ones((bsz,), device=device))
            g_loss = g_adv

            opt_g.zero_grad(set_to_none=True)
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), cfg.grad_clip)
            opt_g.step()

            if global_step % cfg.log_every == 0:
                print(
                    f"[{tag}] [epoch {epoch:03d} step {global_step:06d}] "
                    f"g_loss={g_loss.item():.6f} d_loss={d_loss.item():.6f}",
                    flush=True
                )
            global_step += 1

        if epoch % cfg.save_every_epochs == 0 or epoch == cfg.epochs:
            ckpt_path = os.path.join(cfg.out_dir, ckpt_filename(tag))
            payload = {
                "cfg": cfg.__dict__,
                "ablation": ablation_norm,
                "tag": tag,
                "model_type": "gan",
                "G": G.state_dict(),
                "D": D.state_dict(),
                "scaler": scaler.to_json(),
                "gan_z_dim": z_dim,
                "gan_g_hidden": g_hidden,
                "gan_d_hidden": d_hidden,
            }
            save_checkpoint(ckpt_path, payload)
            save_json(scaler.to_json(), os.path.join(cfg.out_dir, cfg.scaler_name))
            print(f"[SAVE] {ckpt_path}", flush=True)


# ---------------------------
# Sample
# ---------------------------
def sample(cfg: TrainConfig, n: int, label: int, ablation: str, use_ema: bool = True, a_ref_tag: str = ""):
    """
    label: 0~15
    output: sample_{tag}_pd{class}.csv
    Sampling-to-pd mapping always uses the A that belongs to the sampled model:
      - full -> A_full.npy
      - noise -> A_noise.npy
      - noopt -> A_noopt.npy
      - gan -> A_gan.npy
    """
    ablation_norm = normalize_ablation(ablation)
    tag = ablation_to_tag(ablation_norm)

    device = get_device(cfg)
    ensure_dir(cfg.out_dir)

    ckpt_path = os.path.join(cfg.out_dir, ckpt_filename(tag))
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = load_checkpoint(ckpt_path, map_location=str(device))
    scaler = StandardScaler.from_json(ckpt["scaler"])

    # Keep each model's own A during sampling. For this experiment, cross-tag A mapping
    # makes the comparison hard to interpret and can hide the benefit of trained A_full.
    use_tag = tag
    A_path = os.path.join(cfg.out_dir, a_filename(use_tag))
    if not os.path.exists(A_path):
        raise FileNotFoundError(f"Missing A matrix: {A_path}")
    A = np.load(A_path).astype(np.float32)
    if A.shape != (30, 18):
        raise ValueError(f"A has shape {A.shape}, expected (30,18)")
    A_t = torch.tensor(A, dtype=torch.float32, device=device)

    y = torch.full((n,), int(label), device=device, dtype=torch.long)

    model_type = ckpt.get("model_type", "diffusion")
    if tag == "gan" and model_type != "gan":
        raise ValueError(f"Checkpoint {ckpt_path} is not a GAN checkpoint (model_type={model_type}).")

    if model_type == "gan":
        z_dim = int(ckpt.get("gan_z_dim", getattr(cfg, "gan_z_dim", 32)))
        g_hidden = int(ckpt.get("gan_g_hidden", getattr(cfg, "gan_g_hidden", 128)))
        G = ConditionalGenerator(
            z_dim=z_dim,
            x_dim=cfg.x_dim,
            num_classes=cfg.num_classes,
            hidden_dim=g_hidden,
        ).to(device)
        G.load_state_dict(ckpt["G"], strict=True)
        G.eval()
        with torch.no_grad():
            z = torch.randn((n, z_dim), device=device)
            x_norm = G(z, y)
    else:
        # diffusion path (unchanged)
        net = TransformerDenoiser(
            x_dim=cfg.x_dim,
            num_classes=cfg.num_classes,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_layers,
            dropout=cfg.dropout,
        ).to(device)
        ddpm = DDPM(net, cfg.timesteps, cfg.beta_start, cfg.beta_end).to(device)
        ddpm.load_state_dict(ckpt["model"], strict=True)

        # apply EMA if available
        if use_ema and "ema" in ckpt:
            shadow = ckpt["ema"]
            with torch.no_grad():
                for n_, p in ddpm.named_parameters():
                    if p.requires_grad and n_ in shadow:
                        p.copy_(shadow[n_].to(p.device))

        ddpm.eval()
        x_norm = ddpm.sample(n=n, x_dim=cfg.x_dim, y=y, device=device)  # normalized space

    # denorm
    x = torch.tensor(scaler.inverse_transform(x_norm.detach().cpu().numpy()), device=device, dtype=torch.float32)
    pd30 = x @ A_t.T  # (n,30)

    class_num = int(label) + 1  # 1..16
    out_csv = sample_filename(tag, class_num)
    out_path = os.path.join(cfg.out_dir, out_csv)

    import pandas as pd
    df = pd.DataFrame(pd30.detach().cpu().numpy(), columns=[f"pd_bus{i+1}" for i in range(30)])
    df.to_csv(out_path, index=False)

    print(f"[SAMPLE] model={ckpt_path}", flush=True)
    print(f"[SAMPLE] A_used={A_path}", flush=True)
    print(f"[SAMPLE] saved {out_path}", flush=True)


# ---------------------------
# CLI
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", choices=["train", "sample"])
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--out_dir", type=str, default="./output_model")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])

    parser.add_argument("--ablation", type=str, default="full",
                        choices=["full", "no_opt", "noopt", "noise_only", "noise", "gan", "no_recon", "norecon"])

    # train overrides
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lr_A", type=float, default=5e-4)
    parser.add_argument("--lambda_opt", type=float, default=30.0)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--a_seed", type=int, default=123)
    parser.add_argument("--a_init_mode", type=str, default="aligned", choices=["aligned", "random", "biased"])
    parser.add_argument("--a_bias_mix", type=float, default=0.15)
    parser.add_argument("--a_random_alpha", type=float, default=2.0)
    parser.add_argument("--gan_z_dim", type=int, default=32)
    parser.add_argument("--gan_g_hidden", type=int, default=128)
    parser.add_argument("--gan_d_hidden", type=int, default=128)
    parser.add_argument("--gan_lr_d", type=float, default=2e-4)

    # sample args
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--label", type=int, default=0, help="0~15 corresponds to class1~16")
    parser.add_argument("--a_ref_tag", type=str, default="",
                        help="Legacy arg. Sampling now always uses the current model's own A matrix.")

    args = parser.parse_args()

    cfg = TrainConfig()
    cfg.data_dir = args.data_dir
    cfg.out_dir = args.out_dir
    cfg.device = args.device
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.lr = args.lr
    cfg.lr_A = args.lr_A
    cfg.lambda_opt = args.lambda_opt
    cfg.warmup_epochs = args.warmup_epochs
    cfg.a_seed = args.a_seed
    cfg.a_init_mode = args.a_init_mode
    cfg.a_bias_mix = args.a_bias_mix
    cfg.a_random_alpha = args.a_random_alpha
    cfg.gan_z_dim = args.gan_z_dim
    cfg.gan_g_hidden = args.gan_g_hidden
    cfg.gan_d_hidden = args.gan_d_hidden
    cfg.gan_lr_d = args.gan_lr_d

    if args.mode == "train":
        ab_norm = normalize_ablation(args.ablation)
        if ab_norm == "gan":
            train_gan(cfg, ablation=args.ablation)
        else:
            train(cfg, ablation=args.ablation)
    else:
        sample(cfg, n=args.n, label=args.label, ablation=args.ablation, a_ref_tag=args.a_ref_tag)


if __name__ == "__main__":
    main()
