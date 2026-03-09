# models/diffusion.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from opf.pha import pha_solve_pg


class DDPM(nn.Module):
    def __init__(self, model: nn.Module, timesteps: int, beta_start: float, beta_end: float):
        super().__init__()
        self.model = model
        self.timesteps = timesteps

        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphabar = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphabar", alphabar)
        self.register_buffer("sqrt_alphabar", torch.sqrt(alphabar))
        self.register_buffer("sqrt_one_minus_alphabar", torch.sqrt(1.0 - alphabar))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor):
        a = self.sqrt_alphabar[t].unsqueeze(-1)
        b = self.sqrt_one_minus_alphabar[t].unsqueeze(-1)
        return a * x0 + b * noise

    def predict_eps(self, x_t: torch.Tensor, t: torch.Tensor, y: torch.Tensor):
        return self.model(x_t, t, y)

    def predict_x0(self, x_t: torch.Tensor, t: torch.Tensor, eps_hat: torch.Tensor):
        a = self.sqrt_alphabar[t].unsqueeze(-1)
        b = self.sqrt_one_minus_alphabar[t].unsqueeze(-1)
        return (x_t - b * eps_hat) / (a + 1e-8)

    @torch.no_grad()
    def p_sample(self, x_t: torch.Tensor, t: int, y: torch.Tensor):
        betat = self.betas[t]
        alphat = self.alphas[t]
        ab = self.alphabar[t]

        t_batch = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=torch.long)
        eps_hat = self.predict_eps(x_t, t_batch, y)

        coef1 = 1.0 / torch.sqrt(alphat)
        coef2 = betat / (torch.sqrt(1.0 - ab) + 1e-8)
        mean = coef1 * (x_t - coef2 * eps_hat)

        if t == 0:
            return mean
        noise = torch.randn_like(x_t)
        sigma = torch.sqrt(betat)
        return mean + sigma * noise

    @torch.no_grad()
    def sample(self, n: int, x_dim: int, y: torch.Tensor, device: torch.device):
        x = torch.randn((n, x_dim), device=device)
        for t in reversed(range(self.timesteps)):
            x = self.p_sample(x, t, y)
        return x


def _avg_obj_fixed_pg(opf_solver, pd_mw: torch.Tensor, pg_bar: torch.Tensor, chunk: int = 256):
    """
    average Q(pg_bar; scenarios) with pg fixed (2nd-stage recourse solved).
    """
    device = pd_mw.device
    dtype = opf_solver.dtype
    S = pd_mw.shape[0]
    ng = opf_solver.ng

    total = 0.0
    count = 0
    for st in range(0, S, chunk):
        ed = min(S, st + chunk)
        pd_chunk = pd_mw[st:ed]
        pg_fix = pg_bar.to(dtype).unsqueeze(0).expand(ed - st, ng).contiguous()
        _, obj = opf_solver.solve(pd_chunk, gen_fix=pg_fix)
        total += obj.sum().item()
        count += (ed - st)
    return total / max(count, 1)


def diffusion_losses(
    ddpm: DDPM,
    x0: torch.Tensor,
    y: torch.Tensor,
    scaler_inv_fn,
    A_30x18: torch.Tensor,
    opf_solver,
    lambda_noise: float,
    lambda_recon: float,
    lambda_opt: float,
    opt_t_max: int,
    # ---- new knobs ----
    opt_mode: str = "optimal_gap",     # "optimal_gap" or "fixed_pg"
    opt_scenarios: int = 32,
    opt_min_scenarios: int = 4,
    pha_rho: float = 1.0,
    pha_max_iter: int = 10,
    pha_tol: float = 1e-2,
    qp_chunk: int = 256,
):
    """
    Always compute and return metrics: L_noise/L_recon/L_opt.
    If lambda_opt==0 -> compute L_opt under no_grad (metric only).
    opt_mode:
      - "optimal_gap": compare Q*(gen) vs Q*(real) (your old scheme2)
      - "fixed_pg": align with formula(20): Q(pg*;gen) vs Q(pg*;real), pg* from real PHA on a subbatch.
    """
    B = x0.shape[0]
    device = x0.device
    T = ddpm.timesteps
    eps = 1e-6

    # -------------------------
    # (1) noise + recon with random t
    # -------------------------
    t = torch.randint(0, T, (B,), device=device, dtype=torch.long)
    noise = torch.randn_like(x0)
    x_t = ddpm.q_sample(x0, t, noise)
    eps_hat = ddpm.predict_eps(x_t, t, y)

    L_noise = F.mse_loss(eps_hat, noise)

    x0_hat = ddpm.predict_x0(x_t, t, eps_hat)
    L_recon = F.mse_loss(x0_hat, x0)

    # -------------------------
    # (2) build a cleaner x0_hat for OPF using small t in [0,opt_t_max]
    # -------------------------
    opt_t_max = max(int(opt_t_max), 0)
    if opt_t_max == 0:
        t_opt = torch.zeros((B,), device=device, dtype=torch.long)
    else:
        t_opt = torch.randint(0, opt_t_max + 1, (B,), device=device, dtype=torch.long)

    noise_opt = torch.randn_like(x0)
    x_t_opt = ddpm.q_sample(x0, t_opt, noise_opt)
    eps_hat_opt = ddpm.predict_eps(x_t_opt, t_opt, y)
    x0_hat_opt = ddpm.predict_x0(x_t_opt, t_opt, eps_hat_opt)

    # denorm + map to pd30
    x_hat_den = scaler_inv_fn(x0_hat_opt)
    x_den = scaler_inv_fn(x0)

    pd_gen = x_hat_den @ A_30x18.T   # (B,30)
    pd_real = x_den @ A_30x18.T      # (B,30)

    # qpth in double
    pd_gen_d = pd_gen.to(opf_solver.dtype)
    pd_real_d = pd_real.to(opf_solver.dtype)

    # -------------------------
    # (3) OPF loss metric (and grad if enabled)
    # -------------------------
    L_opt = torch.zeros((), device=device, dtype=x0.dtype)

    if opt_mode == "optimal_gap":
        # training-like: compare optimal objectives
        with torch.no_grad():
            _, obj_real = opf_solver.solve(pd_real_d)
        if lambda_opt > 0.0:
            _, obj_gen = opf_solver.solve(pd_gen_d)
            L_opt = ((obj_gen - obj_real).abs() / (obj_real.abs() + eps)).mean()
        else:
            with torch.no_grad():
                _, obj_gen = opf_solver.solve(pd_gen_d)
                L_opt = ((obj_gen - obj_real).abs() / (obj_real.abs() + eps)).mean()

    elif opt_mode == "fixed_pg":
        # Align with formula(20):
        # pick the most represented class in the batch so the OPF term is active more often
        uniq = torch.unique(y)
        cls_counts = torch.stack([(y == cls_i).sum() for cls_i in uniq])
        eligible = torch.where(cls_counts >= max(int(opt_min_scenarios), 2))[0]

        if eligible.numel() > 0:
            best_local = eligible[torch.argmax(cls_counts[eligible])]
        else:
            best_local = torch.argmax(cls_counts)

        cls = uniq[best_local].item()
        idx = torch.where(y == cls)[0]

        if idx.numel() >= 2:
            # sub-sample scenarios for PHA to reduce cost
            if idx.numel() > opt_scenarios:
                idx = idx[torch.randperm(idx.numel(), device=device)[:opt_scenarios]]

            pd_real_sub = pd_real_d[idx]
            pd_gen_sub = pd_gen_d[idx]

            # pg* from REAL via light PHA (no_grad inside pha_solve_pg)
            pg_bar = pha_solve_pg(
                opf_solver=opf_solver,
                pd_scenarios_mw=pd_real_sub,
                rho=pha_rho,
                max_iter=pha_max_iter,
                tol=pha_tol,
                batch_chunk=min(qp_chunk, pd_real_sub.shape[0]),
            )

            # Q_real_fixed no grad
            with torch.no_grad():
                Q_real_fixed = _avg_obj_fixed_pg(opf_solver, pd_real_sub, pg_bar, chunk=min(qp_chunk, pd_real_sub.shape[0]))

            if lambda_opt > 0.0:
                # allow grad through generated scenarios
                # compute average objective with pg fixed (need tensor objs for grad)
                ng = opf_solver.ng
                pg_fix = pg_bar.to(opf_solver.dtype).unsqueeze(0).expand(pd_gen_sub.shape[0], ng).contiguous()
                _, obj_gen = opf_solver.solve(pd_gen_sub, gen_fix=pg_fix)
                Q_gen_fixed = obj_gen.mean()
                L_opt = (Q_gen_fixed - Q_real_fixed).abs() / (abs(Q_real_fixed) + eps)
            else:
                with torch.no_grad():
                    ng = opf_solver.ng
                    pg_fix = pg_bar.to(opf_solver.dtype).unsqueeze(0).expand(pd_gen_sub.shape[0], ng).contiguous()
                    _, obj_gen = opf_solver.solve(pd_gen_sub, gen_fix=pg_fix)
                    Q_gen_fixed = obj_gen.mean().item()
                    L_opt = torch.tensor(abs(Q_gen_fixed - Q_real_fixed) / (abs(Q_real_fixed) + eps),
                                         device=device, dtype=x0.dtype)

    else:
        raise ValueError(f"Unknown opt_mode: {opt_mode}")

    total = lambda_noise * L_noise + lambda_recon * L_recon + lambda_opt * L_opt

    return {
        "total": total,
        "L_noise": L_noise.detach(),
        "L_recon": L_recon.detach(),
        "L_opt": L_opt.detach(),
    }
