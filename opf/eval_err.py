# eval_err.py
import os
import argparse
import numpy as np
import pandas as pd
import torch

from data import load_csv_data
from opf.dc_opf_qpth import SoftDCOPF_QPTH
from opf.pha import pha_solve_pg


def avg_obj_fixed_pg(opf: SoftDCOPF_QPTH, pd_mw: torch.Tensor, pg_bar: torch.Tensor, chunk: int = 256):
    """
    Q(pg_bar; scenarios) = average objective across scenarios with pg fixed.
    """
    device = pd_mw.device
    dtype = opf.dtype
    S = pd_mw.shape[0]
    ng = opf.ng

    total = 0.0
    count = 0

    for st in range(0, S, chunk):
        ed = min(S, st + chunk)
        pd_chunk = pd_mw[st:ed]
        pg_fix = pg_bar.to(dtype).unsqueeze(0).expand(ed - st, ng).contiguous()
        _, obj = opf.solve(pd_chunk, gen_fix=pg_fix)
        total += obj.sum().item()
        count += (ed - st)

    return total / max(count, 1)


def avg_obj_optimal_and_slacks(opf: SoftDCOPF_QPTH, pd_mw: torch.Tensor, chunk: int = 256):
    """
    Q*(scenarios) = average optimal objective across scenarios.
    Also returns avg shedding / curtailment / line-slack and line-slack active rate.
    """
    S = pd_mw.shape[0]
    dtype = opf.dtype

    total_obj = 0.0
    total_s = 0.0
    total_r = 0.0
    total_line_slack = 0.0
    total_line_slack_active = 0.0
    count = 0

    i_s = opf._idx["i_s"]
    i_r = opf._idx["i_r"]
    i_up = opf._idx["i_up"]
    i_un = opf._idx["i_un"]
    nb = opf.nb
    nl = opf.nl

    for st in range(0, S, chunk):
        ed = min(S, st + chunk)
        pd_chunk = pd_mw[st:ed]
        x, obj = opf.solve(pd_chunk)

        s = x[:, i_s:i_s + nb]
        r = x[:, i_r:i_r + nb]
        up = x[:, i_up:i_up + nl]
        un = x[:, i_un:i_un + nl]
        line_slack = up + un

        total_obj += obj.sum().item()
        total_s += s.sum().item()
        total_r += r.sum().item()
        total_line_slack += line_slack.sum().item()
        total_line_slack_active += (line_slack.sum(dim=1) > 1e-8).to(torch.float64).sum().item()
        count += (ed - st)

    avg_obj = total_obj / max(count, 1)
    avg_shed = total_s / max(count, 1)  # sum over buses per scenario
    avg_curt = total_r / max(count, 1)
    avg_line_slack = total_line_slack / max(count, 1)
    line_slack_active_rate = total_line_slack_active / max(count, 1)
    return avg_obj, avg_shed, avg_curt, avg_line_slack, line_slack_active_rate


def pd_distribution_metrics(pd_real: np.ndarray, pd_gen: np.ndarray, eps: float = 1e-6):
    """
    Compare generated vs real 30-bus load distributions with multiple shape-sensitive metrics.
    """
    mu_real = pd_real.mean(axis=0)
    mu_gen = pd_gen.mean(axis=0)

    std_real = pd_real.std(axis=0)
    std_gen = pd_gen.std(axis=0)

    mu_rel_l2 = float(np.linalg.norm(mu_gen - mu_real) / (np.linalg.norm(mu_real) + eps))
    std_rel_l2 = float(np.linalg.norm(std_gen - std_real) / (np.linalg.norm(std_real) + eps))

    # Mean total load mismatch (captures scale mismatch directly)
    real_total = pd_real.sum(axis=1).mean()
    gen_total = pd_gen.sum(axis=1).mean()
    total_rel = float(abs(gen_total - real_total) / (abs(real_total) + eps))

    # Mean profile direction mismatch (captures spatial pattern mismatch)
    mu_cos = float(np.dot(mu_gen, mu_real) / (np.linalg.norm(mu_gen) * np.linalg.norm(mu_real) + eps))

    # Covariance shape mismatch
    if pd_real.shape[0] >= 2 and pd_gen.shape[0] >= 2:
        cov_real = np.cov(pd_real, rowvar=False)
        cov_gen = np.cov(pd_gen, rowvar=False)
        cov_rel_fro = float(np.linalg.norm(cov_gen - cov_real) / (np.linalg.norm(cov_real) + eps))
    else:
        cov_rel_fro = np.nan

    return {
        "mu_rel_l2": mu_rel_l2,
        "std_rel_l2": std_rel_l2,
        "total_rel": total_rel,
        "mu_cos": mu_cos,
        "cov_rel_fro": cov_rel_fro,
    }


def load_sample_pd(out_dir: str, tag: str, class_num: int):
    fname = f"sample_{tag}_pd{class_num}.csv"
    path = os.path.join(out_dir, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing sample file: {path}")
    df = pd.read_csv(path)
    pd30 = df.values.astype(np.float32)
    if pd30.shape[1] != 30:
        raise ValueError(f"{fname} has shape {pd30.shape}, expected (*,30)")
    return pd30


def load_A(out_dir: str, a_tag: str = ""):
    """
    If a_tag is set, use A_{a_tag}.npy.
    Otherwise keep backward compatibility with legacy A.npy.
    """
    if a_tag and len(a_tag.strip()) > 0:
        name = f"A_{a_tag.strip().lower()}.npy"
    else:
        name = "A.npy"

    path = os.path.join(out_dir, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing A matrix: {path}")

    A = np.load(path).astype(np.float32)
    if A.shape != (30, 18):
        raise ValueError(f"{name} has shape {A.shape}, expected (30,18)")
    return A, path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--out_dir", type=str, default="./output_model")
    parser.add_argument("--n_rows", type=int, default=105120)

    parser.add_argument("--tags", nargs="+", default=["noise", "noopt", "full"])

    parser.add_argument("--n_real", type=int, default=800)  # 0 = all
    parser.add_argument("--n_gen", type=int, default=800)   # 0 = all
    parser.add_argument("--a_tag", type=str, default="",
                        help="If set (e.g. full), use A_<tag>.npy for both real-data mapping and OPF evaluation.")

    # PHA
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--pha_max_iter", type=int, default=50)
    parser.add_argument("--pha_tol", type=float, default=1e-3)

    parser.add_argument("--chunk", type=int, default=256)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--no_print_per_class", action="store_true",
                        help="Disable per-class printing (default prints per-class details).")
    args = parser.parse_args()

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    # load A
    A, A_path = load_A(args.out_dir, args.a_tag)
    AT = A.T  # (18,30)
    # keep console compact: final score only by default

    # load real 18d + labels
    x18, y = load_csv_data(args.data_dir, "merged_data.csv", "label_one.csv", args.n_rows)

    # OPF solver
    opf = SoftDCOPF_QPTH(device=str(device), dtype=torch.float64)

    rng = np.random.default_rng(123)
    eps = 1e-6

    rows = []
    for class_num in range(1, 17):
        c = class_num - 1
        idx = np.where(y == c)[0]
        if len(idx) == 0:
            if not args.no_print_per_class:
                print(f"[WARN] class {class_num}: no real samples found, skip", flush=True)
            continue

        # select real scenarios
        if args.n_real > 0:
            m = min(args.n_real, len(idx))
            sel = rng.choice(idx, size=m, replace=False)
        else:
            sel = idx

        x_real = x18[sel]  # (S,18)
        pd_real = (x_real @ AT).astype(np.float32)  # (S,30)
        pd_real_t = torch.tensor(pd_real, device=device, dtype=torch.float64)

        # --- pg* from real SAA (PHA) ---
        pg_bar = pha_solve_pg(
            opf_solver=opf,
            pd_scenarios_mw=pd_real_t,
            rho=args.rho,
            max_iter=args.pha_max_iter,
            tol=args.pha_tol,
            batch_chunk=args.chunk,
        )

        # --- Formula (20) baseline: fixed pg* ---
        Q_real_fixed = avg_obj_fixed_pg(opf, pd_real_t, pg_bar, chunk=args.chunk)

        # --- Optimal baseline: Q*(real) ---
        Q_real_opt, shed_real, curt_real, line_slack_real, line_slack_rate_real = avg_obj_optimal_and_slacks(
            opf, pd_real_t, chunk=args.chunk
        )

        row = {
            "class": class_num,
            "Q_real_fixed": Q_real_fixed,
            "Q_real_opt": Q_real_opt,
            "shed_real": shed_real,
            "curt_real": curt_real,
            "line_slack_real": line_slack_real,
            "line_slack_rate_real": line_slack_rate_real,
        }

        # evaluate generated tags
        for tag in args.tags:
            pd_gen = load_sample_pd(args.out_dir, tag, class_num)
            if args.n_gen > 0:
                pd_gen = pd_gen[: min(args.n_gen, pd_gen.shape[0])]
            pd_gen_t = torch.tensor(pd_gen, device=device, dtype=torch.float64)

            # fixed-pg objective (formula 20 style)
            Q_gen_fixed = avg_obj_fixed_pg(opf, pd_gen_t, pg_bar, chunk=args.chunk)

            # optimal objective
            Q_gen_opt, shed_gen, curt_gen, line_slack_gen, line_slack_rate_gen = avg_obj_optimal_and_slacks(
                opf, pd_gen_t, chunk=args.chunk
            )

            # Objective error (symmetric relative form)
            err_obj = abs(Q_gen_fixed - Q_real_fixed) / (abs(Q_gen_fixed) + abs(Q_real_fixed) + eps)

            # distribution-level metrics on raw pd trajectories
            pd_metrics = pd_distribution_metrics(pd_real, pd_gen, eps=eps)

            # component-wise OPF mismatches
            err_shed = abs(shed_gen - shed_real) / (abs(shed_real) + eps)
            err_curt = abs(curt_gen - curt_real) / (abs(curt_real) + eps)
            err_line_slack = abs(line_slack_gen - line_slack_real) / (abs(line_slack_real) + eps)
            err_line_slack_rate = abs(line_slack_rate_gen - line_slack_rate_real) / (abs(line_slack_rate_real) + eps)

            row[f"Q_{tag}_fixed"] = Q_gen_fixed
            row[f"Q_{tag}_opt"] = Q_gen_opt
            row[f"err_obj_{tag}"] = err_obj

            row[f"shed_{tag}"] = shed_gen
            row[f"curt_{tag}"] = curt_gen
            row[f"line_slack_{tag}"] = line_slack_gen
            row[f"line_slack_rate_{tag}"] = line_slack_rate_gen

            row[f"err_shed_{tag}"] = err_shed
            row[f"err_curt_{tag}"] = err_curt
            row[f"err_line_slack_{tag}"] = err_line_slack
            row[f"err_line_slack_rate_{tag}"] = err_line_slack_rate

            row[f"pd_mu_rel_{tag}"] = pd_metrics["mu_rel_l2"]
            row[f"pd_std_rel_{tag}"] = pd_metrics["std_rel_l2"]
            row[f"pd_total_rel_{tag}"] = pd_metrics["total_rel"]
            row[f"pd_mu_cos_{tag}"] = pd_metrics["mu_cos"]
            row[f"pd_cov_rel_{tag}"] = pd_metrics["cov_rel_fro"]

            # scale-compressed score: keep raw metrics in CSV, but combine on closer scales
            cov_term = 0.0 if np.isnan(pd_metrics["cov_rel_fro"]) else pd_metrics["cov_rel_fro"]
            cov_log = float(np.log1p(max(cov_term, 0.0)))
            line_slack_log = float(np.log1p(max(err_line_slack, 0.0)))

            row[f"pd_cov_log_{tag}"] = cov_log
            row[f"line_slack_log_{tag}"] = line_slack_log

            # composite score (smaller is better)
            row[f"score_{tag}"] = (
                err_obj
                + pd_metrics["mu_rel_l2"]
                + cov_log
                + line_slack_log
            ) / 4.0

        rows.append(row)

        if not args.no_print_per_class:
            msg = f"[OK] class {class_num}: score " + " ".join([f"{t}={row[f'score_{t}']:.4f}" for t in args.tags])
            print(msg, flush=True)

    if len(rows) == 0:
        raise RuntimeError("No valid class rows to evaluate.")

    # final score across classes (mean)
    final = {}
    for tag in args.tags:
        vals = [r[f"score_{tag}"] for r in rows]
        final[tag] = float(np.mean(vals))

    print("[FINAL] score " + " ".join([f"{t}={final[t]:.4f}" for t in args.tags]), flush=True)

    df = pd.DataFrame(rows).sort_values("class")
    out_path = os.path.join(args.out_dir, "err_table_with_opt.csv")
    df.to_csv(out_path, index=False)
    print(f"[SAVE] {out_path}", flush=True)


if __name__ == "__main__":
    main()
