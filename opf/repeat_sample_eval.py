import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def run_cmd(cmd, cwd: Path, capture: bool = False):
    env = os.environ.copy()
    # Avoid OpenMP shared-memory permission issues on some clusters.
    env.setdefault("KMP_USE_SHM", "0")
    env.setdefault("OMP_NUM_THREADS", "1")

    if capture:
        # Stream stdout/stderr in real time while still collecting it for parsing.
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        collected = []
        assert proc.stdout is not None
        for line in proc.stdout:
            collected.append(line)
            print(line, end="", flush=True)
        ret = proc.wait()
        if ret != 0:
            raise subprocess.CalledProcessError(ret, cmd, output="".join(collected))
        return "".join(collected)
    subprocess.run(cmd, cwd=str(cwd), check=True, env=env)
    return ""


def parse_final_score(stdout_text: str, tags):
    # expected line: [FINAL] score noise=... noopt=... full=...
    lines = stdout_text.splitlines()
    final_lines = [ln for ln in lines if ln.startswith("[FINAL] score ")]
    if not final_lines:
        raise RuntimeError("Cannot find '[FINAL] score ...' in eval output.")
    line = final_lines[-1]

    found = dict(re.findall(r"([A-Za-z0-9_]+)=([-+0-9.eE]+)", line))
    out = {}
    for t in tags:
        if t not in found:
            raise RuntimeError(f"Tag '{t}' missing in final line: {line}")
        out[t] = float(found[t])
    return out


def parse_class_scores(stdout_text: str, tags):
    """
    Parse lines like:
      [OK] class 1: score noise=... noopt=... full=... | ...
    """
    class_scores = {}
    pat = re.compile(r"^\[OK\]\s+class\s+(\d+):\s+score\s+(.+?)(?:\s+\||$)")

    for line in stdout_text.splitlines():
        m = pat.match(line.strip())
        if not m:
            continue
        cls = int(m.group(1))
        score_part = m.group(2)
        found = dict(re.findall(r"([A-Za-z0-9_]+)=([-+0-9.eE]+)", score_part))
        if all(t in found for t in tags):
            class_scores[cls] = {t: float(found[t]) for t in tags}

    return class_scores


def plot_final_repeat(class_stats, tags, out_dir: Path, repeats: int):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[WARN] matplotlib not available, skip plotting: {e}", flush=True)
        return

    classes = sorted(class_stats.keys())
    if not classes:
        print("[WARN] No class stats found, skip plotting.", flush=True)
        return

    x = np.arange(len(classes), dtype=np.float64)
    width = 0.8 / max(len(tags), 1)

    fig, ax = plt.subplots(figsize=(13, 5))
    for i, tag in enumerate(tags):
        means = []
        stds = []
        for cls in classes:
            arr = np.array(class_stats[cls][tag], dtype=np.float64)
            if arr.size == 0:
                means.append(np.nan)
                stds.append(np.nan)
            else:
                means.append(float(arr.mean()))
                stds.append(float(arr.std(ddof=0)))
        means = np.array(means, dtype=np.float64)
        stds = np.nan_to_num(np.array(stds, dtype=np.float64), nan=0.0)
        ax.bar(x + (i - (len(tags) - 1) / 2.0) * width, means, width, yerr=stds, capsize=2, label=tag)

    ax.set_xticks(x)
    ax.set_xticklabels([f"class {c}" for c in classes], rotation=30, ha="right")
    ax.set_ylabel("Score (mean ± std)")
    ax.set_title(f"Final Repeat Class-wise Score (repeats={repeats})")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()

    png_path = out_dir / "final_repeat_class_score.png"
    pdf_path = out_dir / "final_repeat_class_score.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVE] {png_path}", flush=True)
    print(f"[SAVE] {pdf_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--out_dir", type=str, default="./output_model")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--tags", nargs="+", default=["noise", "noopt", "full"])
    parser.add_argument("--num_classes", type=int, default=16)
    parser.add_argument("--n", type=int, default=50, help="samples per class per tag")
    parser.add_argument("--n_real", type=int, default=500)
    parser.add_argument("--n_gen", type=int, default=50)
    parser.add_argument("--a_tag", type=str, default="full")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--save_prefix", type=str, default="repeat_eval", help="Deprecated, kept for compatibility.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    scores_by_tag = {t: [] for t in args.tags}
    class_scores_runs = {
        cls: {t: [] for t in args.tags}
        for cls in range(1, args.num_classes + 1)
    }
    for run_idx in range(1, args.repeats + 1):
        print(f"\n[INFO] ===== Repeat {run_idx}/{args.repeats} =====", flush=True)

        # 1) sample
        for tag in args.tags:
            for label in range(args.num_classes):
                cmd = [
                    sys.executable,
                    "main.py",
                    "--mode",
                    "sample",
                    "--ablation",
                    tag,
                    "--out_dir",
                    str(args.out_dir),
                    "--label",
                    str(label),
                    "--n",
                    str(args.n),
                    "--device",
                    args.device,
                ]
                run_cmd(cmd, root, capture=False)

        # 2) eval
        cmd_eval = [
            sys.executable,
            "eval_err.py",
            "--data_dir",
            str(args.data_dir),
            "--out_dir",
            str(args.out_dir),
            "--tags",
            *args.tags,
            "--n_real",
            str(args.n_real),
            "--n_gen",
            str(args.n_gen),
            "--device",
            args.device,
            "--a_tag",
            str(args.a_tag),
        ]
        eval_stdout = run_cmd(cmd_eval, root, capture=True)
        final_scores = parse_final_score(eval_stdout, args.tags)
        per_class_scores = parse_class_scores(eval_stdout, args.tags)

        print("[FINAL_RUN] score " + " ".join([f"{t}={final_scores[t]:.4f}" for t in args.tags]), flush=True)
        for t in args.tags:
            scores_by_tag[t].append(final_scores[t])
        for cls in range(1, args.num_classes + 1):
            if cls in per_class_scores:
                for t in args.tags:
                    class_scores_runs[cls][t].append(per_class_scores[cls][t])

    # final mean±std of score across repeats
    parts = []
    overall_rows = []
    for t in args.tags:
        arr = np.array(scores_by_tag[t], dtype=np.float64)
        mean = float(arr.mean())
        std = float(arr.std(ddof=0))
        parts.append(f"{t}={arr.mean():.4f}±{arr.std(ddof=0):.4f}")
        overall_rows.append({"tag": t, "score_mean": mean, "score_std": std, "repeats": int(arr.size)})
    print("\n[FINAL_REPEAT] overall score " + " ".join(parts), flush=True)

    print("[FINAL_REPEAT] class-wise score mean±std", flush=True)
    class_rows = []
    for cls in range(1, args.num_classes + 1):
        class_parts = []
        for t in args.tags:
            arr = np.array(class_scores_runs[cls][t], dtype=np.float64)
            if arr.size == 0:
                class_parts.append(f"{t}=nan±nan")
                class_rows.append({"class": cls, "tag": t, "score_mean": np.nan, "score_std": np.nan, "repeats": 0})
            else:
                mean = float(arr.mean())
                std = float(arr.std(ddof=0))
                class_parts.append(f"{t}={mean:.4f}±{std:.4f}")
                class_rows.append({"class": cls, "tag": t, "score_mean": mean, "score_std": std, "repeats": int(arr.size)})
        print(f"[FINAL_REPEAT] class {cls:02d} " + " ".join(class_parts), flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    overall_csv = out_dir / "final_repeat_overall_score.csv"
    class_csv = out_dir / "final_repeat_class_score.csv"
    pd.DataFrame(overall_rows).to_csv(overall_csv, index=False)
    pd.DataFrame(class_rows).to_csv(class_csv, index=False)
    print(f"[SAVE] {overall_csv}", flush=True)
    print(f"[SAVE] {class_csv}", flush=True)
    plot_final_repeat(class_scores_runs, args.tags, out_dir=out_dir, repeats=args.repeats)


if __name__ == "__main__":
    main()
