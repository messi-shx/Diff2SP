# diff2sp_real

This directory contains the code for the real-world case study in Section 5.2 of `diff2sp.pdf`.

The task is conditional scenario generation for power systems:
- `noise`: vanilla diffusion baseline
- `noopt`: diffusion without optimization loss
- `full`: optimization-guided Diff2SP
- `gan`: GAN baseline

## Files

- `main.py`: main entry for training and sampling
- `eval_err.py`: OPF-based evaluation for one run
- `repeat_sample_eval.py`: repeated sampling and evaluation
- `test.sh`: train the `full` model
- `sample.sh`: run repeated evaluation for `gan/noise/noopt/full`
- `models/`: diffusion, transformer, and GAN models
- `opf/`: DC-OPF and PHA solver code
- `data/merged_data.csv`, `data/label_one.csv`: real-world data
- `output_model/`: saved checkpoints and mapping matrices

## Environment

Recommended:
- Python 3.8+
- `torch`
- `numpy`
- `pandas`
- `matplotlib`

## What Is Already Included

This submission already includes:
- trained checkpoints in `output_model/model_*.ckpt`
- saved mapping matrices in `output_model/A_*.npy`
- scaler statistics in `output_model/scaler.json`

So you can run evaluation directly without retraining.

## How To Run

Run from this directory:

```bash
cd diff2sp_real
```

### 1. Train `full`

`test.sh` trains the optimization-guided `full` model:

```bash
bash test.sh
```

Equivalent command:

```bash
python -u main.py \
  --mode train \
  --data_dir ./data \
  --out_dir ./output_model \
  --ablation full \
  --epochs 20 \
  --batch_size 128 \
  --device cuda \
  --lambda_opt 100 \
  --lr_A 2e-3 \
  --warmup_epochs 1 \
  --a_seed 123 \
  --a_init_mode aligned
```

Outputs:
- `output_model/model_full.ckpt`
- `output_model/A_full.npy`
- `output_model/A_full_init.npy`

### 2. Run repeated evaluation

`sample.sh` uses the saved checkpoints to repeatedly sample and evaluate all four methods:

```bash
bash sample.sh
```

Equivalent command:

```bash
python -u repeat_sample_eval.py \
  --out_dir ./output_model \
  --data_dir ./data \
  --device cuda \
  --tags gan noise noopt full \
  --n 50 \
  --n_real 500 \
  --n_gen 100 \
  --a_tag full \
  --repeats 5 \
  --save_prefix repeat_eval_gan
```

This script:
- generates samples for each class and each tag
- evaluates them with `eval_err.py`
- reports mean and std over repeated runs

## Main Outputs

Model files:
- `output_model/model_full.ckpt`
- `output_model/model_gan.ckpt`
- `output_model/model_noise.ckpt`
- `output_model/model_noopt.ckpt`

Mapping files:
- `output_model/A_full.npy`
- `output_model/A_full_init.npy`
- `output_model/A_gan.npy`
- `output_model/A_noise.npy`
- `output_model/A_noopt.npy`

Evaluation output printed to terminal:
- `[FINAL] score ...`
- `[FINAL_REPEAT] overall score ...`
- `[FINAL_REPEAT] class XX ...`

## Notes

- `sample.sh` assumes the checkpoint files in `output_model/` already exist.
- If you want to retrain `noise`, `noopt`, or `gan`, run `main.py` manually with `--ablation noise`, `--ablation noopt`, or `--ablation gan`.
- `eval_err.py` evaluates generated 30-bus demand scenarios through DC-OPF.
