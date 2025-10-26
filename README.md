# VGG6 CIFAR-10 Experiment — Command-Line Execution Guide

This README provides full instructions to reproduce **CS6886W – System Engineering for Deep Learning (Assignment 1)** experiments entirely via the **command line** (no Jupyter Notebook required).

---

## 📦 1. Environment Setup

```bash
# Clone repository
git clone <repo_link_here>
cd <repo_name>

# (Optional) create and activate virtual environment
python -m venv vgg_env
source vgg_env/bin/activate        # Linux / macOS
vgg_env\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## 🧩 2. Baseline Training (Q1)

Train the baseline **VGG6** model with **ReLU activation** and **BatchNorm** enabled.

```bash
python -m vgg6_cifar.scripts.train_baseline \
  --data_dir ./data --out_dir ./runs/baseline \
  --epochs 60 --batch_size 128 --lr 0.1 --optimizer sgd --momentum 0.9 \
  --weight_decay 5e-4 --label_smoothing 0.0 \
  --aug_hflip --aug_crop --aug_cutout --aug_jitter --amp --seed 42
```

Generate loss and accuracy plots:
```bash
python -m vgg6_cifar.scripts.plot_curves \
  --metrics_csv ./runs/baseline/metrics.csv \
  --out_dir ./runs/baseline
```

Outputs: `metrics.csv`, `final_test_metrics.json`, `accuracy_curves.png`, `loss_curves.png`.

---

## ⚙️ 3. Activation Function Sweep (Q2a)

Run experiments for multiple activations (ReLU, Sigmoid, Tanh, SiLU, GELU):

```bash
for act in relu silu gelu tanh sigmoid; do
  python -m vgg6_cifar.scripts.train_experiment \
    --data_dir ./data --out_dir ./runs/act_${act} \
    --activation $act --optimizer sgd --lr 0.1 --batch_size 128 --epochs 40 \
    --use_bn --aug_hflip --aug_crop --aug_cutout --aug_jitter --amp --seed 42 --wandb
done
```

Compare validation and test accuracies across activations in Weights & Biases.

---

## 🚀 4. Optimizer Comparison (Q2b)

Test multiple optimizers using ReLU activation:

```bash
python -m vgg6_cifar.scripts.train_experiment --data_dir ./data --out_dir ./runs/opt_sgd      --activation relu --optimizer sgd         --lr 0.1   --batch_size 128 --epochs 40 --use_bn --amp --seed 42 --wandb
python -m vgg6_cifar.scripts.train_experiment --data_dir ./data --out_dir ./runs/opt_nesterov --activation relu --optimizer nesterov-sgd --lr 0.1   --batch_size 128 --epochs 40 --use_bn --amp --seed 42 --wandb
python -m vgg6_cifar.scripts.train_experiment --data_dir ./data --out_dir ./runs/opt_adam     --activation relu --optimizer adam        --lr 0.001 --batch_size 128 --epochs 40 --use_bn --amp --seed 42 --wandb
python -m vgg6_cifar.scripts.train_experiment --data_dir ./data --out_dir ./runs/opt_adamw    --activation relu --optimizer adamw       --lr 0.001 --batch_size 128 --epochs 40 --use_bn --amp --seed 42 --wandb
python -m vgg6_cifar.scripts.train_experiment --data_dir ./data --out_dir ./runs/opt_rmsprop  --activation relu --optimizer rmsprop     --lr 0.01  --batch_size 128 --epochs 40 --use_bn --amp --seed 42 --wandb
python -m vgg6_cifar.scripts.train_experiment --data_dir ./data --out_dir ./runs/opt_nadam    --activation relu --optimizer nadam       --lr 0.001 --batch_size 128 --epochs 40 --use_bn --amp --seed 42 --wandb
python -m vgg6_cifar.scripts.train_experiment --data_dir ./data --out_dir ./runs/opt_adagrad  --activation relu --optimizer adagrad     --lr 0.05  --batch_size 128 --epochs 40 --use_bn --amp --seed 42 --wandb
```

---

## 🧮 5. Hyperparameter Variations (Q2c)

Example 1 — small batch, long training:
```bash
python -m vgg6_cifar.scripts.train_experiment \
  --data_dir ./data --out_dir ./runs/hp_bs64_e80_lr005_m09_bnTrue \
  --activation relu --optimizer sgd --lr 0.05 --batch_size 64 --epochs 80 \
  --momentum 0.9 --use_bn --amp --seed 42 --wandb
```

Example 2 — large batch, short training:
```bash
python -m vgg6_cifar.scripts.train_experiment \
  --data_dir ./data --out_dir ./runs/hp_bs256_e40_lr02_m00_bnFalse \
  --activation relu --optimizer sgd --lr 0.2 --batch_size 256 --epochs 40 \
  --momentum 0.0 --no_bn --amp --seed 42 --wandb
```

Full grid sweep:
```bash
python -m vgg6_cifar.scripts.sweep_grid \
  --data_dir ./data --base_out ./runs/sweeps_ext \
  --epochs_list 20,40,60,80,100 \
  --batch_sizes 32,64,128,256,512 \
  --lrs 0.2,0.1,0.05,0.01,0.001 \
  --momentums 0.0,0.9 \
  --optimizers sgd,nesterov-sgd,adam,adamw,rmsprop,nadam,adagrad \
  --activations relu,silu,gelu,tanh,sigmoid \
  --batch_norms true,false \
  --amp --wandb --seed 42
```

---

## 📊 6. Plot Generation (Q3)

Parallel coordinates and scatter plots are generated automatically via **W&B** or through the following commands:

Validation Accuracy vs. Step:
```bash
python -m vgg6_cifar.scripts.plot_scatter_valacc_vs_step \
  --metrics_csv ./runs/act_gelu/metrics.csv \
  --out_png ./runs/act_gelu/scatter_valacc_vs_step.png
```

Training & Validation Curves:
```bash
python -m vgg6_cifar.scripts.plot_curves \
  --metrics_csv ./runs/act_gelu/metrics.csv \
  --out_dir ./runs/act_gelu
```

---

## 🏆 7. Final Best Configuration (Q4)

Re-run the best model from W&B (e.g., ReLU + Nesterov-SGD + LR 0.1 + BatchNorm):

```bash
python -m vgg6_cifar.scripts.train_experiment \
  --data_dir ./data --out_dir ./runs/final_best \
  --activation relu --optimizer nesterov-sgd \
  --lr 0.1 --batch_size 200 --epochs 80 \
  --momentum 0.9 --use_bn \
  --aug_hflip --aug_crop --aug_cutout --aug_jitter --amp --seed 42 --wandb
```

Artifacts:  
- `runs/final_best/best.pt`  
- `runs/final_best/final_test_metrics.json`

---

## 🔁 8. Reproducibility & Environment (Q5)

- **Python version:** ≥ 3.9  
- **Torch:** ≥ 2.0, **Torchvision:** ≥ 0.15  
- **CUDA:** ≥ 11.8 (if GPU available)  
- **Seed:** use `--seed 42` for reproducibility  
- All output logs & metrics saved under `/runs/<experiment_name>/`  

---


## ✅ 9. Validation-only: Quick Test Accuracy

Use this script to evaluate a saved checkpoint on CIFAR-10 **from the command line**.

1) Ensure the repo is installed/available on PYTHONPATH (run from repo root).  
2) Run:

```bash
python eval_test_accuracy.py   --data_dir ./data   --checkpoint ./runs/final_best/best.pt   --batch_size 128   --device cpu   --run_dir ./runs/final_best
```

**Notes**
- If you trained with custom normalization, the script will try to read it from `./runs/final_best/final_test_metrics.json` (or `train_meta.json` / `meta.json`). If not found, it defaults to CIFAR-10 mean/std.
- Use `--device cpu` if you don’t have a GPU.
- Output prints a single line: `Test Accuracy: XX.XX%`.
