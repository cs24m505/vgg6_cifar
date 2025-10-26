
#!/usr/bin/env python3
from __future__ import annotations
import os, itertools, subprocess, argparse, json, csv

def build_parser():
    p = argparse.ArgumentParser("Run grid sweeps (acts, BN, bs, epochs, lr, momentum, optimizer)")
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--base_out", type=str, default="./runs/sweeps")
    p.add_argument("--epochs_list", type=str, default="20,40,60,80,100")
    p.add_argument("--batch_sizes", type=str, default="32,64,128,256,512")
    p.add_argument("--lrs", type=str, default="0.2,0.1,0.05,0.01,0.001")
    p.add_argument("--momentums", type=str, default="0.0,0.9")
    p.add_argument("--optimizers", type=str, default="sgd,nesterov-sgd,adam,adamw,rmsprop,nadam,adagrad")
    p.add_argument("--activations", type=str, default="relu,silu,gelu,tanh,sigmoid")
    p.add_argument("--batch_norms", type=str, default="true,false")
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--label_smoothing", type=float, default=0.0)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="vgg6-cifar10-assignment")
    p.add_argument("--wandb_group", type=str, default="q2q5-sweeps")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0, help="limit runs for quick tests (0=all)")
    return p

def main():
    args = build_parser().parse_args()
    os.makedirs(args.base_out, exist_ok=True)
    epochs_list = [int(x) for x in args.epochs_list.split(",") if x]
    batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x]
    lrs = [float(x) for x in args.lrs.split(",") if x]
    momentums = [float(x) for x in args.momentums.split(",") if x]
    optimizers = [x.strip() for x in args.optimizers.split(",") if x]
    activations = [x.strip() for x in args.activations.split(",") if x]
    bns = [x.strip().lower() in ('true','1','yes','y') for x in args.batch_norms.split(",") if x]
    grid = list(itertools.product(activations, bns, batch_sizes, epochs_list, lrs, momentums, optimizers))
    rows = []
    for i, (act, bn, bs, eph, lr, mom, opt) in enumerate(grid, 1):
        if args.limit and i > args.limit: break
        out_dir = os.path.join(args.base_out, f"act{act}_bn{bn}_bs{bs}_e{eph}_lr{lr}_m{mom}_{opt}")
        cmd = f"""
            python -m vgg6_cifar.scripts.train_experiment \
                --data_dir {args.data_dir} \
                --out_dir {out_dir} \
                --activation {act} \
                --batch_size {bs} \
                --epochs {eph} \
                --lr {lr} \
                --momentum {mom} \
                --optimizer {opt} \
                --weight_decay {args.weight_decay} \
                --label_smoothing {args.label_smoothing} \
                --seed {args.seed} \
                --aug_hflip \
                --aug_crop \
                --aug_cutout \
                --aug_jitter \
                {"--use_bn" if bn else "--no_bn"} \
                {"--amp" if args.amp else ""} \
                {"--wandb --wandb_project " + args.wandb_project + " --wandb_group " + args.wandb_group + " --run_name i{:04d}".format(i) if args.wandb else ""}
            """
        print(">>>", cmd)
        subprocess.run(cmd, shell=True, check=True)
        mpath = os.path.join(out_dir, "final_test_metrics.json")
        if os.path.exists(mpath):
            with open(mpath) as f: m = json.load(f)
            rows.append({"out_dir": out_dir, "activation": act, "use_bn": bn, "batch_size": bs, "epochs": eph,
                         "lr": lr, "momentum": mom, "optimizer": opt, "best_val_acc": m.get("best_val_acc"),
                         "test_top1_acc": m.get("test_top1_acc")})
    sum_path = os.path.join(args.base_out, "sweep_summary.csv")
    with open(sum_path, "w", newline="") as f:
        import csv
        w = csv.DictWriter(f, fieldnames=["out_dir","activation","use_bn","batch_size","epochs","lr","momentum","optimizer","best_val_acc","test_top1_acc"])
        w.writeheader(); [w.writerow(r) for r in rows]
    print("Wrote sweep summary:", sum_path)

if __name__ == "__main__":
    main()
