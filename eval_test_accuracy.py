
import argparse
import json
import os
import sys
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Import VGG6 from the project
try:
    from vgg6_cifar.models.vgg6 import VGG6
except Exception as e:
    print("ERROR: Failed to import VGG6. Make sure you're running from the repo root and the package is on PYTHONPATH.", file=sys.stderr)
    raise

DEFAULT_MEAN = [0.4914, 0.4822, 0.4465]
DEFAULT_STD  = [0.2023, 0.1994, 0.2010]

def load_norm_from_run_dir(run_dir: str):
    """
    Tries to load normalization (mean/std) if it was saved during training.
    Falls back to CIFAR-10 defaults.
    """
    meta_candidates = [
        os.path.join(run_dir, "final_test_metrics.json"),
        os.path.join(run_dir, "train_meta.json"),
        os.path.join(run_dir, "meta.json"),
    ]
    for p in meta_candidates:
        if os.path.exists(p):
            try:
                with open(p, "r") as f:
                    meta = json.load(f)
                mean = meta.get("norm_mean", None)
                std = meta.get("norm_std", None)
                if mean and std and len(mean) == 3 and len(std) == 3:
                    return mean, std
            except Exception:
                pass
    return DEFAULT_MEAN, DEFAULT_STD

def build_transform(mean, std):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

def evaluate(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    return 100.0 * correct / total

def main():
    parser = argparse.ArgumentParser(description="Evaluate VGG6 on CIFAR-10 test set.")
    parser.add_argument("--data_dir", type=str, default="./data", help="Path to CIFAR-10 data dir")
    parser.add_argument("--checkpoint", type=str, default="./runs/final_best/best.pt", help="Path to checkpoint (.pt)")
    parser.add_argument("--batch_size", type=int, default=128, help="Eval batch size")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="cuda or cpu")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--run_dir", type=str, default="./runs/final_best", help="Run directory (to read norm params if available)")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Build model
    model = VGG6(num_classes=10)

    # Load checkpoint
    map_location = torch.device("cpu") if device.type == "cpu" else None
    ckpt = torch.load(args.checkpoint, map_location=map_location)
    state = ckpt.get("model", ckpt)  # support raw state_dict or dict with 'model'
    if isinstance(state, dict):
        model.load_state_dict(state)
    else:
        print("ERROR: Checkpoint format not recognized (expected a dict or state_dict).", file=sys.stderr)
        sys.exit(1)
    model.to(device)

    # Transforms (try from run dir else CIFAR defaults)
    mean, std = load_norm_from_run_dir(args.run_dir)
    transform = build_transform(mean, std)

    # Dataset & loader
    test_set = datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type=="cuda"))

    # Evaluate
    acc = evaluate(model, test_loader, device)
    print(f"Test Accuracy: {acc:.2f}%")

if __name__ == "__main__":
    main()
