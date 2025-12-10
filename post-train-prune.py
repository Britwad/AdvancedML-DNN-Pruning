import time
import numpy as np
import copy
import torch
import os
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# --- Model Imports ---
from models.VGG import VGG
from models.ResNet32 import ResNet32Masked

from prune import prune
from train import train, evaluate
from Pruners.Pruners import Rand, Mag, SNIP, GraSP, SynFlow


def run_experiment(config):
    """
    Generic function to run the Train -> Prune -> Finetune pipeline
    based on the provided configuration dictionary.
    """

    # ---------------- Setup ----------------
    torch.backends.cudnn.benchmark = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Running Experiment: {config['name']}")

    # ---------------- Data ----------------
    print("Preparing Data...")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(config['norm_mean'], config['norm_std']),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(config['norm_mean'], config['norm_std']),
    ])

    # Dynamic Dataset Loading
    train_dataset = config['dataset_class']("./data", train=True, download=True, transform=transform_train)
    test_dataset = config['dataset_class']("./data", train=False, download=True, transform=transform_test)

    # 90/10 Train/Val Split
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_ds, val_ds = random_split(train_dataset, [train_size, val_size])

    # Optimized Loader Kwargs
    loader_kwargs = {'batch_size': 128, 'num_workers': 2, 'pin_memory': True, 'persistent_workers': True}

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    # Sample for pruning methods
    sample_inputs, sample_targets = next(iter(train_loader))
    sample_dataloader = [(sample_inputs, sample_targets)]

    # ---------------- Phase 1: Pre-train Dense Baseline ----------------
    print(f"\n===== Phase 1: Pre-training Dense Baseline ({config['name']}) =====")

    # Initialize Model using the lambda/function in config
    model = config['model_init']().to(DEVICE)

    # Check for saved dense weights to save time
    if os.path.exists(config['dense_ckpt']):
        print(f"Found '{config['dense_ckpt']}'! Loading weights...")
        model.load_state_dict(torch.load(config['dense_ckpt'], map_location=DEVICE))
    else:
        print(f"'{config['dense_ckpt']}' not found. Training from scratch...")
        optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
        scheduler = MultiStepLR(optimizer, milestones=[80, 120], gamma=0.1)

        t_start = time.time()
        train(model, optimizer, train_loader, val_loader, DEVICE, config['pretrain_epochs'], 128, scheduler)
        print(f"Dense Pre-training finished in {(time.time() - t_start) / 60:.1f} min")
        torch.save(model.state_dict(), config['dense_ckpt'])

    # Evaluate Baseline
    _, dense_acc = evaluate(model, test_loader, DEVICE, 128)
    print(f"Dense Baseline Accuracy: {dense_acc * 100:.2f}%")
    dense_state_dict = copy.deepcopy(model.state_dict())

    torch.save(model.state_dict(), config['dense_ckpt'])
    print(f"Saved baseline model to {config['dense_ckpt']}")

    # ---------------- Phase 2 & 3: Prune & Fine-tune ----------------
    pruners = {
        "Random": Rand,
        "Magnitude": Mag,
        "SynFlow": SynFlow,
        "SNIP": SNIP,
        "GraSP": GraSP,
    }
    ratios = [0.5, 0.9]  # Sparsity (% Removed)
    mask_type = "global"

    results = {}  # (method, ratio) -> (test_acc, time_min)
    all_val_acc = {}  # (method, ratio) -> [acc_epoch_1, ...]
    finetune_epochs = 40

    for name, PrunerClass in pruners.items():
        for sparsity in ratios:
            print(f"\n===== {name} | Pruning {int(sparsity * 100)}% (Keeping {100 - int(sparsity * 100)}%) =====")

            # 1. Reload Dense Model
            model.load_state_dict(copy.deepcopy(dense_state_dict))
            model.to(DEVICE)

            # 2. Prune
            start_time = time.time()
            masked_params = prune(model, DEVICE, PrunerClass, sample_dataloader, mask_type, sparsity)

            for mask, weight in masked_params:
                mask.requires_grad_(False)
                weight.requires_grad_(True)

            # 3. Fine-tune
            optimizer_ft = SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
            scheduler_ft = MultiStepLR(optimizer_ft, milestones=[20, 30], gamma=0.1)

            _, _, _, val_acc_hist = train(
                model=model,
                optimizer=optimizer_ft,
                train_loader=train_loader,
                val_loader=val_loader,
                DEVICE=DEVICE,
                epochs=finetune_epochs,
                batch_size=128,
                scheduler=scheduler_ft,
            )

            # 4. Evaluation
            _, test_acc = evaluate(model, test_loader, DEVICE, 128)
            total_time = (time.time() - start_time) / 60
            results[(name, sparsity)] = (test_acc, total_time)
            all_val_acc[(name, sparsity)] = val_acc_hist

            print(
                f"→ {name} @ {int(sparsity * 100)}% Pruned | Test Acc: {test_acc * 100:.2f}% | FT Time: {total_time:.1f} min")

    # ---------------- Plotting ----------------
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Define consistent colors for methods
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors_cycle = prop_cycle.by_key()['color']
    method_colors = {name: colors_cycle[i % len(colors_cycle)] for i, name in enumerate(pruners.keys())}

    # (1) Fine-tuning Validation Accuracy
    for (name, sparsity), acc in all_val_acc.items():
        label = f"{name} {int(sparsity * 100)}%"

        # Style logic: 90% is dotted, 50% is solid. Same color per method.
        color = method_colors[name]
        linestyle = ':' if sparsity == 0.9 else '-'

        axes[0].plot(range(1, finetune_epochs + 1), np.array(acc) * 100, label=label, color=color, linestyle=linestyle)

    axes[0].set_title(f"Fine-tuning Recovery ({config['name']})")
    axes[0].set_xlabel("Fine-tune Epoch")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].axhline(y=dense_acc * 100, color='k', linestyle='--', label='Dense Baseline')
    axes[0].legend(fontsize='small', loc='lower right')
    axes[0].grid(True)

    # (2) Final Test Accuracy
    methods_list = list(pruners.keys())
    accs50 = [results[(m, 0.5)][0] * 100 for m in methods_list]
    accs90 = [results[(m, 0.9)][0] * 100 for m in methods_list]
    x = np.arange(len(methods_list))
    width = 0.35

    axes[1].bar(x - width / 2, accs50, width, label="50% Pruned")
    axes[1].bar(x + width / 2, accs90, width, label="90% Pruned")
    axes[1].axhline(y=dense_acc * 100, color='k', linestyle='--', label='Dense Baseline')
    axes[1].set_xticks(x);
    axes[1].set_xticklabels(methods_list)
    axes[1].set_ylabel("Test Accuracy (%)")
    axes[1].set_title("Final Test Accuracy")
    axes[1].legend();
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].set_ylim(bottom=10)

    # (3) Time
    times50 = [results[(m, 0.5)][1] for m in methods_list]
    times90 = [results[(m, 0.9)][1] for m in methods_list]
    axes[2].bar(x - width / 2, times50, width, color="orange", label="50% Pruned")
    axes[2].bar(x + width / 2, times90, width, color="red", label="90% Pruned")
    axes[2].set_xticks(x);
    axes[2].set_xticklabels(methods_list)
    axes[2].set_ylabel("Time (min)")
    axes[2].set_title("Fine-tuning Time")
    axes[2].legend();
    axes[2].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()

    # ---------------- Summary ----------------
    print("\n=== Final Results ===")
    print(f"Experiment: {config['name']}")
    print(f"Dense Baseline: {dense_acc * 100:.2f}%")
    print(f"{'Method':<12} | {'Sparsity':<10} | {'Test Acc':<10} | {'FT Time':<10}")
    for name in pruners.keys():
        for sparsity in ratios:
            acc, t = results[(name, sparsity)]
            print(f"{name:<12} | {int(sparsity * 100)}%       | {acc * 100:6.2f}%    | {t:.1f} min")


def main():
    # ==========================================
    # CONFIGURATION SWITCH
    # ==========================================
    EXPERIMENT = "VGG_CIFAR10"
    # EXPERIMENT = "RESNET_CIFAR100"
    # ==========================================

    if EXPERIMENT == "RESNET_CIFAR100":
        config = {
            'name': 'VGG-16 on CIFAR-10',
            'dataset_class': torchvision.datasets.CIFAR10,
            'model_init': lambda: VGG(num_classes=10),
            'pretrain_epochs': 160,
            'norm_mean': (0.4914, 0.4822, 0.4465),
            'norm_std': (0.2023, 0.1994, 0.2010),
            'dense_ckpt': "dense_vgg_cifar10.pth"
        }

    elif EXPERIMENT == "RESNET_CIFAR100":
        config = {
            'name': 'ResNet-32 on CIFAR-100',
            'dataset_class': torchvision.datasets.CIFAR100,
            'model_init': lambda: ResNet32Masked(num_classes=100),
            'pretrain_epochs': 160,
            'norm_mean': (0.5071, 0.4865, 0.4409),
            'norm_std': (0.2673, 0.2564, 0.2762),
            'dense_ckpt': "dense_resnet32_cifar100.pth"
        }

    run_experiment(config)


if __name__ == "__main__":
    main()