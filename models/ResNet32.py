from torch.nn import functional as F
from VGG import Conv2dMasked, LinearMasked
import torch, time, numpy as np, matplotlib.pyplot as plt
from torch import nn
from torch.optim import SGD
import torchvision
from torch.utils.data import DataLoader, random_split

from Pruners.Pruners import Rand, Mag, SNIP, GraSP, SynFlow
from train import train, evaluate

# ---------- Residual Block ----------

class BasicBlockMasked(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = Conv2dMasked(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = Conv2dMasked(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # Projection shortcut if dimensions change
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                Conv2dMasked(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


# ---------- Masked ResNet-32 ----------

class ResNetMasked(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100):
        super().__init__()
        self.in_planes = 16

        self.conv1 = Conv2dMasked(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = LinearMasked(64 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


def ResNet32Masked(num_classes=100):
    # 3 stages × {5, 5, 5} residual blocks → 32 layers total
    return ResNetMasked(BasicBlockMasked, [5, 5, 5], num_classes=num_classes)

def get_masked_parameters(model):
    pairs = []
    for module in model.modules():
        if hasattr(module, "weight") and hasattr(module, "weight_mask"):
            pairs.append((module.weight_mask, module.weight))
    return pairs

def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using:", torch.cuda.get_device_name(0))

    # ---------------- Data ----------------
    transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            (0.5071, 0.4865, 0.4409),
            (0.2673, 0.2564, 0.2762)
        ),
    ])

    train_dataset = torchvision.datasets.CIFAR100("./data", train=True, download=True, transform=transform)
    test_dataset  = torchvision.datasets.CIFAR100("./data", train=False, download=True, transform=transform)
    train_dataset, val_dataset = random_split(train_dataset, [int(0.9*len(train_dataset)), int(0.1*len(train_dataset))])

    batch_size, epochs = 128, 160
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_dataset,   batch_size, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)
    test_loader  = DataLoader(test_dataset,  batch_size, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)

    # sample data for pruning score computation
    sample_inputs, sample_targets = next(iter(train_loader))
    sample_dataloader = [(sample_inputs, sample_targets)]
    loss_fn = nn.CrossEntropyLoss()

    # ---------------- Experiments ----------------
    pruners = {
        "Random": Rand,
        "Magnitude": Mag,
        "SynFlow": SynFlow,
        "SNIP": SNIP,
        "GraSP": GraSP,
    }
    ratios = [0.5, 0.9]        # 50% and 90% remaining weights
    mask_type = "global"

    results = {}  # results[(method, ratio)] = test_acc
    all_val_acc = {}

    for name, PrunerClass in pruners.items():
        for ratio in ratios:
            print(f"\n===== {name} | Retain {int(ratio*100)}% weights =====")
            sparsity = 1 - ratio

            model = ResNet32Masked(num_classes=100).to(DEVICE)
            masked_params = get_masked_parameters(model)

            # ---- Prune ----
            start = time.time()
            pruner = PrunerClass(masked_params)
            pruner.score(model, loss_fn, sample_dataloader, DEVICE)
            pruner.mask(sparsity, mask_type)
            pruner.apply_mask()

            for mask, weight in masked_params:
                mask.requires_grad_(False)
                weight.requires_grad_(True)

            # ---- Train ----
            optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120], gamma=0.1)

            _, _, _, val_acc = train(
                model=model,
                optimizer=optimizer,
                train_loader=train_loader,
                val_loader=val_loader,
                DEVICE=DEVICE,
                epochs=epochs,
                batch_size=batch_size,
                scheduler=scheduler,
            )

            all_val_acc[(name, ratio)] = val_acc

            # ---- Evaluate ----
            _, test_acc = evaluate(model, test_loader, DEVICE, batch_size)
            elapsed = time.time() - start
            results[(name, ratio)] = (test_acc, elapsed/60)
            print(f"→ {name}, {int(ratio*100)}% kept | Test Acc {test_acc:.4f} | {elapsed/60:.1f} min")

    # ---------------- Plot ----------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (1) Val accuracy vs epoch for each method/ratio
    for (name, ratio), acc in all_val_acc.items():
        label = f"{name} {int(ratio*100)}%"
        axes[0].plot(range(1, epochs+1), np.array(acc)*100, label=label)
    axes[0].set_title("Validation Accuracy vs Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].legend()
    axes[0].grid(True)

    # (2) Final test accuracy bar chart
    methods, accs50, accs90 = [], [], []
    for name in pruners.keys():
        acc50 = results[(name, 0.5)][0]*100
        acc90 = results[(name, 0.9)][0]*100
        methods.append(name); accs50.append(acc50); accs90.append(acc90)
    x = np.arange(len(methods))
    axes[1].bar(x-0.15, accs50, width=0.3, label="50% kept")
    axes[1].bar(x+0.15, accs90, width=0.3, label="10% kept (90% pruned)")
    axes[1].set_xticks(x); axes[1].set_xticklabels(methods)
    axes[1].set_ylabel("Test Accuracy (%)")
    axes[1].set_title("Final Test Accuracy per Method")
    axes[1].legend(); axes[1].grid(axis="y")

    # (3) Training time
    times50 = [results[(m, 0.5)][1] for m in methods]
    times90 = [results[(m, 0.9)][1] for m in methods]
    axes[2].bar(x-0.15, times50, width=0.3, color="orange", label="50% kept")
    axes[2].bar(x+0.15, times90, width=0.3, color="red", label="10% kept")
    axes[2].set_xticks(x); axes[2].set_xticklabels(methods)
    axes[2].set_ylabel("Training Time (min)")
    axes[2].set_title("Training Time per Method")
    axes[2].legend(); axes[2].grid(axis="y")

    plt.tight_layout()
    plt.show()

    # ---------------- Print Summary ----------------
    print("\n=== Final Results (Test Accuracy %) ===")
    for name in pruners.keys():
        for ratio in ratios:
            acc, t = results[(name, ratio)]
            print(f"{name:9s} | {int(ratio*100)}% kept | {acc*100:6.2f}% | {t:5.1f} min")


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
