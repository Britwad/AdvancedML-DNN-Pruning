import torch
from torch import nn
from torch.optim import SGD
import torchvision
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from Pruners.Pruners import SNIP, GraSP, SynFlow, Mag, Pruner
from typing import Tuple, List
from models.VGG import VGG, Conv2dMasked
from prune import prune
import matplotlib.pyplot as plt


def train(
    model: nn.Module,
    optimizer: SGD,
    train_loader: DataLoader,
    val_loader: DataLoader,
    DEVICE: torch.device,
    epochs: int,
    batch_size: int,
    scheduler,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    print("in training now")
    loss = nn.CrossEntropyLoss()
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    model.to(DEVICE)
    for e in tqdm(range(epochs)):
        model.train()
        train_loss = 0.0
        train_acc = 0.0


        # Use to ensure that your global sparsity is consistent with what you initially wanted
        # nz, total = 0, 0
        # for layer in model.modules():
        #     if hasattr(layer, "weight_mask") and hasattr(layer, "weight"):
        #         nz  += torch.count_nonzero(layer.weight).item()
        #         total += layer.weight.numel()
        # print("Global effective sparsity:", nz/total)

        for x_batch, labels in train_loader:
            x_batch, labels = x_batch.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            labels_pred = model(x_batch)
            batch_loss = loss(labels_pred, labels)
            train_loss = train_loss + batch_loss.item()

            labels_pred_max = torch.argmax(labels_pred, 1)
            batch_acc = torch.sum(labels_pred_max == labels)
            train_acc = train_acc + batch_acc.item()

            batch_loss.backward()
            optimizer.step()
        train_losses.append(train_loss / len(train_loader))
        train_accuracies.append(train_acc / (batch_size * len(train_loader)))
        
        model.eval()
        val_loss = 0.0
        val_acc = 0.0

        with torch.no_grad():
            for v_batch, labels in val_loader:
                v_batch, labels = v_batch.to(DEVICE), labels.to(DEVICE)
                labels_pred = model(v_batch)
                v_batch_loss = loss(labels_pred, labels)
                val_loss = val_loss + v_batch_loss.item()

                v_pred_max = torch.argmax(labels_pred, 1)
                batch_acc = torch.sum(v_pred_max == labels)
                val_acc = val_acc + batch_acc.item()
        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(val_acc / (batch_size * len(val_loader)))
        scheduler.step()
    return train_losses, train_accuracies, val_losses, val_accuracies


def evaluate(
    model: nn.Module, loader: DataLoader, DEVICE, batch_size
) -> Tuple[float, float]:
    """Computes test loss and accuracy of model on loader."""
    loss = nn.CrossEntropyLoss()
    model.eval()
    test_loss = 0.0
    test_acc = 0.0
    with torch.no_grad():
        for batch, labels in loader:
            batch, labels = batch.to(DEVICE), labels.to(DEVICE)
            y_batch_pred = model(batch)
            batch_loss = loss(y_batch_pred, labels)
            test_loss = test_loss + batch_loss.item()

            pred_max = torch.argmax(y_batch_pred, 1)
            batch_acc = torch.sum(pred_max == labels)
            test_acc = test_acc + batch_acc.item()
        test_loss = test_loss / len(loader)
        test_acc = test_acc / (batch_size * len(loader))
        return test_loss, test_acc


def main():
    torch.backends.cudnn.benchmark = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = torchvision.datasets.CIFAR10(
        "./data", train=True, download=True, transform=torchvision.transforms.ToTensor()
    )
    test_dataset = torchvision.datasets.CIFAR10(
        "./data",
        train=False,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )

    batch_size = 128
    epochs = 120
    train_dataset, val_dataset = random_split(
        train_dataset, [int(0.9 * len(train_dataset)), int(0.1 * len(train_dataset))]
    )

    kwargs = {'num_workers': 2, 'pin_memory': True, 'persistent_workers': True}

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

    sample_inputs, sample_targets = next(iter(train_loader))
    sample_dataloader = [(sample_inputs, sample_targets)]
    model = VGG(10).to(DEVICE)
    

    mask_type = "global"
    sparsity = 0.2
    masked_parameters = prune(model, DEVICE, SNIP, sample_dataloader, mask_type, sparsity)
    print("---Pruning Complete---")

    for mask, weight in masked_parameters:
        mask.grad = None
        mask.requires_grad_(False)
        weight.requires_grad_(True)



    print("---training begins---")
    optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[60, 120], gamma=0.1
    )
    train_loss, train_accuracy, val_loss, val_accuracy = train(
        model,
        optimizer,
        train_loader,
        val_loader,
        DEVICE,
        epochs,
        batch_size,
        scheduler,
    )
    plot(epochs, train_accuracy, val_accuracy)

    print("---running test---")
    print(evaluate(model, test_loader, DEVICE, batch_size))



def plot(epochs,train_accuracy, val_accuracy):
    epochs = range(1, epochs + 1)
    plt.plot(epochs, train_accuracy, label="Train Accuracy")
    plt.plot(epochs, val_accuracy, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy for CIFAR-10 vs Epoch")
    plt.show()


if __name__ == "__main__":
    main()
