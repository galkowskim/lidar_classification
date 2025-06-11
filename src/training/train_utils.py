import logging
from pathlib import Path

import torch
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import textwrap
from src.training.metrics import compute_metrics

from sklearn.metrics import confusion_matrix
from src.data_preparation.constants import FOLDER_TO_LABEL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0
    all_preds = []
    all_labels = []

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    metrics = compute_metrics(all_labels, all_preds)
    return epoch_loss, metrics


def validate_one_epoch(model, loader, criterion, device, save_dir: Path = None):
    model.eval()
    running_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    metrics = compute_metrics(all_labels, all_preds)

    if save_dir:
        save_confusion_matrix(all_preds, all_labels, loader, save_dir)

    return epoch_loss, metrics


def save_confusion_matrix(all_preds, all_labels, loader, save_dir: Path) -> None:
    """
    Saves a confusion matrix plot to the specified directory.
    Args:
        all_preds (list): List of predicted labels.
        all_labels (list): List of true labels.
        loader (DataLoader): DataLoader containing the dataset.
        save_dir (Path): Directory where the confusion matrix will be saved.
    """
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(14, 14))
    class_labels = loader.dataset.classes
    class_ids = list(range(len(class_labels)))

    mapped_labels = [FOLDER_TO_LABEL.get(name, name) for name in class_labels]

    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.xlabel("Predicted class", fontsize=18)
    plt.ylabel("True class", fontsize=18)
    plt.title("Confusion Matrix", fontsize=22)

    plt.xticks(class_ids, class_ids, fontsize=14)
    plt.yticks(class_ids, class_ids, fontsize=14)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14,
            )

    max_width = 25
    legend_labels = [f"{i}: " + "\n   ".join(textwrap.wrap(mapped_labels[i], max_width)) for i in class_ids]
    handles = [mpatches.Patch(color="none", label=label) for label in legend_labels]

    legend = plt.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=14,
        ncol=1,
        frameon=True,
        title="Class mapping",
        borderaxespad=0,
        handlelength=1,
        handletextpad=0.5,
        columnspacing=0.5,
    )
    legend.get_title().set_fontsize(16)

    plot_path = save_dir / "confusion_matrix.png"
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Confusion matrix saved to {plot_path}")


def plot_metrics(history: dict, save_dir: Path) -> None:
    epochs = history["epoch"]
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss", color="blue")
    plt.plot(epochs, history["val_loss"], label="Test Loss", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss over Epochs")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_accuracy"], label="Train Accuracy", color="blue")
    plt.plot(epochs, history["val_accuracy"], label="Test Accuracy", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("Balanced accuracy")
    plt.title("Accuracy over Epochs")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plot_path = save_dir / "training_metrics.png"
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Training metrics plot saved to {plot_path}")
