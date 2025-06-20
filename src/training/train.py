import os
from pathlib import Path

import click
import pandas as pd
import torch
import torch.nn as nn
import yaml
import logging

from src.training.models import MODEL_REGISTRY
from src.training.dataloaders import get_dataloaders
from src.training.optimizers_and_schedulers import get_scheduler
from src.training.train_utils import train_one_epoch, validate_one_epoch
from src.training.optimizers_and_schedulers import get_optimizer
from src.training.train_utils import plot_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).parent.parent.parent / "training_artifacts"


def load_config(path: str) -> dict:
    with open(path, "r") as file:
        return yaml.safe_load(file)


@click.command()
@click.option(
    "--config_path",
    default="config/config.yaml",
    help="Path to the configuration YAML file.",
)
def main(
    config_path: str,
) -> None:
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    path_to_save = ARTIFACTS_DIR / config["model_save_path"]
    os.makedirs(path_to_save, exist_ok=True)

    model_config = config["model"]
    model_name = model_config["name"]
    model_params = model_config.get("params", {})

    if model_name in MODEL_REGISTRY:
        model_class = MODEL_REGISTRY[model_name]
        model = model_class(**model_params).to(device)
    else:
        raise ValueError(f"Model '{model_name}' is not registered in MODEL_REGISTRY.")

    train_loader, val_loader = get_dataloaders(config)
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model.parameters(), config)
    scheduler = get_scheduler(optimizer, config)

    best_val_loss = float("inf")
    patience = config["early_stopping_patience"]
    counter = 0

    history = {
        "epoch": [],
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    for epoch in range(config["num_epochs"]):
        train_loss, train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_metrics = validate_one_epoch(model, val_loader, criterion, device)

        if scheduler:
            scheduler.step()

        logger.info(
            f"Epoch {epoch + 1}/{config['num_epochs']} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} - Train Accuracy: {train_metrics.get('accuracy', 0):.4f}, Val Accuracy: {val_metrics.get('accuracy', 0):.4f}"
        )

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_metrics.get("accuracy", 0))
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_metrics.get("accuracy", 0))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            model_save_path = path_to_save / f"{model_name}_epoch_{epoch + 1}.pth"
            torch.save(model.state_dict(), model_save_path)
            logger.info(f"Saved Best Model at {model_save_path}!")
        else:
            counter += 1
            if counter >= patience:
                logger.info("Early stopping triggered.")
                break

    metrics_df = pd.DataFrame(history)
    metrics_save_path = path_to_save / "training_metrics.csv"
    metrics_df.to_csv(metrics_save_path, index=False)
    logger.info(f"Training metrics saved to {metrics_save_path}")

    plot_metrics(history, path_to_save)


if __name__ == "__main__":
    main()
