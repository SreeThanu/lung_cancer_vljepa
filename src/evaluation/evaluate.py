"""
Fine-Tuning & Evaluation Script (Research-Ready Final Version)

Features:
✔ Deterministic seed control
✔ Pretrained JEPA encoder loading
✔ Linear probing or full fine-tuning
✔ Early stopping
✔ TensorBoard logging
✔ Sensitivity & Specificity reporting
✔ Grad-CAM visualization support
✔ Periodic + best checkpoint saving
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
from pathlib import Path
import sys
from tqdm import tqdm
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from models.encoder_3d import ViT3DEncoder
from models.classification_head import ClassificationHead
from utils.config import Config
from utils.dataloader import create_dataloaders
from utils.metrics import (
    compute_classification_metrics,
    plot_roc_curve,
    plot_confusion_matrix,
    print_metrics_summary
)
from utils.seed import set_global_seed
from explainability.gradcam import GradCAM3D


# ============================================================
# MODEL
# ============================================================

class LungCancerClassifier(nn.Module):

    def __init__(self, encoder: nn.Module, classifier: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier

    def forward(self, x):

        features = self.encoder(x)       # [B, N, D]
        features = features.mean(dim=1)  # Global average pooling
        logits = self.classifier(features)

        return logits


# ============================================================
# TRAINER
# ============================================================

class ClassificationTrainer:

    def __init__(self, config: Config, checkpoint_path: str = None):

        self.config = config
        self.device = torch.device(
            config.get("project.device", "cuda")
            if torch.cuda.is_available()
            else "cpu"
        )

        print(f"Using device: {self.device}")

        self.model = self._build_model(checkpoint_path)
        self._build_optimizer()

        self.criterion = nn.CrossEntropyLoss(
            weight=self._get_class_weights()
            if config.get("finetuning.use_class_weights", False)
            else None
        )

        self.scaler = GradScaler(
            enabled=config.get("finetuning.amp", True)
        )

        self.best_auc = 0.0
        self.early_stop_counter = 0
        self.patience = config.get("finetuning.early_stopping_patience", 10)

        self.writer = SummaryWriter(
            log_dir=config.get("logging.log_dir", "./logs")
        )


    # --------------------------------------------------------

    def _build_model(self, checkpoint_path):

        config = self.config

        encoder = ViT3DEncoder(
            input_size=tuple(config.get("data.input_shape")[1:]),
            patch_size=tuple(config.get("model.encoder.patch_size")),
            in_channels=1,
            embed_dim=config.get("model.encoder.embed_dim"),
            depth=config.get("model.encoder.depth"),
            num_heads=config.get("model.encoder.num_heads"),
        )

        if checkpoint_path and Path(checkpoint_path).exists():
            print(f"Loading pretrained encoder from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            encoder.load_state_dict(checkpoint["context_encoder"])

        encoder = encoder.to(self.device)

        # Freeze encoder if linear probing
        if config.get("finetuning.freeze_encoder", False):
            print("🔒 Linear probing mode — freezing encoder")
            for param in encoder.parameters():
                param.requires_grad = False

        classifier = ClassificationHead(
            embed_dim=config.get("model.encoder.embed_dim"),
            hidden_dims=config.get("model.classifier.hidden_dims"),
            num_classes=config.get("model.classifier.num_classes"),
            dropout=config.get("model.classifier.dropout"),
        ).to(self.device)

        return LungCancerClassifier(encoder, classifier).to(self.device)


    # --------------------------------------------------------

    def _build_optimizer(self):

        config = self.config

        if config.get("finetuning.freeze_encoder", False):
            params = self.model.classifier.parameters()
        else:
            params = self.model.parameters()

        self.optimizer = torch.optim.AdamW(
            params,
            lr=config.get("finetuning.learning_rate"),
            weight_decay=config.get("finetuning.weight_decay"),
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get("finetuning.epochs"),
            eta_min=config.get("finetuning.min_lr"),
        )


    # --------------------------------------------------------

    def _get_class_weights(self):
        return torch.tensor([1.0, 2.0]).to(self.device)


    # --------------------------------------------------------

    def train_epoch(self, dataloader):

        self.model.train()
        epoch_loss = 0.0

        for volumes, labels in tqdm(dataloader, desc="Training"):

            volumes = volumes.to(self.device)
            labels = labels.to(self.device)

            with autocast(
                device_type=self.device.type,
                enabled=self.config.get("finetuning.amp", True)
            ):
                logits = self.model(volumes)
                loss = self.criterion(logits, labels)

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            epoch_loss += loss.item()

        return epoch_loss / len(dataloader)


    # --------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, dataloader):

        self.model.eval()

        all_preds = []
        all_labels = []
        all_probs = []

        for volumes, labels in tqdm(dataloader, desc="Evaluating"):

            volumes = volumes.to(self.device)

            logits = self.model(volumes)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

        metrics = compute_classification_metrics(
            np.array(all_labels),
            np.array(all_preds),
            np.array(all_probs)
        )

        return metrics, all_labels, all_preds, all_probs


    # --------------------------------------------------------

    def train(self, train_loader, val_loader, epochs):

        save_freq = self.config.get("finetuning.save_freq", 10)

        for epoch in range(epochs):

            train_loss = self.train_epoch(train_loader)
            val_metrics, _, _, _ = self.evaluate(val_loader)

            val_auc = val_metrics.get("roc_auc", 0)

            print(f"\nEpoch {epoch+1}/{epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val ROC-AUC: {val_auc:.4f}")
            print(f"Sensitivity: {val_metrics.get('sensitivity', 0):.4f}")
            print(f"Specificity: {val_metrics.get('specificity', 0):.4f}")

            self.writer.add_scalar("Loss/Train", train_loss, epoch)
            self.writer.add_scalar("AUC/Val", val_auc, epoch)

            if val_auc > self.best_auc:
                self.best_auc = val_auc
                self.save_checkpoint("best_model.pth")
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1

            if self.early_stop_counter >= self.patience:
                print("⛔ Early stopping triggered.")
                break

            if (epoch + 1) % save_freq == 0:
                self.save_checkpoint(f"epoch_{epoch+1}.pth")

            self.scheduler.step()

        self.save_checkpoint("final_model.pth")
        self.writer.close()


    # --------------------------------------------------------

    def save_checkpoint(self, filename):

        save_dir = Path(self.config.get("pretraining.save_dir"))
        save_dir.mkdir(parents=True, exist_ok=True)

        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_auc": self.best_auc
        }, save_dir / filename)

        print(f"✔ Saved: {filename}")


# ============================================================
# MAIN
# ============================================================

def main():

    config = Config("configs/jepa_config.yaml")

    # 🔥 Deterministic seed
    set_global_seed(config.get("project.seed", 42))

    checkpoint_dir = Path(config.get("pretraining.save_dir"))
    checkpoints = list(checkpoint_dir.glob("jepa_checkpoint_*.pth"))

    latest_checkpoint = None
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
        print(f"Using checkpoint: {latest_checkpoint}")

    metadata_path = Path(config.get("data.processed_dir")) / "metadata" / "labels.csv"

    train_loader, val_loader, test_loader = create_dataloaders(
        config, str(metadata_path)
    )

    trainer = ClassificationTrainer(
        config,
        str(latest_checkpoint) if latest_checkpoint else None
    )

    trainer.train(
        train_loader,
        val_loader,
        config.get("finetuning.epochs")
    )

    print("\nFINAL TEST SET EVALUATION")
    test_metrics, labels, preds, probs = trainer.evaluate(test_loader)

    print_metrics_summary(test_metrics)

    plot_roc_curve(np.array(labels), np.array(probs))
    plot_confusion_matrix(np.array(labels), np.array(preds))

    # -------------------------------------------------------
    # Grad-CAM Visualization (for publication)
    # -------------------------------------------------------

    gradcam = GradCAM3D(
        model=trainer.model,
        target_layer=trainer.model.encoder.transformer.layers[-1]
    )

    sample_volume, _ = next(iter(test_loader))
    sample_volume = sample_volume.to(trainer.device)

    cam = gradcam.generate(sample_volume[0:1])

    print("✔ Grad-CAM generated for sample case.")


if __name__ == "__main__":
    main()
