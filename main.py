import warnings
warnings.filterwarnings("ignore")

import argparse
import lightning as L
import torch
import os
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, MulticlassConfusionMatrix, F1Score
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, R2Score, SpearmanCorrCoef
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger 
import wandb  
import matplotlib.pyplot as plt

from models import Model
from dataset import DataModule
from utils import layer_wise_decay
import numpy as np
import torch.nn as nn

class LitClassifier(L.LightningModule):

    def __init__(self, backbone_type, lr=2e-4, lr_decay=0.75, weight_decay=1e-4, img_size=384, hard_label=False):
        super().__init__()
        self.save_hyperparameters()
        self.lr_decay = lr_decay

        # Backbone + head
        self.model = Model(
            backbone_type=backbone_type,
            img_size=img_size,
        )
        self.loss = torch.nn.BCEWithLogitsLoss()
        self.val_auroc = BinaryAUROC()
        self.test_auroc = BinaryAUROC()

    # ---------------------------- forward ----------------------------- #
    def forward(self, cfp):
        return self.model(cfp)

    def training_step(self, batch, batch_idx):
        cfp_img = batch["cfp_image"]
        label = batch["label"]

        outs = self(cfp_img)
        logits = outs['logits']
        loss = self.loss(logits, label.float()).mean()
        self.log("train_loss", loss, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)

        return loss
    
    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch, batch_idx):
        cfp_img = batch["cfp_image"]
        label = batch["label"] > 0.5

        outs = self(cfp_img)
        logits = outs['logits']
        loss = self.loss(logits, label.float()).mean()

        self.log("val_loss", loss, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.val_auroc.update(logits, label)
        return loss


    def on_validation_epoch_end(self):
        val_auroc = self.val_auroc.compute()
        self.log("val_auroc", val_auroc, prog_bar=True, sync_dist=True)
        self.val_auroc.reset()

    def test_step(self, batch, batch_idx):
        cfp_img = batch["cfp_image"]
        label = batch["label"] > 0.5

        outs = self(cfp_img)
        logits = outs['logits']
        loss = self.loss(logits, label.float()).mean()

        self.log("test_loss", loss, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.test_auroc.update(logits, label)
        return loss
        

    def on_test_epoch_end(self):
        test_auroc = self.test_auroc.compute()
        self.log("test_auroc", test_auroc, prog_bar=True, sync_dist=True)
        self.test_auroc.reset()

    # ------------------- optimizer & scheduler ------------------------ #
    def configure_optimizers(self):
        base_lr = self.hparams.lr
        weight_decay = self.hparams.weight_decay
        lr_decay = self.hparams.lr_decay

        if self.hparams.backbone_type in ["vit_small", "retfound_dinov2"]:

            if hasattr(self.model.backbone, "num_layers"):
                num_layers = int(self.model.backbone.num_layers)
            else:
                num_layers = len(getattr(self.model.backbone, "blocks", []))
            

            groups = layer_wise_decay(self.named_parameters(), num_layers=num_layers, base_lr=base_lr, weight_decay=weight_decay, decay_rate=lr_decay)

            optimizer = torch.optim.AdamW(groups)
            
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1e-8, 
                end_factor=1.0,
                total_iters=int(0.1 * self.trainer.max_epochs),
            )
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs - int(0.1 * self.trainer.max_epochs),
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[int(0.1 * self.trainer.max_epochs)],
            )

        else:
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=base_lr,
                weight_decay=weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.trainer.max_epochs
            )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


# ---------------------------------------------------------------------- #
# CLI entry point
# ---------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Train binary classifier with Lightning")

    # Data
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)

    # Model & optimisation
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-2)
    parser.add_argument("--lr_decay", type=float, default=0.65)
    parser.add_argument("--backbone_type", type=str, default="convnext_small")
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)
    parser.add_argument("--hard_label", type=float, default=0.0)

    # Trainer
    parser.add_argument("--accelerator", type=str, default="gpu", help="Training accelerator, e.g. gpu, cpu, auto")
    parser.add_argument("--devices", type=int, default=1, help="How many GPUs to use (if accelerator=gpu)")
    parser.add_argument("--precision", type=str, default="16-mixed", help="32, 16-mixed, bf16-mixed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)

    args = parser.parse_args()
    L.seed_everything(args.seed, workers=True)
    args.hard_label = args.hard_label > 0.5

    # DataModule
    datamodule = DataModule(
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers,
        hard_label = args.hard_label,
    )

    # LightningModule
    model = LitClassifier(
        backbone_type=args.backbone_type,
        lr=args.lr,
        lr_decay=args.lr_decay,
        weight_decay=args.weight_decay,
        img_size=args.img_size,
        hard_label=args.hard_label,
    )
    # Callbacks
    callbacks = [
        ModelCheckpoint(monitor="epoch", mode="max", save_top_k=1, save_last=False, filename="best-{epoch:02d}-{val_acc:.4f}", auto_insert_metric_name=False),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    wandb_logger = WandbLogger(
    )

    # Trainer
    trainer = L.Trainer(
        default_root_dir=
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        strategy="ddp",  # distributed training across GPUs
        precision=args.precision,
        deterministic=True,
        callbacks=callbacks,
        log_every_n_steps=500,
        logger=wandb_logger,
        enable_progress_bar=False,    
        gradient_clip_val=args.gradient_clip_val,  # gradient clipping
        accumulate_grad_batches=args.accumulate_grad_batches,  # gradient accumulation
        check_val_every_n_epoch = 5,
    )

    print(f"Training with {args.accelerator} on {args.devices} devices, precision: {args.precision}")

    # Fit & evaluate
    trainer.fit(model, datamodule=datamodule)
    trainer.test(ckpt_path="last", datamodule=datamodule)


if __name__ == "__main__":
    main()