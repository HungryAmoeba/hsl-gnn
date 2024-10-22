import torch
import torch.nn.functional as F
from torchmetrics import Accuracy, MeanMetric, MaxMetric
import pytorch_lightning as L
from typing import Tuple, Dict, Any

class LightningGNN(L.LightningModule):
    def __init__(self, gnn, optimizer, scheduler=None):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.gnn = gnn
        self.optimizer = optimizer
        self.scheduler = scheduler

        # Loss function
        self.criterion = torch.nn.NLLLoss()

        # Metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy(task="multiclass", num_classes=gnn.num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=gnn.num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=gnn.num_classes)

        # For averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # For tracking the best validation accuracy
        self.val_acc_best = MaxMetric()

    def forward(self, x):
        return self.gnn(x)

    def on_train_start(self):
        """Reset metrics at the start of training."""
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def model_step(self, batch: Tuple[torch.Tensor, torch.Tensor]):
        """Perform a model step for a batch."""
        out = self.gnn(batch)
        loss = self.criterion(out, batch.y)
        preds = torch.argmax(out, dim=1)
        return loss, preds, batch.y

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """Training step with logging of loss and accuracy."""
        loss, preds, targets = self.model_step(batch)

        # Update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """Validation step with logging of loss and accuracy."""
        loss, preds, targets = self.model_step(batch)

        # Update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        """Track the best validation accuracy."""
        acc = self.val_acc.compute()
        self.val_acc_best(acc)
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """Test step with logging of loss and accuracy."""
        loss, preds, targets = self.model_step(batch)

        # Update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizers and learning rate schedulers."""
        optimizer = self.hparams.optimizer(self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
