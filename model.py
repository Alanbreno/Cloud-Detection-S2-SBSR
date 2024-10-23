import torch
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch.nn.functional as F
from torch.optim import lr_scheduler


class UNet_S2_Br(pl.LightningModule):
    def __init__(self, encoder_name, classes, in_channels, learning_rate):
        super().__init__()
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=None,
            classes=classes,
            in_channels=in_channels,
        )
        self.loss = torch.nn.CrossEntropyLoss()
        self.lr = learning_rate
        self.save_hyperparameters()
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, x):
        return self.model(x)

    def shared_epoch_end(self, outputs, stage):
        # aggregate step metics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro")
        iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro")
        f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="macro")
        recall = smp.metrics.recall(tp, fp, fn, tn, reduction="macro")

        metrics = {
            f"{stage}_acuracia": accuracy,
            f"{stage}_dataset_iou": iou,
            f"{stage}_f1_score": f1_score,
            f"{stage}_recall": recall,
        }

        self.log_dict(metrics, on_epoch=True)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)

        output = F.softmax(y_hat, dim=1)
        output = torch.argmax(output, dim=1)

        tp, fp, fn, tn = smp.metrics.get_stats(
            output, y, mode="multiclass", num_classes=4
        )

        self.training_step_outputs.append(
            {
                "loss": loss,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }
        )

        self.log("train_loss", loss, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        self.shared_epoch_end(self.training_step_outputs, "train")
        # empty set output list
        self.training_step_outputs.clear()
        return

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)

        output = F.softmax(y_hat, dim=1)
        output = torch.argmax(output, dim=1)

        tp, fp, fn, tn = smp.metrics.get_stats(
            output, y, mode="multiclass", num_classes=4
        )

        self.validation_step_outputs.append(
            {
                "loss": loss,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }
        )

        self.log("val_loss", loss, on_epoch=True)
        return loss

    def on_validation_epoch_end(self):
        self.shared_epoch_end(self.validation_step_outputs, "valid")
        self.validation_step_outputs.clear()
        return

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)

        output = F.softmax(y_hat, dim=1)
        output = torch.argmax(output, dim=1)

        tp, fp, fn, tn = smp.metrics.get_stats(
            output, y, mode="multiclass", num_classes=4
        )

        self.test_step_outputs.append(
            {
                "loss": loss,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }
        )

        self.log("test_loss", loss, on_epoch=True)
        return loss

    def on_test_epoch_end(self):
        self.shared_epoch_end(self.test_step_outputs, "test")
        # empty set output list
        self.test_step_outputs.clear()
        return

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # Define o scheduler para reduzir LR se a perda de validação não diminuir por 4 épocas
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=4, min_lr=1e-6, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",  # Monitora a perda de validação
                "interval": "epoch",  # Aplica o ajuste por época
                "frequency": 1,  # Frequência de monitoramento (1 = a cada época)
            },
        }
