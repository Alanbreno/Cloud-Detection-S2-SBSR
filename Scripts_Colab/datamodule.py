import torch
import pandas as pd
import pytorch_lightning as pl
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataset import CoreDataset
import mlstac

# Pipeline de augmentations com Shift e Rotação
augmentation_pipeline = A.Compose(
    [
        A.HorizontalFlip(p=0.5),  # Flip horizontal com 50% de chance
        A.VerticalFlip(p=0.5),  # Flip vertical com 50% de chance
        A.RandomRotate90(p=0.5),  # Rotação em múltiplos de 90 graus (90, 180, 270)
        ToTensorV2(),  # Converte para tensores PyTorch
    ]
)


class CoreDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 4,
        train_index_mask: int = 13,
        val_index_mask: int = 13,
        test_index_mask: int = 13,
    ):
        super().__init__()
        data = mlstac.load(snippet="./drive/MyDrive/CloudSen12+/main.json").metadata
        
        # Separar o DataFrame em datasets de treino, validação e teste
        self.train_dataset = data[(data['proj_shape'] == 509) & (data['label_type'] == 'high')& (data['split'] == 'train')]
        self.validation_dataset = data[(data['proj_shape'] == 509) & (data['label_type'] == 'high')& (data['split'] == 'validation')]
        self.test_dataset = data[(data['proj_shape'] == 509) & (data['label_type'] == 'high')& (data['split'] == 'test')]

        # Definir o batch_size
        self.batch_size = batch_size
        self.train_index_mask = train_index_mask
        self.val_index_mask = val_index_mask
        self.test_index_mask = test_index_mask

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=CoreDataset(
                self.train_dataset,
                index_mask=self.train_index_mask,
                augmentations=augmentation_pipeline,
            ),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=11
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=CoreDataset(
                self.validation_dataset, index_mask=self.val_index_mask
            ),
            batch_size=self.batch_size,
            num_workers=11
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=CoreDataset(self.test_dataset, index_mask=self.test_index_mask),
            batch_size=self.batch_size,
            num_workers=11
        )
