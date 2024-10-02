# %% [markdown]
# # **Imports e Definições**
# 

# %%
import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import glob
from torch.utils.data import Dataset
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import segmentation_models_pytorch as smp
import torch.nn.functional as F
from pytorch_lightning.loggers import TensorBoardLogger
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.optim import lr_scheduler

# %% [markdown]
# ## **Criação do Dataset com os paths das imagens**

# %%
diretorio_base = '/home/mseruffo/'

# Listar arquivos de treino, validação e teste para as imagens de 512x512
train_files_512_nolabel = glob.glob(diretorio_base + 'CloudSen12_Br_Resized/p509/nolabel/train_br/*/*.tif')
val_files_512_nolabel = glob.glob(diretorio_base + 'CloudSen12_Br_Resized/p509/nolabel/val_br/*/*.tif')

# Criar DataFrames e adicionar a coluna 'set_type'
train_df = pd.DataFrame(train_files_512_nolabel, columns=['file_path'])
train_df['set_type'] = 'train'
val_df = pd.DataFrame(val_files_512_nolabel, columns=['file_path'])
val_df['set_type'] = 'val'

# Concatenar todos os DataFrames
df_nolabel = pd.concat([train_df, val_df], ignore_index=True)

# Listar arquivos de treino, validação e teste para as imagens de 512x512
train_files_512_scribble = glob.glob(diretorio_base + 'CloudSen12_Br_Resized/p509/scribble/train_br/*/*.tif')
val_files_512_scribble = glob.glob(diretorio_base + 'CloudSen12_Br_Resized/p509/scribble/val_br/*/*.tif')
test_files_512_scribble = glob.glob(diretorio_base + 'CloudSen12_Br_Resized/p509/scribble/test_br/*/*.tif')

# Criar DataFrames e adicionar a coluna 'set_type'
train_df = pd.DataFrame(train_files_512_scribble, columns=['file_path'])
train_df['set_type'] = 'train'

val_df = pd.DataFrame(val_files_512_scribble, columns=['file_path'])
val_df['set_type'] = 'val'

test_df = pd.DataFrame(test_files_512_scribble, columns=['file_path'])
test_df['set_type'] = 'test'

# Concatenar todos os DataFrames
df_scribble = pd.concat([train_df, val_df, test_df], ignore_index=True)


# Listar arquivos de treino, validação e teste para as imagens de 512x512
train_files_512 = glob.glob(diretorio_base + 'CloudSen12_Br_Resized/p509/high/train_br/*/*.tif')
val_files_512 = glob.glob(diretorio_base + 'CloudSen12_Br_Resized/p509/high/val_br/*/*.tif')
test_files_512 = glob.glob(diretorio_base + 'CloudSen12_Br_Resized/p509/high/test_br/*/*.tif')

# Listar arquivos de treino, validação e teste para as imagens de 2048x2048
train_files_2048 = glob.glob(diretorio_base + 'CloudSen12_Br_Resized/p2000/train_br/*/*.tif')
val_files_2048 = glob.glob(diretorio_base + 'CloudSen12_Br_Resized/p2000/val_br/*/*.tif')
test_files_2048 = glob.glob(diretorio_base + 'CloudSen12_Br_Resized/p2000/test_br/*/*.tif')

# Concatenando os conjuntos de treino, validação e teste
train_files = train_files_512 + train_files_2048
val_files = val_files_512 + val_files_2048
test_files = test_files_512 + test_files_2048

# Criar DataFrames e adicionar a coluna 'set_type'
train_df = pd.DataFrame(train_files, columns=['file_path'])
train_df['set_type'] = 'train'

val_df = pd.DataFrame(val_files, columns=['file_path'])
val_df['set_type'] = 'val'

test_df = pd.DataFrame(test_files, columns=['file_path'])
test_df['set_type'] = 'test'

# Concatenar todos os DataFrames
df = pd.concat([train_df, val_df, test_df], ignore_index=True)

# %% [markdown]
# # **Define o data augmentation**

# %%
# Pipeline de augmentations com Shift e Rotação
augmentation_pipeline = A.Compose([
    A.HorizontalFlip(p=0.5),  # Flip horizontal com 50% de chance
    A.VerticalFlip(p=0.5),    # Flip vertical com 50% de chance
    A.RandomRotate90(p=0.5),  # Rotação em múltiplos de 90 graus (90, 180, 270)
    ToTensorV2()  # Converte para tensores PyTorch
])

# %% [markdown]
# # **Data Module (Datasets)**

# %%
class CoreDataset(Dataset):
    def __init__(self, subset: pd.DataFrame, index_mask, augmentations=None):
        subset.reset_index(drop=True, inplace=True)
        self.subset = subset
        self.index_mask = index_mask
        self.augmentations = augmentations

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, index: int):
        # Obter o caminho do arquivo a partir do DataFrame
        img_path = self.subset.iloc[index]['file_path']

        #Lê todas as bandas da imagem
        bandas = rasterio.open(img_path).read()

        #Transforma em array numpy
        bandas = np.array(bandas)

        # Assumindo que as bandas estão nos primeiros canais
        X = bandas[0:13, :, :].astype(np.float32) / 10000

        # Assumindo que o alvo está no canal 14 (index 13)
        y = bandas[self.index_mask, :, :].astype(np.int64)


        if self.augmentations:
            # Transpor a imagem de (bands, height, width) para (height, width, bands) para trabalhar com Albumentations
            augmented = self.augmentations(image=X.transpose(1, 2, 0), mask=y)
            X = augmented['image'].float()  # Convertendo para tensor e float32
            y = augmented['mask'].long()  # Convertendo a máscara para tensor long (para classificação)

        return X, y


# %%
class CoreDataModule(pl.LightningDataModule):
    def __init__(self, dataframe: pd.DataFrame, batch_size: int = 4, train_index_mask: int = 13, val_index_mask: int = 13, test_index_mask: int = 13):
        super().__init__()

        # Separar o DataFrame em datasets de treino, validação e teste
        self.train_dataset = dataframe[dataframe['set_type'] == 'train']
        self.validation_dataset = dataframe[dataframe['set_type'] == 'val']
        self.test_dataset = dataframe[dataframe['set_type'] == 'test']

        # Definir o batch_size
        self.batch_size = batch_size
        self.train_index_mask = train_index_mask
        self.val_index_mask = val_index_mask
        self.test_index_mask = test_index_mask

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=CoreDataset(self.train_dataset, index_mask=self.train_index_mask, augmentations=augmentation_pipeline),
            batch_size=self.batch_size,
            shuffle=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=CoreDataset(self.validation_dataset, index_mask=self.val_index_mask),
            batch_size=self.batch_size

        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=CoreDataset(self.test_dataset, index_mask=self.test_index_mask),
            batch_size=self.batch_size

        )

# %%
encoder_name="resnet34"
name="Unet_"+encoder_name
dir_root = '/home/mseruffo/Unet/lightning_logs/'+name
log_path = "/home/mseruffo/Unet/lightning_logs/"
batch_size = 16
EPOCHS = 100

# %% [markdown]
# # **Define o Modelo**

# %%
class UNet_S2_Br(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = smp.Unet(encoder_name=encoder_name, encoder_weights=None, classes=4, in_channels=13)
        self.loss = torch.nn.CrossEntropyLoss()

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

        accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro")
        iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
        recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro")

        metrics = {
            f"{stage}_acuracia": accuracy,
            f"{stage}_dataset_iou": iou,
            f"{stage}_f1_score": f1_score,
            f"{stage}_recall": recall
        }

        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)

        output = F.softmax(y_hat, dim=1)
        output = torch.argmax(output, dim=1)

        tp, fp, fn, tn = smp.metrics.get_stats(output, y, mode='multiclass', num_classes=4)

        self.training_step_outputs.append({"loss": loss,
                                            "tp": tp,
                                            "fp": fp,
                                            "fn": fn,
                                            "tn": tn,})

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
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

        tp, fp, fn, tn = smp.metrics.get_stats(output, y, mode='multiclass', num_classes=4)

        self.validation_step_outputs.append({"loss": loss,
                                              "tp": tp,
                                              "fp": fp,
                                              "fn": fn,
                                              "tn": tn,})

        self.log("val_loss", loss, prog_bar=True)
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

        tp, fp, fn, tn = smp.metrics.get_stats(output, y, mode='multiclass', num_classes=4)

        self.test_step_outputs.append({   "loss": loss,
                                          "tp": tp,
                                          "fp": fp,
                                          "fn": fn,
                                          "tn": tn,})

        self.log("test_loss", loss, prog_bar=True)
        return loss

    def on_test_epoch_end(self):
        self.shared_epoch_end(self.test_step_outputs, "test")
        # empty set output list
        self.test_step_outputs.clear()
        return

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        # Define o scheduler para reduzir LR se a perda de validação não diminuir por 4 épocas
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4, min_lr=1e-6, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',  # Monitora a perda de validação
                'interval': 'epoch',  # Aplica o ajuste por época
                'frequency': 1  # Frequência de monitoramento (1 = a cada época)
            }
        }

# %% [markdown]
# # **Define e executa o treinamento com as imagens NoLabel**

# %%
checkpoint_callback = pl.callbacks.ModelCheckpoint(
                                                    dirpath=dir_root,
                                                    filename='{epoch}-{train_loss:.2f}-{val_loss:.2f}-trainNoLabel',
                                                    monitor="val_loss",
                                                    mode="min",
                                                    save_top_k=2
                                                  )

earlystopping_callback = pl.callbacks.EarlyStopping(
                                            monitor="val_loss",
                                            patience=10,
                                            mode="min"
                                          )
# Define the callbacks
callbacks = [checkpoint_callback, earlystopping_callback]


tb_logger = TensorBoardLogger(log_path, name=name)

# Define the trainer
trainer = pl.Trainer(
    max_epochs=EPOCHS,
    callbacks=callbacks,
    log_every_n_steps=1,
    accelerator="auto",
    precision="16-mixed",
    logger=tb_logger,
    default_root_dir=dir_root
)

train_dataloader = torch.utils.data.DataLoader(
                  dataset=CoreDataset(df_nolabel[df_nolabel['set_type'] == 'train'], index_mask=14),
                  batch_size=batch_size

                                              )
val_dataloader = torch.utils.data.DataLoader(
                  dataset=CoreDataset(df_nolabel[df_nolabel['set_type'] == 'val'], index_mask=14),
                  batch_size=batch_size

                                              )

# Define the model
model = UNet_S2_Br()

#Start the training
trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

# %% [markdown]
# ## **Recupera o melhor modelo baseado no checkpoint**

# %%
# Carregar o melhor modelo diretamente
model = UNet_S2_Br.load_from_checkpoint(checkpoint_callback.best_model_path)

# %% [markdown]
# # **Define e executa o treinamento com as imagens Scribble**

# %%
checkpoint_callback = pl.callbacks.ModelCheckpoint(
                                                    dirpath=dir_root,
                                                    filename='{epoch}-{train_loss:.2f}-{val_loss:.2f}-trainScribble',
                                                    monitor="val_loss",
                                                    mode="min",
                                                    save_top_k=2
                                                  )

earlystopping_callback = pl.callbacks.EarlyStopping(
                                            monitor="val_loss",
                                            patience=10,
                                            mode="min"
                                          )
# Define the callbacks
callbacks = [checkpoint_callback, earlystopping_callback]


tb_logger = TensorBoardLogger(log_path, name=name)

# Define the trainer
trainer = pl.Trainer(
    max_epochs=EPOCHS,
    callbacks=callbacks,
    log_every_n_steps=1,
    accelerator="auto",
    precision="16-mixed",
    logger=tb_logger,
    default_root_dir=dir_root
)

# Define the model
model = UNet_S2_Br()

# Define the datamodule
datamodule = CoreDataModule(dataframe=df_scribble, batch_size=batch_size, train_index_mask=14, val_index_mask=14, test_index_mask=14)

#Start the training
trainer.fit(model=model, datamodule=datamodule)

# %% [markdown]
# ## **Recupera o melhor modelo baseado no checkpoint**

# %%
# Carregar o melhor modelo diretamente
model = UNet_S2_Br.load_from_checkpoint(checkpoint_callback.best_model_path)

# %%
# run val dataset
val_metrics = trainer.validate(model, datamodule=datamodule, verbose=True)
print(val_metrics)

# %%
# run test dataset
test_metrics = trainer.test(model, datamodule=datamodule, verbose=True)
print(test_metrics)

# %% [markdown]
# # **Define e executa o treinamento com as imagens HIGH**

# %%
# Define the datamodule
datamodule = CoreDataModule(dataframe=df, batch_size=batch_size)


checkpoint_callback = pl.callbacks.ModelCheckpoint(
                                                    dirpath=dir_root,
                                                    filename='{epoch}-{train_loss:.2f}-{val_loss:.2f}-trainHigh',
                                                    monitor="val_loss",
                                                    mode="min",
                                                    save_top_k=2
                                                  )

earlystopping_callback = pl.callbacks.EarlyStopping(
                                            monitor="val_loss",
                                            patience=10,
                                            mode="min"
                                          )
# Define the callbacks
callbacks = [checkpoint_callback, earlystopping_callback]


tb_logger = TensorBoardLogger(log_path, name=name)

# Define the trainer
trainer = pl.Trainer(
    max_epochs=EPOCHS,
    callbacks=callbacks,
    log_every_n_steps=1,
    accelerator="auto",
    precision="16-mixed",
    logger=tb_logger,
    default_root_dir=dir_root
)

#Start the training
trainer.fit(model=model, datamodule=datamodule)

# %% [markdown]
# ## **Recupera o melhor modelo baseado no checkpoint**

# %%
#Carregar o melhor modelo diretamente
model = UNet_S2_Br.load_from_checkpoint(checkpoint_callback.best_model_path)

# %% [markdown]
# ## **Executa a validação**

# %%
# run val dataset
val_metrics = trainer.validate(model, datamodule=datamodule, verbose=True)
print(val_metrics)

# %% [markdown]
# ## **Executa o test**

# %%
# run test dataset
test_metrics = trainer.test(model, datamodule=datamodule, verbose=True)
print(test_metrics)

# %% [markdown]
# ## **Executa o test para cálculo das métricas finais**

# %%
steps_outputs_metrics = []

for images, gt_masks in datamodule.test_dataloader():
    with torch.no_grad():
        model.eval()
        logits = model.model(images)
    #pr_masks = logits.sigmoid()
    pr_masks = F.softmax(logits, dim=1)
    pr_masks = torch.argmax(pr_masks, dim=1)

    # first compute statistics for true positives, false positives, false negative and
    # true negative "pixels"
    tp, fp, fn, tn = smp.metrics.get_stats(gt_masks, pr_masks, mode='multiclass', num_classes=4)
    steps_outputs_metrics.append({  "tp": tp,
                                    "fp": fp,
                                    "fn": fn,
                                    "tn": tn})

tp = torch.cat([x["tp"] for x in steps_outputs_metrics])
fp = torch.cat([x["fp"] for x in steps_outputs_metrics])
fn = torch.cat([x["fn"] for x in steps_outputs_metrics])
tn = torch.cat([x["tn"] for x in steps_outputs_metrics])

acuracia = smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro-imagewise")
acuracia_balanceada = smp.metrics.balanced_accuracy(tp, fp, fn, tn, reduction="micro-imagewise")
iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro-imagewise")
f2_score = smp.metrics.fbeta_score(tp, fp, fn, tn, beta=2, reduction="micro-imagewise")
recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro-imagewise")

print(f'Acurácia no conjunto de teste: {acuracia:.4f}')
print(f'Acurácia Balanceada no conjunto de teste: {acuracia_balanceada:.4f}')
print(f'IoU no conjunto de teste: {iou:.4f}')
print(f'F1 no conjunto de teste: {f1_score:.4f}')
print(f'F2 no conjunto de teste: {f2_score:.4f}')
print(f'Recall no conjunto de teste: {recall:.4f}')


# %% [markdown]
# # **Salva o modelo treinado**

# %%
smp_model = model.model
# if push_to_hub=True, model will be saved to repository with this name
smp_model.save_pretrained(dir_root + name)


