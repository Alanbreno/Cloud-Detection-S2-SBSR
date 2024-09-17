# %% [markdown]
# <a href="https://colab.research.google.com/drive/1m8LUoa1n7SDC6N5eqCGOTcC-nwPQfIoT"><img align="left" src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab" title="Open and Execute in Google Colaboratory"></a>
# 
# <br>
# 
# <a href="https://cloudsen12.github.io/"><img align="left" src="https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white"></a>
# 
# 
# <br><br>
# 
# <!--COURSE_INFORMATION-->
# <img align="left" style="padding-right:10px;" src="https://cloudsen12plus.github.io/assets/logo.webp" width=10% >
# 
# 
# >>>> *This notebook is part of the paper [CloudSEN12+: The largest dataset of expert-labeled pixels for cloud and cloud shadow detection in Sentinel-2](https://cloudsen12.github.io/); the content is available [on GitHub](https://github.com/cloudsen12)* and released under the [CC0 1.0 Universal - Creative Commons](https://creativecommons.org/publicdomain/zero/1.0/deed.en) license.
# 
# <br>
# 
# - See our paper [here](https://www.sciencedirect.com/science/article/pii/S2352340924008163).
# 
# - See cloudSEN12 website [here](https://cloudsen12.github.io/).
# 
# - See cloudSEN12 in Science Data Bank [here](https://www.scidb.cn/en/detail?dataSetId=2036f4657b094edfbb099053d6024b08&version=V1).
# 
# 
# - See cloudSEN12 in GitHub [here](https://github.com/cloudsen12).
# 
# - See cloudSEN12 in Google Earth Engine [here](https://samapriya.github.io/awesome-gee-community-datasets/projects/cloudsen12/).
# 
# - See CloudApp [here](https://cloudsen12.github.io/en/blog/cloudapp/).
# 
# The CloudSEN12 dataset and the pre-trained models are released under a [CC0 1.0 Universal - Creative Commons](https://creativecommons.org/publicdomain/zero/1.0/deed.en) license.

# %% [markdown]
# ## **1. Create a DataModule**
# 
# *Using the streaming support of MLS STAC __will slow down the DataLoader__ \_\_getitem\_\_. For better I/O performance, consider downloading the data first.*

# %%
import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import mlstac

# Create a DataLoader object.
class CoreDataset(torch.utils.data.DataLoader):
    def __init__(self, subset:pd.DataFrame):
        subset.reset_index(drop=True, inplace=True)
        self.subset = subset

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, index: int):
        # Retrieve the data from HuggingFace
        sample = mlstac.get_data(dataset=self.subset.iloc[index], quiet=False).squeeze()

        # Load the Sentinel-2 all bands
        # We set <0:32> to make it faster and run in CPU
        X = sample[0:13, :, :].astype(np.float32) / 10000

        # Load the target
        y = sample[13, :, :].astype(np.int64)

        return X, y

class CoreDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 4):
        super().__init__()

        # Load the metadata from the MLSTAC Collection file
        metadata = mlstac.load(snippet="isp-uv-es/CloudSEN12Plus").metadata

        # Split the metadata into train, validation and test sets
        self.train_dataset = metadata[(metadata["split"] == "train") & (metadata["label_type"] == "high") & (metadata["proj_shape"] == 509)]
        self.validation_dataset = metadata[(metadata["split"] == "validation") & (metadata["label_type"] == "high") & (metadata["proj_shape"] == 509)]
        self.test_dataset = metadata[(metadata["split"] == "test") & (metadata["label_type"] == "high") & (metadata["proj_shape"] == 509)]

        # Define the batch_size
        self.batch_size = batch_size

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=CoreDataset(self.train_dataset),
            batch_size=self.batch_size,
            shuffle=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=CoreDataset(self.validation_dataset),
            batch_size=self.batch_size
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=CoreDataset(self.test_dataset),
            batch_size=self.batch_size
        )

# %% [markdown]
# ## **2. Define a Model**

# %%
import segmentation_models_pytorch as smp
import torch

class litmodel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = smp.Unet(encoder_name="mobilenet_v2", encoder_weights=None, classes=4, in_channels=13)
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

# %% [markdown]
# ## **3. Define the Trainer**

# %%
# Define the callbacks
callbacks = [
    pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=3
    ),
    pl.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        mode="min"
    )
]

# Define the trainer
trainer = pl.Trainer(
    max_epochs=100,
    callbacks=callbacks,
    accelerator="auto",
    precision="16-mixed"
)

# Define the datamodule
datamodule = CoreDataModule(batch_size=16)

# Define the model
model = litmodel()

# Start the training
trainer.fit(model=model, datamodule=datamodule)

# %%
# run validation dataset
valid_metrics = trainer.validate(model, datamodule=datamodule, verbose=True)
print(valid_metrics)

# %%
# run test dataset
test_metrics = trainer.test(model, datamodule=datamodule, verbose=True)
print(test_metrics)

# %%
smp_model = model.model
# if push_to_hub=True, model will be saved to repository with this name
smp_model.save_pretrained('./unet_test')


