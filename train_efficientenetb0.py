import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from datamodule import CoreDataModule
from model import UNet_S2_Br
import config
import read_paths_image as rpi
import metrics

# Para usar outro modelo de encoder, basta alterar o valor da variável ENCODER_NAME_<Modelo>, NAME_<Modelo>, e DIR_ROOT_<Modelo>.
# <Modelo> pode ser: MOBILENET, RESNET34, RESNET50, EFFICIENTNETB0, EFFICIENTNETB2, EFFICIENTNETB3
# Para usar modelos diferentes, basta adicionar as variáveis correspondentes no arquivo config.py

# Primeiro Treinamento com as imagens 512x512
tb_logger = TensorBoardLogger(config.DIR_LOG, name=config.NAME_EFFICIENTNETB0)

# Gera o dataframe com as imagens 512x512 e 2048x2048
df_512 = rpi.get_image_paths("high", config.DIR_BASE, proj_shape=512)
df_2048 = rpi.get_image_paths("high", config.DIR_BASE2, proj_shape=2048)

# Define the datamodule
datamodule = CoreDataModule(
    dataframe=df_512,
    batch_size=config.BATCH_SIZE_512
)

# Define the model
model = UNet_S2_Br(
    encoder_name=config.ENCODER_NAME_EFFICIENTNETB0,
    classes=config.CLASSES,
    in_channels=config.IN_CHANNELS,
    learning_rate=config.LEARNING_RATE,
)

checkpoint_callback = pl.callbacks.ModelCheckpoint(
    dirpath=config.DIR_ROOT_EFFICIENTNETB0,
    filename="{epoch}-{train_loss:.2f}-{val_loss:.2f}-trainHigh512",
    monitor="val_loss",
    mode="min",
    save_top_k=1,
)

earlystopping_callback = pl.callbacks.EarlyStopping(
    monitor="val_loss", patience=10, mode="min"
)

callbacks = [checkpoint_callback, earlystopping_callback]

# Define the trainer
trainer = pl.Trainer(
    max_epochs=config.EPOCHS,
    log_every_n_steps=1,
    callbacks=callbacks,
    accelerator=config.ACCELERATOR,
    precision="16-mixed",
    logger=tb_logger,
    default_root_dir=config.DIR_ROOT_EFFICIENTNETB0,
)

# Start the training
#trainer.fit(model=model, datamodule=datamodule)
# Carregar o melhor modelo diretamente
model = UNet_S2_Br.load_from_checkpoint(
    "/home/mseruffo/Unet/lightning_logs/Unet_efficientnet-b0/epoch=48-train_loss=0.22-val_loss=0.24-trainHigh512.ckpt",
    encoder_name=config.ENCODER_NAME_EFFICIENTNETB0,
    classes=config.CLASSES,
    in_channels=config.IN_CHANNELS,
    learning_rate=config.LEARNING_RATE,
)

# run val dataset
val_metrics = trainer.validate(model, datamodule=datamodule, verbose=True)
print(val_metrics)

# run test dataset
test_metrics = trainer.test(model, datamodule=datamodule, verbose=True)
print(test_metrics)

acuracia, acuracia_balanceada, iou, f1_score, f2_score, recall = metrics.calculate_metrics(datamodule.test_dataloader(), model.model)

# Executa o treinamento com as imagens HIGH 2048x2048
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    dirpath=config.DIR_ROOT_EFFICIENTNETB0,
    filename="{epoch}-{train_loss:.2f}-{val_loss:.2f}-trainHigh2048",
    monitor="val_loss",
    mode="min",
    save_top_k=1,
)

earlystopping_callback = pl.callbacks.EarlyStopping(
    monitor="val_loss", patience=10, mode="min"
)
# Define the callbacks
callbacks = [checkpoint_callback, earlystopping_callback]

tb_logger = TensorBoardLogger(config.DIR_LOG, name=config.NAME_EFFICIENTNETB0)

df_2048 = rpi.get_image_paths("high", config.DIR_BASE2, proj_shape=2048)

# Define the datamodule
datamodule = CoreDataModule(dataframe=df_2048, batch_size=config.BATCH_SIZE_2048)

# Define the trainer
trainer = pl.Trainer(
    max_epochs=config.EPOCHS,
    callbacks=callbacks,
    log_every_n_steps=1,
    accelerator=config.ACCELERATOR,
    precision="16-mixed",
    logger=tb_logger,
    default_root_dir=config.DIR_ROOT_EFFICIENTNETB0,
)

# Start the training
trainer.fit(model=model, datamodule=datamodule)

# Carregar o melhor modelo diretamente
model = UNet_S2_Br.load_from_checkpoint(
    checkpoint_callback.best_model_path,
    encoder_name=config.ENCODER_NAME_EFFICIENTNETB0,
    classes=config.CLASSES,
    in_channels=config.IN_CHANNELS,
    learning_rate=config.LEARNING_RATE,
)

# run val dataset
val_metrics = trainer.validate(model, datamodule=datamodule, verbose=True)
print(val_metrics)

# run test dataset
test_metrics = trainer.test(model, datamodule=datamodule, verbose=True)
print(test_metrics)

# Calcula as métricas de teste
acuracia, acuracia_balanceada, iou, f1_score, f2_score, recall = metrics.calculate_metrics(datamodule.test_dataloader(), model.model)

# Salva o modelo treinado
smp_model = model.model
smp_model.save_pretrained(config.DIR_ROOT_EFFICIENTNETB0 + "/" + config.NAME_EFFICIENTNETB0)
