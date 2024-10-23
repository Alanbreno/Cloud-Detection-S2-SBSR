import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from datamodule import CoreDataModule
from model import UNet_S2_Br
import config
import read_paths_image as rpi
import metrics

# Para usar outro modelo de encoder, basta alterar o valor da variável ENCODER_NAME_<Modelo>, NAME_<Modelo>, e DIR_ROOT_<Modelo>.
# <Modelo> pode ser: MOBILENET, RESNET34, RESNET50, EFICIENTNETB0, EFICIENTNETB2, EFICIENTNETB3
# Para usar modelos diferentes, basta adicionar as variáveis correspondentes no arquivo config.py

# Primeiro Treinamento com as imagens NoLabel
tb_logger = TensorBoardLogger(config.DIR_LOG, name=config.NAME_MOBILENET)

# Gera o dataframe com as imagem NoLabel
df_nolabel = rpi.get_image_paths("nolabel", config.DIR_BASE)

# Define the datamodule
datamodule = CoreDataModule(
    dataframe=df_nolabel,
    batch_size=config.BATCH_SIZE,
    train_index_mask=14,
    val_index_mask=14,
    test_index_mask=14,
)

# Define the model
model = UNet_S2_Br(
    encoder_name=config.ENCODER_NAME_MOBILENET,
    classes=config.CLASSES,
    in_channels=config.IN_CHANNELS,
    learning_rate=config.LEARNING_RATE,
)

checkpoint_callback = pl.callbacks.ModelCheckpoint(
    dirpath=config.DIR_ROOT_MOBILENET,
    filename="{epoch}-{train_loss:.2f}-{val_loss:.2f}-trainNoLabel",
    monitor="val_loss",
    mode="min",
    save_top_k=2,
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
    default_root_dir=config.DIR_ROOT_MOBILENET,
)

# Start the training
trainer.fit(model=model, datamodule=datamodule)
# Carregar o melhor modelo diretamente
model = UNet_S2_Br.load_from_checkpoint(checkpoint_callback.best_model_path)


# Executa o segundo treinamento com as imagens Scribble
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    dirpath=config.DIR_ROOT_MOBILENET,
    filename="{epoch}-{train_loss:.2f}-{val_loss:.2f}-trainScribble",
    monitor="val_loss",
    mode="min",
    save_top_k=2,
)

earlystopping_callback = pl.callbacks.EarlyStopping(
    monitor="val_loss", patience=10, mode="min"
)
# Define the callbacks
callbacks = [checkpoint_callback, earlystopping_callback]


tb_logger = TensorBoardLogger(config.DIR_LOG, name=config.NAME_MOBILENET)

# Define the trainer
trainer = pl.Trainer(
    max_epochs=config.EPOCHS,
    callbacks=callbacks,
    log_every_n_steps=1,
    accelerator=config.ACCELERATOR,
    precision="16-mixed",
    logger=tb_logger,
    default_root_dir=config.DIR_ROOT_MOBILENET,
)

df_scribble = rpi.get_image_paths("scribble", config.DIR_BASE)

# Define the datamodule
datamodule = CoreDataModule(
    dataframe=df_scribble,
    batch_size=config.BATCH_SIZE,
    train_index_mask=14,
    val_index_mask=14,
    test_index_mask=14,
)

# Start the training
trainer.fit(model=model, datamodule=datamodule)

# Carregar o melhor modelo diretamente
model = UNet_S2_Br.load_from_checkpoint(checkpoint_callback.best_model_path)

# run val dataset
val_metrics = trainer.validate(model, datamodule=datamodule, verbose=True)
print(val_metrics)

# run test dataset
test_metrics = trainer.test(model, datamodule=datamodule, verbose=True)
print(test_metrics)

# Executa o treinamento com as imagens HIGH
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    dirpath=config.DIR_ROOT_MOBILENET,
    filename="{epoch}-{train_loss:.2f}-{val_loss:.2f}-trainHigh",
    monitor="val_loss",
    mode="min",
    save_top_k=2,
)

earlystopping_callback = pl.callbacks.EarlyStopping(
    monitor="val_loss", patience=10, mode="min"
)
# Define the callbacks
callbacks = [checkpoint_callback, earlystopping_callback]

tb_logger = TensorBoardLogger(config.DIR_LOG, name=config.NAME_MOBILENET)

df = rpi.get_image_paths("high", config.DIR_BASE)

# Define the datamodule
datamodule = CoreDataModule(dataframe=df, batch_size=config.BATCH_SIZE)

# Define the trainer
trainer = pl.Trainer(
    max_epochs=config.EPOCHS,
    callbacks=callbacks,
    log_every_n_steps=1,
    accelerator=config.ACCELERATOR,
    precision="16-mixed",
    logger=tb_logger,
    default_root_dir=config.DIR_ROOT_MOBILENET,
)

# Start the training
trainer.fit(model=model, datamodule=datamodule)

# Carregar o melhor modelo diretamente
model = UNet_S2_Br.load_from_checkpoint(checkpoint_callback.best_model_path)

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
smp_model.save_pretrained(config.DIR_ROOT_MOBILENET + "/" + config.NAME_MOBILENET)
