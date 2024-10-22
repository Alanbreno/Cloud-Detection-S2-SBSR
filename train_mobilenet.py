import torch
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch.nn.functional as F
from pytorch_lightning.loggers import TensorBoardLogger
from datamodule import CoreDataModule
from dataset import CoreDataset
from model import UNet_S2_Br
import config
import read_paths_image as rpi

# Primeiro Treinamento com as imagens NoLabel
tb_logger = TensorBoardLogger(config.DIR_LOG, name=config.NAME_MOBILENET)

df_nolabel = rpi.get_image_paths("nolabel", config.DIR_BASE)

train_dataloader = torch.utils.data.DataLoader(
    dataset=CoreDataset(df_nolabel[df_nolabel["set_type"] == "train"], index_mask=14),
    batch_size=config.BATCH_SIZE,
)
val_dataloader = torch.utils.data.DataLoader(
    dataset=CoreDataset(df_nolabel[df_nolabel["set_type"] == "val"], index_mask=14),
    batch_size=config.BATCH_SIZE,
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
trainer.fit(
    model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
)
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

# Executa o teste para registro das métricas
steps_outputs_metrics = []

for images, gt_masks in datamodule.test_dataloader():
    with torch.no_grad():
        model.eval()
        logits = model.model(images)
    # pr_masks = logits.sigmoid()
    pr_masks = F.softmax(logits, dim=1)
    pr_masks = torch.argmax(pr_masks, dim=1)

    # first compute statistics for true positives, false positives, false negative and
    # true negative "pixels"
    tp, fp, fn, tn = smp.metrics.get_stats(
        gt_masks, pr_masks, mode="multiclass", num_classes=4
    )
    steps_outputs_metrics.append({"tp": tp, "fp": fp, "fn": fn, "tn": tn})

tp = torch.cat([x["tp"] for x in steps_outputs_metrics])
fp = torch.cat([x["fp"] for x in steps_outputs_metrics])
fn = torch.cat([x["fn"] for x in steps_outputs_metrics])
tn = torch.cat([x["tn"] for x in steps_outputs_metrics])

acuracia = smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro-imagewise")
acuracia_balanceada = smp.metrics.balanced_accuracy(
    tp, fp, fn, tn, reduction="micro-imagewise"
)
iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro-imagewise")
f2_score = smp.metrics.fbeta_score(tp, fp, fn, tn, beta=2, reduction="micro-imagewise")
recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro-imagewise")

print(f"Acurácia no conjunto de teste: {acuracia:.4f}")
print(f"Acurácia Balanceada no conjunto de teste: {acuracia_balanceada:.4f}")
print(f"IoU no conjunto de teste: {iou:.4f}")
print(f"F1 no conjunto de teste: {f1_score:.4f}")
print(f"F2 no conjunto de teste: {f2_score:.4f}")
print(f"Recall no conjunto de teste: {recall:.4f}")

# Salva o modelo treinado
smp_model = model.model
smp_model.save_pretrained(config.DIR_ROOT_MOBILENET + "/" + config.NAME_MOBILENET)
