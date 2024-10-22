# Encoders disponíveis no Segmentation Models Pytorch
ENCODER_NAME_MOBILENET = "mobilenet_v2"
ENCODER_NAME_RESNET34 = "resnet34"
ENCODER_NAME_RESNET50 = "resnet50"
ENCODER_NAME_EFICIENTNETB0 = "efficientnet-b0"
ENCODER_NAME_EFICIENTNETB2 = "efficientnet-b2"
ENCODER_NAME_EFICIENTNETB3 = "efficientnet-b3"

NAME_MOBILENET = "Unet_mobilenet_v2"
NAME_RESNET34 = "Unet_resnet34"
NAME_RESNET50 = "Unet_resnet50"
NAME_EFICIENTNETB0 = "Unet_efficientnet-b0"
NAME_EFICIENTNETB2 = "Unet_efficientnet-b2"
NAME_EFICIENTNETB3 = "Unet_efficientnet-b3"

# hiperparâmetros do modelo
LEARNING_RATE = 0.001
EPOCHS = 100
BATCH_SIZE = 16
CLASSES = 4
IN_CHANNELS = 13
ACCELERATOR = "AUTO"

DIR_BASE = "/home/mseruffo/"
DIR_LOG = "/home/mseruffo/Unet/lightning_logs/"
DIR_ROOT_MOBILENET = "/home/mseruffo/Unet/lightning_logs/Unet_mobilenet_v2"
DIR_ROOT_RESNET34 = "/home/mseruffo/Unet/lightning_logs/Unet_resnet34"
DIR_ROOT_RESNET50 = "/home/mseruffo/Unet/lightning_logs/Unet_resnet50"
DIR_ROOT_EFICIENTNETB0 = "/home/mseruffo/Unet/lightning_logs/Unet_efficientnet-b0"
DIR_ROOT_EFICIENTNETB2 = "/home/mseruffo/Unet/lightning_logs/Unet_efficientnet-b2"
DIR_ROOT_EFICIENTNETB3 = "/home/mseruffo/Unet/lightning_logs/Unet_efficientnet-b3"
