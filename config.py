# Encoders disponíveis no Segmentation Models Pytorch
ENCODER_NAME_MOBILENET = "mobilenet_v2"
ENCODER_NAME_RESNET34 = "resnet34"
ENCODER_NAME_RESNET50 = "resnet50"
ENCODER_NAME_EFFICIENTNETB0 = "efficientnet-b0"
ENCODER_NAME_EFFICIENTNETB2 = "efficientnet-b2"
ENCODER_NAME_EFFICIENTNETB3 = "efficientnet-b3"

# Nomes das pastas dos modelos
NAME_MOBILENET = "Unet_mobilenet_v2"
NAME_RESNET34 = "Unet_resnet34"
NAME_RESNET50 = "Unet_resnet50"
NAME_EFFICIENTNETB0 = "Unet_efficientnet-b0"
NAME_EFFICIENTNETB2 = "Unet_efficientnet-b2"
NAME_EFFICIENTNETB3 = "Unet_efficientnet-b3"

# Hiperparâmetros do modelo
LEARNING_RATE = 0.001
EPOCHS = 100
BATCH_SIZE_512 = 16
BATCH_SIZE_2048 = 1
CLASSES = 4
IN_CHANNELS = 13
ACCELERATOR = "auto"

# Diretórios
DIR_BASE = "/home/mseruffo/"
DIR_BASE2 = "/home/mseruffo/scratch/"
DIR_LOG = "/home/mseruffo/Unet/lightning_logs/"

# Diretórios raiz dos modelos
DIR_ROOT_MOBILENET = "/home/mseruffo/Unet/lightning_logs/Unet_mobilenet_v2"
DIR_ROOT_RESNET34 = "/home/mseruffo/Unet/lightning_logs/Unet_resnet34"
DIR_ROOT_RESNET50 = "/home/mseruffo/Unet/lightning_logs/Unet_resnet50"
DIR_ROOT_EFFICIENTNETB0 = "/home/mseruffo/Unet/lightning_logs/Unet_efficientnet-b0"
DIR_ROOT_EFFICIENTNETB2 = "/home/mseruffo/Unet/lightning_logs/Unet_efficientnet-b2"
DIR_ROOT_EFFICIENTNETB3 = "/home/mseruffo/Unet/lightning_logs/Unet_efficientnet-b3"
