import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import rasterio


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
        img_path = self.subset.iloc[index]["file_path"]

        # Lê todas as bandas da imagem
        bandas = rasterio.open(img_path).read()

        # Transforma em array numpy
        bandas = np.array(bandas)

        # Assumindo que as bandas estão nos primeiros canais
        X = bandas[0:13, :, :].astype(np.float32) / 10000

        # Assumindo que o alvo está no canal 14 (index 13)
        y = bandas[self.index_mask, :, :].astype(np.int64)

        if self.augmentations:
            # Transpor a imagem de (bands, height, width) para (height, width, bands) para trabalhar com Albumentations
            augmented = self.augmentations(image=X.transpose(1, 2, 0), mask=y)
            X = augmented["image"].float()  # Convertendo para tensor e float32
            y = augmented[
                "mask"
            ].long()  # Convertendo a máscara para tensor long (para classificação)

        return X, y
