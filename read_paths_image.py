import pandas as pd
import glob


def get_image_paths(image_type, diretorio_base):

    if image_type == "nolabel":
        # Listar arquivos de treino, validação e teste para as imagens de 512x512
        train_files_512_nolabel = glob.glob(
            diretorio_base + "CloudSen12_Br_Resized/p509/nolabel/train_br/*/*.tif"
        )
        val_files_512_nolabel = glob.glob(
            diretorio_base + "CloudSen12_Br_Resized/p509/nolabel/val_br/*/*.tif"
        )

        # Criar DataFrames e adicionar a coluna 'set_type'
        train_df = pd.DataFrame(train_files_512_nolabel, columns=["file_path"])
        train_df["set_type"] = "train"
        val_df = pd.DataFrame(val_files_512_nolabel, columns=["file_path"])
        val_df["set_type"] = "val"

        # Concatenar todos os DataFrames
        df = pd.concat([train_df, val_df], ignore_index=True)

    elif image_type == "scribble":
        # Listar arquivos de treino, validação e teste para as imagens de 512x512
        train_files_512_scribble = glob.glob(
            diretorio_base + "CloudSen12_Br_Resized/p509/scribble/train_br/*/*.tif"
        )
        val_files_512_scribble = glob.glob(
            diretorio_base + "CloudSen12_Br_Resized/p509/scribble/val_br/*/*.tif"
        )
        test_files_512_scribble = glob.glob(
            diretorio_base + "CloudSen12_Br_Resized/p509/scribble/test_br/*/*.tif"
        )

        # Criar DataFrames e adicionar a coluna 'set_type'
        train_df = pd.DataFrame(train_files_512_scribble, columns=["file_path"])
        train_df["set_type"] = "train"

        val_df = pd.DataFrame(val_files_512_scribble, columns=["file_path"])
        val_df["set_type"] = "val"

        test_df = pd.DataFrame(test_files_512_scribble, columns=["file_path"])
        test_df["set_type"] = "test"

        # Concatenar todos os DataFrames
        df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    elif image_type == "high":
        # Listar arquivos de treino, validação e teste para as imagens de 512x512
        train_files_512 = glob.glob(
            diretorio_base + "CloudSen12_Br_Resized/p509/high/train_br/*/*.tif"
        )
        val_files_512 = glob.glob(
            diretorio_base + "CloudSen12_Br_Resized/p509/high/val_br/*/*.tif"
        )
        test_files_512 = glob.glob(
            diretorio_base + "CloudSen12_Br_Resized/p509/high/test_br/*/*.tif"
        )

        # Listar arquivos de treino, validação e teste para as imagens de 2048x2048
        train_files_2048 = glob.glob(
            diretorio_base + "CloudSen12_Br_Resized/p2000/train_br/*/*.tif"
        )
        val_files_2048 = glob.glob(
            diretorio_base + "CloudSen12_Br_Resized/p2000/val_br/*/*.tif"
        )
        test_files_2048 = glob.glob(
            diretorio_base + "CloudSen12_Br_Resized/p2000/test_br/*/*.tif"
        )

        # Concatenando os conjuntos de treino, validação e teste
        train_files = train_files_512 + train_files_2048
        val_files = val_files_512 + val_files_2048
        test_files = test_files_512 + test_files_2048

        # Criar DataFrames e adicionar a coluna 'set_type'
        train_df = pd.DataFrame(train_files, columns=["file_path"])
        train_df["set_type"] = "train"

        val_df = pd.DataFrame(val_files, columns=["file_path"])
        val_df["set_type"] = "val"

        test_df = pd.DataFrame(test_files, columns=["file_path"])
        test_df["set_type"] = "test"

        # Concatenar todos os DataFrames
        df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    return df
