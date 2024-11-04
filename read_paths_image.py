import pandas as pd
import glob


def get_image_paths(image_type, diretorio_base, proj_shape):

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
        
        if proj_shape == 512:
            # Listar arquivos de treino, validação e teste para as imagens de 512x512
            train_files_512 = glob.glob(
                diretorio_base + "CloudSen12High_512/train/*/*/*.tif"
            )
            val_files_512 = glob.glob(
                diretorio_base + "CloudSen12High_512/val/*/*/*.tif"
            )
            test_files_512 = glob.glob(
                diretorio_base + "CloudSen12High_512/test/*/*/*.tif"
            )

            # Concatenando os conjuntos de treino, validação e teste
            train_files = train_files_512
            val_files = val_files_512 
            test_files = test_files_512
            
            # Criar DataFrames e adicionar a coluna 'set_type'
            train_df = pd.DataFrame(train_files, columns=["file_path"])
            train_df["set_type"] = "train"

            val_df = pd.DataFrame(val_files, columns=["file_path"])
            val_df["set_type"] = "val"

            test_df = pd.DataFrame(test_files, columns=["file_path"])
            test_df["set_type"] = "test"
            df = pd.concat([train_df, val_df, test_df], ignore_index=True)
            
        elif proj_shape == 2048:
            # Listar arquivos de treino, validação e teste para as imagens de 2048x2048
            train_files_2048 = glob.glob(
                diretorio_base + "CloudSen12High_2000/train/*/*.tif"
            )
            val_files_2048 = glob.glob(
                diretorio_base + "CloudSen12High_2000/val/*.tif"
            )
            test_files_2048 = glob.glob(
                diretorio_base + "CloudSen12High_2000/test/*.tif"
            )

            # Concatenando os conjuntos de treino, validação e teste
            train_files = train_files_2048
            val_files = val_files_2048
            test_files = test_files_2048

            # Criar DataFrames e adicionar a coluna 'set_type'
            train_df = pd.DataFrame(train_files, columns=["file_path"])
            train_df["set_type"] = "train"

            val_df = pd.DataFrame(val_files, columns=["file_path"])
            val_df["set_type"] = "val"

            test_df = pd.DataFrame(test_files, columns=["file_path"])
            test_df["set_type"] = "test"

            # Concatenar todos os DataFrames
            df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    print(f"Total de imagens: {len(df)}")
    return df
