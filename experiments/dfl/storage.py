import pandas as pd
import pathlib


def append(directory: str, name: str, dataframe: pd.DataFrame):
    if directory:
        data_dir = pathlib.Path(directory)

        if data_dir.is_dir():
            csv_file = data_dir / name

            if csv_file.is_file():
                df = pd.read_csv(csv_file)
                udf = pd.concat([df, dataframe], axis=0)
                udf.to_csv(csv_file, index=False)
            else:
                dataframe.to_csv(csv_file, index=False)
        else:
            print(f"{data_dir} is not a valid directory.. stopping save of dataframe.")
    else:
        print("Need to set directory.")


def read(directory: str, name: str) -> pd.DataFrame:
    if directory:
        data_dir = pathlib.Path(directory)

        if data_dir.is_dir():
            csv_file = data_dir / name

            if csv_file.is_file():
                df = pd.read_csv(csv_file)
                return df
            else:
                raise AssertionError(f"{csv_file} is not a file.")
        else:
            raise AssertionError(f"{data_dir} is not a valid directory.")
    else:
        raise AssertionError(f"Need to set a directory.")
