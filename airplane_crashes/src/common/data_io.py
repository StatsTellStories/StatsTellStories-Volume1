from pathlib import Path
import pandas as pd
# import polars as pl


def export_dict_to_txt(list_years: dict, output_dir: str):
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    for year, dates in list_years.items():
        file_path = path / f"{year}.txt"

        with file_path.open("w") as f:
            for d in dates:
                f.write(d + "\n")

    print("Export successful.")

def export_df(df, directory, filename, index=False, header=True, **kwargs):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    suffix = path.suffix.lower()

    if suffix == ".csv":
        df.to_csv(path, index=index, header=header, **kwargs)
    elif suffix == ".parquet":
        df.to_parquet(path, index=index, **kwargs)  # kein header hier
    elif suffix in [".xlsx", ".xls"]:
        df.to_excel(path, index=index, header=header, **kwargs)
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Use .csv, .parquet, or .xlsx")


def import_df(directory, filename, **kwargs):
    """
    Import a pandas DataFrame from a directory and filename.
    Supported formats: .csv, .parquet, .xlsx
    """

    path = Path(directory) / filename

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path, **kwargs)
    elif suffix == ".parquet":
        return pd.read_parquet(path, **kwargs)
    elif suffix in [".xlsx", ".xls"]:
        return pd.read_excel(path, **kwargs)
    else:
        raise ValueError(
            f"Unsupported file type: {suffix}. "
            "Use .csv, .parquet, or .xlsx"
        )
