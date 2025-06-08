import pandas as pd
import numpy as np
import pathlib

IN_PARQUET = pathlib.Path("data/40_preprocessed_classification/articles.parquet")
OUT_PARQUET = pathlib.Path("data/50_cluster/articles.parquet")

def main():
    df = pd.read_parquet(IN_PARQUET)

    

if __name__ == "__main__":
    main()
