import pandas as pd
import pathlib
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

IN_PARQUET_STEP_3 = pathlib.Path("data/30_embedded/articles.parquet")
IN_PARQUET_STEP_4 = pathlib.Path("data/40_preprocessed/41_classification/articles.parquet")

tasks = [
    (IN_PARQUET_STEP_3, "streamlit_app/data_sample/articles_30_embedded_sample.parquet"),
    (IN_PARQUET_STEP_4, "streamlit_app/data_sample/articles_41_classification_sample.parquet"),
]

def process_file(task):
    in_path, out_path = task
    df = pd.read_parquet(in_path)
    df_sample = df.sample(n=12000, random_state=42)
    df_sample.to_parquet(out_path, index=False)

if __name__ == "__main__":
    with ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(process_file, tasks), total=len(tasks), desc="Sampling Parquet Files"))