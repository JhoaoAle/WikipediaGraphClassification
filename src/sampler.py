import pandas as pd
import pathlib

IN_PARQUET = pathlib.Path("data/30_embedded/articles.parquet")

# Load the full file locally
df = pd.read_parquet(IN_PARQUET)

# Take a sample (e.g., 150 rows)
df_sample = df.sample(n=10000, random_state=42)



# Save it
df_sample.to_parquet("articles_sample.parquet", index=False)