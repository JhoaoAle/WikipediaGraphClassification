# src/analysis/category_analysis.py

import pandas as pd
from collections import Counter

def compute_category_stats(df):
    exploded_categories = df['categories'].explode()
    category_counts = Counter(exploded_categories)
    categories_df = pd.DataFrame.from_dict(category_counts, orient='index',
                                           columns=['frequency']) \
                                .sort_values('frequency', ascending=False)
    stats = {
        "total_unique_categories": len(categories_df),
        "most_common_category": categories_df.index[0],
        "most_common_count": int(categories_df.iloc[0, 0]),
        "categories_once": int((categories_df['frequency'] == 1).sum()),
        "categories_twice": int((categories_df['frequency'] == 2).sum()),
        "categories_thrice": int((categories_df['frequency'] == 3).sum()),
    }
    return categories_df, stats