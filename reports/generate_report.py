import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import os
# Render LaTeX template
from jinja2 import Environment, FileSystemLoader

# Create output folder if it doesn't exist
os.makedirs('reports/generated/plots', exist_ok=True)

# Load data
df = pd.read_parquet('data/30_embedded/articles.parquet')

# Explode categories
exploded_categories = df['categories'].explode()

# Count frequencies
category_counts = Counter(exploded_categories)

# Build dataframe
categories_df = pd.DataFrame.from_dict(category_counts, orient='index', 
                                       columns=['frequency']) \
                            .sort_values('frequency', ascending=False)

# Plot histogram → save
plt.figure(figsize=(12, 6))
plt.hist(categories_df['frequency'], bins=100, log=True, color='salmon')
plt.title("Histogram of Category Frequencies (Log Scale)")
plt.xlabel("Frequency (number of occurrences)")
plt.ylabel("Number of Categories")
plt.tight_layout()

# Save instead of show
plt.savefig('reports/generated/plots/histogram_plot.png')

env = Environment(loader=FileSystemLoader('reports/templates'))
template = env.get_template('report_template.tex')

template_vars = {
    "total_unique_categories": len(categories_df),
    "most_common_category": categories_df.index[0],
    "most_common_count": int(categories_df.iloc[0, 0]),
    "categories_once": int((categories_df['frequency'] == 1).sum()),
    "categories_twice": int((categories_df['frequency'] == 2).sum()),
    "categories_thrice": int((categories_df['frequency'] == 3).sum()),
    "top_categories_plot_path": "plots/top_categories.png",
    "cumulative_plot_path": "plots/cumulative_plot.png",
    "histogram_plot_path": "plots/histogram_plot.png",
}

output_tex_path = 'reports/generated/report.tex'
with open(output_tex_path, 'w') as f:
    f.write(template.render(**template_vars))

# Optional: automatically run pdflatex
import subprocess
subprocess.run(['pdflatex', '-output-directory', 'reports/generated', output_tex_path])