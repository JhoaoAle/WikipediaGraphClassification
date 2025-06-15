import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pathlib
import plotly.graph_objects as go
import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_score,
    completeness_score,
    v_measure_score
)
from pathlib import Path

# Set page title
st.set_page_config(page_title="📊 Clustering Comparison")

st.title("📊 Clustering Comparison Metrics")

# Load data
@st.cache_data
def load_data(relative_path: str):
    current_dir = Path(__file__).parent
    local_path = (current_dir / relative_path).resolve()

    if not local_path.exists():
        st.error(f"File not found: {local_path}")
        st.stop()

    return pd.read_parquet(local_path)

# Path to your sample data
df = load_data("../data_sample/articles_50_clustered_sample.parquet")

# Define comparisons
cluster_pairs = [
    ("louvain_community", "kmeans_cluster"),
    ("louvain_community", "leiden_community"),
    ("kmeans_cluster", "leiden_community"),
]

results = []

# Compute and collect metrics
for col1, col2 in cluster_pairs:
    df_clean = df.dropna(subset=[col1, col2])
    labels1 = df_clean[col1].astype(int)
    labels2 = df_clean[col2].astype(int)

    ari = adjusted_rand_score(labels1, labels2)
    nmi = normalized_mutual_info_score(labels1, labels2)
    homogeneity = homogeneity_score(labels1, labels2)
    completeness = completeness_score(labels1, labels2)
    v_measure = v_measure_score(labels1, labels2)

    results.append({
        "Clustering 1": col1,
        "Clustering 2": col2,
        "ARI": ari,
        "NMI": nmi,
        "Homogeneity": homogeneity,
        "Completeness": completeness,
        "V-Measure": v_measure,
        "Valid Rows": len(df_clean)
    })

# Convert to DataFrame for display
results_df = pd.DataFrame(results)

# Display as table
st.subheader("📋 Metric Table")
st.dataframe(results_df.style.format(precision=4), use_container_width=True)

# Heatmap of the metrics
st.subheader("🔍 Metric Comparison Heatmap")

# Melt the dataframe for heatmap-style Plotly heatmap
melted = results_df.melt(id_vars=["Clustering 1", "Clustering 2"], 
                         value_vars=["ARI", "NMI", "Homogeneity", "Completeness", "V-Measure"],
                         var_name="Metric", value_name="Score")

# Combine the two clustering columns for labeling
melted["Comparison"] = melted["Clustering 1"] + " vs " + melted["Clustering 2"]

# Pivot for heatmap
heatmap_data = melted.pivot(index="Comparison", columns="Metric", values="Score")

# Plotly heatmap
fig = go.Figure(data=go.Heatmap(
    z=heatmap_data.values,
    x=heatmap_data.columns,
    y=heatmap_data.index,
    colorscale='Blues',
    hoverongaps=False,
    zmin=0,
    zmax=1
))

fig.update_layout(
    height=400,
    xaxis_title="Metric",
    yaxis_title="Comparison",
    title="Clustering Metrics Heatmap",
    margin=dict(l=10, r=10, t=50, b=10)
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("🔀 Cluster Mapping Sankey Diagram")

# Let user select a pair
selected_pair = st.selectbox(
    "Select a clustering comparison for Sankey visualization:",
    cluster_pairs,
    format_func=lambda pair: f"{pair[0]} vs {pair[1]}"
)

# Extract column names
col1, col2 = selected_pair
df_sankey = df.dropna(subset=[col1, col2]).copy()

# Add prefixed string labels
df_sankey["source_label"] = col1 + "_" + df_sankey[col1].astype(int).astype(str)
df_sankey["target_label"] = col2 + "_" + df_sankey[col2].astype(int).astype(str)

# Get top 10 source and target labels
top_source = df_sankey["source_label"].value_counts().nlargest(10).index.tolist()
top_target = df_sankey["target_label"].value_counts().nlargest(10).index.tolist()

# Filter to rows with top clusters only
df_filtered = df_sankey[
    df_sankey["source_label"].isin(top_source) & df_sankey["target_label"].isin(top_target)
]

# Count transitions
transition_counts = df_filtered.groupby(["source_label", "target_label"]).size().reset_index(name="count")

# Build label list (source first, then target to force left-to-right)
source_labels = sorted(transition_counts["source_label"].unique())
target_labels = sorted(transition_counts["target_label"].unique())
all_labels = source_labels + target_labels

# Map labels to indices
label_to_index = {label: idx for idx, label in enumerate(all_labels)}
transition_counts["source"] = transition_counts["source_label"].map(label_to_index)
transition_counts["target"] = transition_counts["target_label"].map(label_to_index)

# Create Sankey diagram
fig_sankey = go.Figure(data=[go.Sankey(
    arrangement="snap",
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=all_labels,
        color="lightblue"
    ),
    link=dict(
        source=transition_counts["source"],
        target=transition_counts["target"],
        value=transition_counts["count"]
    )
)])

fig_sankey.update_layout(
    title_text=f"Cluster Mapping (Top 10): {col1} → {col2}",
    font_size=12,
    margin=dict(l=10, r=10, t=50, b=10)
)

st.plotly_chart(fig_sankey, use_container_width=True)

