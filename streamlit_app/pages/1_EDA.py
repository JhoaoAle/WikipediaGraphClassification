# streamlit_app_name: Initial exploration
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pathlib
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def main():
    st.title("🔍 Exploratory Data Analysis")

st.set_page_config(
    page_title="Exploratory Data Analysis",
    page_icon="🔍",
    layout="centered"
)

IN_PARQUET = pathlib.Path("data/30_embedded/articles.parquet")

# Load your dataframe here
df = pd.read_parquet(IN_PARQUET)

st.title("Exploratory Data Analysis")
tab1, tab2, tab3 = st.tabs(["Understanding the Problem", "Inspecting Data", "Reducing Dimensionality"])

with tab1:
    st.header("Understanding the Problem")

    st.subheader("The Problem Being Solved by This Project")

    st.markdown("""
    This project tackles the significant data engineering challenge of transforming the vast, raw, and complex data of Wikipedia into a clean and structured dataset suitable for advanced analysis.
    """)

    st.markdown(""" ### More specifically, it addresses the following sub-problems:
    - **Accessibility and Usability**  
    Raw Wikipedia dumps are difficult to work with from the start for directly for graph analysis or machine learning. This project creates a more accessible dataset stored in .parquet format, more suited for large data storage than a csv, specially for a case where the data columns contain special characters such as "," and ";".

    - **Noise Reduction and Feature Extraction**  
    Wikitext contains a lot of markup and non-content elements. The pipeline **cleans the text** and extracts **structural features** (like outgoing links which form graph edges).

    - **Semantic Understanding**  
    Simple link analysis doesn’t capture the meaning of articles. By generating **text embeddings**, the project adds a layer of **semantic understanding** to the article nodes.

    - **Reproducibility and Efficiency**  
    With a clear and modular pipeline, the project ensures a **reproducible** and **scalable** way to produce analysis-ready graph data.
    """)

    st.markdown("### In essence:")
    st.markdown("""
    > **How do we efficiently and effectively prepare Wikipedia data to be used as a high-quality, semantically-aware graph for research and analysis?**
    """)

    st.subheader("Research Question Enabled by This Work")
    st.markdown("""
    > **To what extent does the integration of semantic content (via text embeddings) with the explicit hyperlink structure of Wikipedia improve the detection, coherence, and characterization of thematic communities or knowledge domains within the encyclopedia, compared to analyses relying on network topology alone?**
    """)   

with tab2:
    st.header("Inspecting Data")

    # Section: Word Frequency Distribution
    st.subheader("Distribution of relevant missing values")
    df['word_count'] = df['cleaned_article_body'].apply(lambda x: len(x.split()))

    col1, col2 = st.columns([2, 1])

    with col1:
        # Count rows with and without a category mapped
        has_category = df['categories'].apply(lambda x: len(x) > 0).sum()
        no_category = df['categories'].apply(lambda x: len(x) == 0).sum()

        # Create DataFrame
        pie_df = pd.DataFrame({
            'Status': ['Has Assigned Category', 'No Assigned Category'],
            'Count': [has_category, no_category]
        })

        # Plotly Pie Chart
        fig = px.pie(
            pie_df,
            names='Status',
            values='Count',
            title='Availability of Category Distribution',
            hole=0.4,  # donut-style, optional
        )

        # Customize appearance
        fig.update_traces(
            textinfo='percent+label',
            hoverinfo='label+percent+value'
        )
        fig.update_layout(
            showlegend=True,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )

        # Show in Streamlit
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Count rows with and without linked articles
        has_links = df['linked_article_titles'].apply(lambda x: len(x) > 0).sum()
        no_links = df['linked_article_titles'].apply(lambda x: len(x) == 0).sum()

        # Create DataFrame
        pie_df = pd.DataFrame({
            'Status': ['Has Links', 'No Links'],
            'Count': [has_links, no_links]
        })

        # Plotly Pie Chart
        fig = px.pie(
            pie_df,
            names='Status',
            values='Count',
            title='Linked Articles Distribution',
            hole=0.4,  # donut-style, optional
        )

        # Customize appearance
        fig.update_traces(
            textinfo='percent+label',
            hoverinfo='label+percent+value'
        )
        fig.update_layout(
            showlegend=True,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )

        # Show in Streamlit
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Distribution of Most Popular Categories (Log Scale)")

    # Flatten all categories from lists into one big list
    all_categories = df['categories'].explode()
    category_counts = Counter(all_categories)

    # Convert to DataFrame and get top
    category_df = pd.DataFrame(category_counts.items(), columns=['Category', 'Count'])
    top_categories = category_df.sort_values(by='Count', ascending=False).head(50)

    # Add log-transformed column
    top_categories['LogCount'] = np.log10(top_categories['Count'])

    # Plot using log-transformed values
    fig = px.bar(
        top_categories,
        x='Category',
        y='LogCount',
        title='Top 15 Most Frequent Categories (Logarithmic Scale)',
        color='LogCount',
        color_continuous_scale='Blues',
        labels={'LogCount': 'log₁₀(Count)'}
    )

    # Optional: Improve layout
    fig.update_layout(
        yaxis_title='log₁₀(Number of Articles)',
        xaxis_title='Category',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis_tickangle=45
    )

    # Optional: Show both raw and log count on hover
    fig.update_traces(
        hovertemplate='<b>%{x}</b><br>log₁₀(Count): %{y:.2f}<br>Raw Count: %{customdata}',
        customdata=top_categories[['Count']].values
    )

    # Show chart
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Reducing Dimensionality")
    st.subheader("PCA Visualization of Article Embeddings")

    # Step 1: Select embedding columns
    embedding_cols = [col for col in df.columns if col.startswith('emb_')]
    X = df[embedding_cols].values
    df.drop(columns=embedding_cols + ['cleaned_article_body'], inplace=True)

    # Step 2: Apply PCA
    pca = PCA()
    X_pca = pca.fit_transform(X)

    # Cumulative variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    components = list(range(1, len(cumulative_variance) + 1))

    # Create plotly figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=components,
        y=cumulative_variance,
        mode='lines+markers',
        name='Cumulative Explained Variance',
        line=dict(color='royalblue'),
        marker=dict(size=6)
    ))

    fig.update_layout(
        title='Explained Variance by PCA Components',
        xaxis_title='Number of PCA Components',
        yaxis_title='Cumulative Explained Variance',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(gridcolor='lightgray'),
        yaxis=dict(gridcolor='lightgray'),
        height=500
    )

    st.write("Placeholder for explanatory text")
