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
import os
import requests
from pathlib import Path

# This breaks the code when deploying to streamlit cloud, so we use a relative path
# IN_PARQUET = pathlib.Path("../data/30_embedded/articles.parquet")

@st.cache_data
def load_data():
    url = "https://huggingface.co/datasets/Qu4ntz/articles_simple_wiki/resolve/main/articles.parquet"
    local_path = Path("articles.parquet")  # Use safe path

    if not local_path.exists():
        with st.spinner("Downloading dataset..."):
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

    return pd.read_parquet(local_path)

def main():
    st.title("🔍 Exploratory Data Analysis")

st.set_page_config(
    page_title="Exploratory Data Analysis",
    page_icon="🔍",
    layout="centered"
)



# Load your dataframe here
df = load_data()

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


    st.markdown("""
    First of all, we want to explore the behaviour of the msot relevant missing columns to understand hwo the data behaves on its initial state. 
    To do so, we will focus in the two most relevant columns in the basic dataset: 
    - `categories` column, which contains the categories assigned to each article in Simple Wikipedia
    - `linked_article_titles` column, which contains the titles of articles that can be accessed from the current article through hyperlinks.
    """)   

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

    st.markdown("""
    Observing the above plots, we can see that:
        - 99% of articles have at least one article to which they link, which is a good sign that the articles are well connected.
        - 30% of articles have no assigned category, which is a significant portion. This could indicate that the categories are not consistently applied.     
    """)   

    st.subheader("Distribution of Most Popular Categories (Log Scale)")

    st.markdown("""
    Along the categorized articles, what we want to explore is the distribution of the most popular categories in the dataset; to evaluate if they behave in a balanced manner or if there are categories that are more heavily represented than others.  
    """)   

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

    st.markdown("""
    Observing the above plot, we can see that there is a certain bias: The most popular category, 'Living people', has 34886 articles, while the second most popular category, '2020 Deaths', has only 4950 articles. 
    
    This indicates that the categories are not evenly distributed, with the most popular article having 7 times more articles than the second most popular category. This could lead to an imbalance in the dataset when we realize a neighboring analysis.
                
    Another important consideration is that the categories are not mutually exclusive, meaning that an article can belong to multiple categories; this means that having this unequity in the categories distribution could indicate an inherent bias in the dataset, which could affect the results of the analysis.
    """) 
 
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

    # Show chart
    st.plotly_chart(fig, use_container_width=True)

    st.write("Despite starting with embedding vectors of 768 dimensions, PCA allows us to reduce the dimensionality while retaining a significant amount of variance. The cumulative explained variance plot shows how many components are needed to capture most of the variance in the data." \
    "" \
    "With 200 dimensions, we can retain over 90% of the variance, which is a good balance between dimensionality reduction and information preservation.")
