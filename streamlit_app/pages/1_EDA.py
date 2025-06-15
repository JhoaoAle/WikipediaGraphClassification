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
from sklearn.decomposition import PCA, TruncatedSVD
import os
import requests
from pathlib import Path

# This breaks the code when deploying to streamlit cloud, so we use a relative path
# IN_PARQUET = pathlib.Path("/30_embedded/articles.parquet")


@st.cache_data
def load_data(relative_path: str):
    # Resolve the relative path based on the current file's location
    current_dir = Path(__file__).parent  # This is streamlit_app/pages/
    local_path = (current_dir / relative_path).resolve()

    if not local_path.exists():
        st.error(f"File not found: {local_path}")
        st.stop()

    return pd.read_parquet(local_path)



# Load your dataframe here
df = load_data("../data_sample/articles_41_classification_sample.parquet")

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

    st.markdown("""
    First of all, we want to explore the behaviour of the most relevant missing columns to understand hwo the data behaves on its initial state. 
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

   

    # Title and description
    st.title("📊 Distribution of Text Features")
    st.markdown("""
    This section shows histograms of selected text features.  
    Each plot includes a **slider** to control the x-axis range, enabling focused exploration and dynamic binning.
    """)

    count_features = [
        'char_count',
        'word_count',
        'sentence_count',
        'uppercase_word_count'
    ]

    ratio_features = [
        'avg_word_length',
        'avg_sentence_length',
        'stopword_ratio',
        'punctuation_ratio'
    ]

    st.markdown("### 📘 Count-Based Features (Left) & 📗 Ratio Features (Right)")
    # Columns layout
    col1, col2 = st.columns(2)

    # Count-based plots in col1
    with col1:
        for feature in count_features:
            with st.expander(f"{feature}", expanded=True):
                min_val = float(df[feature].min())
                max_val = float(df[feature].max())
                range_vals = st.slider(
                    f"Range for `{feature}`",
                    min_value=min_val,
                    max_value=max_val,
                    value=(min_val, max_val)
                )
                filtered = df[(df[feature] >= range_vals[0]) & (df[feature] <= range_vals[1])]
                fig = px.histogram(filtered, x=feature, nbins=30, color_discrete_sequence=["skyblue"])
                fig.update_layout(
                    xaxis_title=feature,
                    yaxis_title="Frequency",
                    bargap=0.1,
                    margin=dict(l=40, r=20, t=40, b=40)
                )
                st.plotly_chart(fig, use_container_width=True)

    # Ratio-based plots in col2
    with col2:
        for feature in ratio_features:
            with st.expander(f"{feature}", expanded=True):
                min_val = float(df[feature].min())
                max_val = float(df[feature].max())
                range_vals = st.slider(
                    f"Range for `{feature}`",
                    min_value=min_val,
                    max_value=max_val,
                    value=(min_val, max_val)
                )
                filtered = df[(df[feature] >= range_vals[0]) & (df[feature] <= range_vals[1])]
                fig = px.histogram(filtered, x=feature, nbins=30, color_discrete_sequence=["lightgreen"])
                fig.update_layout(
                    xaxis_title=feature,
                    yaxis_title="Frequency",
                    bargap=0.1,
                    margin=dict(l=40, r=20, t=40, b=40)
                )
                st.plotly_chart(fig, use_container_width=True)

 

with tab3:
    st.header("Reducing Dimensionality")
    st.subheader("PCA vs SVD: Cumulative Explained Variance")

    st.write("""
        Although we start with high-dimensional embedding vectors (768 dimensions),
        both PCA and Truncated SVD allow us to reduce dimensionality while preserving variance.
        This comparison helps us understand how well each method captures the underlying structure.
    """)

    # Load data
    df = load_data("../data_sample/articles_30_embedded_sample.parquet")

    # Step 1: Extract embedding vectors
    embedding_cols = [col for col in df.columns if col.startswith('emb_')]
    X = df[embedding_cols].values
    df.drop(columns=embedding_cols + ['cleaned_article_body'], inplace=True)

    # Step 2: PCA
    pca = PCA(n_components=300)
    X_pca = pca.fit_transform(X)
    pca_cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    # Step 3: Truncated SVD
    svd = TruncatedSVD(n_components=300, random_state=42)
    X_svd = svd.fit_transform(X)
    svd_cumulative_variance = np.cumsum(svd.explained_variance_ratio_)

    # Components index
    components = list(range(1, 301))

    # Plotly figure
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=components,
        y=pca_cumulative_variance,
        mode='lines+markers',
        name='PCA',
        line=dict(color='royalblue'),
        marker=dict(size=1)
    ))

    fig.add_trace(go.Scatter(
        x=components,
        y=svd_cumulative_variance,
        mode='lines+markers',
        name='Truncated SVD',
        line=dict(color='firebrick'),
        marker=dict(size=1)
    ))

    fig.update_layout(
        title='Cumulative Explained Variance: PCA vs Truncated SVD',
        xaxis_title='Number of Components',
        yaxis_title='Cumulative Explained Variance',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(gridcolor='lightgray'),
        yaxis=dict(gridcolor='lightgray'),
        height=400,
        legend=dict(bgcolor='rgba(255,255,255,0.5)')
    )

    # Show chart
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
The plot above shows the **cumulative explained variance** as we increase the number of components for both **PCA** and **Truncated SVD**.
Each point on the lines represents the total variance retained by the first *n* components, giving us insight into how much of the original data’s structure is preserved after dimensionality reduction.

We observe that the first **200 components** capture approximately **88.66%** of the variance using PCA and **88.59%** with Truncated SVD.  
This strong alignment between the two methods highlights an important insight: despite using different mathematical strategies, both techniques are effectively capturing the **same essential structure** of the data.

### Why are the curves almost identical?

- **PCA** computes the eigenvectors of the covariance matrix of the data, centering it and projecting it onto directions of maximal variance.
- **Truncated SVD** performs a low-rank approximation directly on the (uncentered) data matrix.
- When the input data is **dense and well-behaved** (like our sentence embeddings), Truncated SVD approximates PCA very closely.

### Why is this similarity relevant?

- **Interpretability & Trust**: The near-identical variance retention curves reinforce confidence in the results — the variance captured is intrinsic to the data, not an artifact of the method.
- **Flexibility**: Since both methods yield similar outcomes, we can choose based on **speed**, **scalability**, or **tooling support** without sacrificing quality.
- **Validation of Embedding Quality**: The smooth shape of the curves suggests that information is well distributed across embedding dimensions — a desirable property in semantic spaces.

### Conclusion

The convergence of PCA and Truncated SVD results validates our choice of **200 components** as an efficient trade-off between dimensionality reduction and information preservation.  
This allows us to reduce from 768 to 200 dimensions while still retaining **nearly 89%** of the original variance — enabling faster processing, simpler models, and better visualization without significant loss of semantic content.
""")

