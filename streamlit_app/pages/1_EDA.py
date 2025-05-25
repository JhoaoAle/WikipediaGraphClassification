import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pathlib

IN_PARQUET = pathlib.Path("data/30_embedded/articles.parquet")

# Load your dataframe here
df = pd.read_parquet(IN_PARQUET)

st.title("Exploratory Data Analysis")
tab1, tab2, tab3 = st.tabs(["Understanding the Problem", "Inspecting Data", "Managing missing values", "Handling outliers", "Dimensionality reduction"])

with tab1:
    st.header("Understanding the Problem")
    # Word count distribution
    df['word_count'] = df['cleaned_article_body'].apply(lambda x: len(x.split()))
    fig, ax = plt.subplots()
    sns.histplot(df['word_count'], bins=50, ax=ax)
    ax.set_title("Word Count Distribution")
    st.pyplot(fig)

    st.write("Data sample:")
    st.dataframe(df.head())

with tab2:
    st.header("Embeddings")
    st.write("PCA or t-SNE visualization here...")

with tab3:
    st.header("Future Work")
    st.write("Placeholders for future sections.")
