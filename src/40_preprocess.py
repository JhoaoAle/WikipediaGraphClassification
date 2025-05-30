import pandas as pd
import pathlib  
from typing import List, Dict
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import string
from nltk.corpus import stopwords
import re
from tqdm import tqdm
from functools import partial
tqdm.pandas()


# Only needed the first time
#nltk.download('punkt')
#nltk.download('stopwords')

def extract_text_features(text, stop_words):
    if not isinstance(text, str) or not text.strip():
        return pd.Series([0, 0, 0, 0, 0, 0, 0, 0], 
                         index=['char_count', 'word_count', 'sentence_count', 'avg_word_length',
                                'avg_sentence_length', 'uppercase_word_count', 
                                'stopword_ratio', 'punctuation_ratio'])

    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    punctuations = set(string.punctuation)

    char_count = len(text)
    word_count = len(words)
    sentence_count = len([s for s in sentences if s.strip()])
    avg_word_length = np.mean([len(word) for word in words]) if words else 0
    avg_sentence_length = word_count / sentence_count if sentence_count else 0
    uppercase_word_count = sum(1 for word in words if word.isupper())
    stopword_count = sum(1 for word in words if word.lower() in stop_words)
    punctuation_count = sum(1 for char in text if char in punctuations)
    
    return pd.Series([
        char_count,
        word_count,
        sentence_count,
        avg_word_length,
        avg_sentence_length,
        uppercase_word_count,
        stopword_count / word_count if word_count else 0,
        punctuation_count / char_count if char_count else 0
    ], index=[
        'char_count',
        'word_count',
        'sentence_count',
        'avg_word_length',
        'avg_sentence_length',
        'uppercase_word_count',
        'stopword_ratio',
        'punctuation_ratio'
    ])


IN_PARQUET = pathlib.Path("data/30_embedded/articles.parquet")
OUT_PARQUET = pathlib.Path("data/40_preprocessed/articles.parquet")



def main():
    stop_words = set(stopwords.words('english'))

    df = pd.read_parquet(IN_PARQUET)

    # Step 1: Select embedding columns
    embedding_cols = [col for col in df.columns if col.startswith('emb_')]
    X = df[embedding_cols].values


    # Step 3: Apply PCA to reduce dimensions (you can change n_components as needed)
    pca = PCA(n_components=200)  
    X_pca = pca.fit_transform(X)

    # Add PCA columns to dataframe
    pca_df = pd.DataFrame(X_pca, columns=[f'pca_{i}' for i in range(X_pca.shape[1])], index=df.index)
    df = pd.concat([df, pca_df], axis=1)

    # Apply to DataFrame
    feature_extractor = partial(extract_text_features, stop_words=stop_words)

    df_text_features = df['cleaned_article_body'].progress_apply(feature_extractor)

    # Concatenate new features and drop original text column
    df = pd.concat([df, df_text_features], axis=1)

    # Drop original embedding columns
    df.drop(columns=embedding_cols + ['cleaned_article_body'], inplace=True)

    # Save preprocessed dataframe
    df.to_parquet(OUT_PARQUET, index=False)

    print("✓ wrote", OUT_PARQUET, len(df), "rows")

if __name__ == "__main__":
    main()
    