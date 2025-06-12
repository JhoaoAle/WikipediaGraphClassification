
## Table of Contents
- [Table of Contents](#table-of-contents)
- [Wikipedia Graph Analysis Project](#wikipedia-graph-analysis-project)
  - [🔧 Setup Instructions](#-setup-instructions)
    - [1. Activate Virtual Environment (Windows PowerShell)](#1-activate-virtual-environment-windows-powershell)
    - [2. Install Dependencies](#2-install-dependencies)
  - [🗂️ Project Structure](#️-project-structure)
  - [🚀 Running the Pipeline](#-running-the-pipeline)
    - [Step 0: Ingest Data (Optional)](#step-0-ingest-data-optional)
    - [Step 1: Parse XML to Parquet](#step-1-parse-xml-to-parquet)
    - [Step 2: Transform and Clean Data](#step-2-transform-and-clean-data)
    - [Step 3: Generate embeddings vector](#step-3-generate-embeddings-vector)
    - [Step 4: Preprocess the generated dataset](#step-4-preprocess-the-generated-dataset)
    - [Execution Summary](#execution-summary)
    - [Streamlit Dashboard execution](#streamlit-dashboard-execution)
  - [📚 Acknowledgements](#-acknowledgements)


## Wikipedia Graph Analysis Project

This project provides a pipeline for downloading, parsing, and cleaning Wikipedia dumps to extract article text and outgoing links. This data can then be used for various graph analysis tasks.

### 🔧 Setup Instructions

Follow these steps to set up your project environment.

#### 1. Activate Virtual Environment (Windows PowerShell)

**Note:** Depending on your installation, you might require to use `py` instead of `python` whenever trying to run a python command presented in this documentation

It's recommended to use a virtual environment to manage project dependencies:

``` powershell
python -m venv venv
Set-ExecutionPolicy Unrestricted -Scope Process
.\\venv\Scripts\Activate.ps1
 ```

#### 2. Install Dependencies

Install the necessary Python libraries using the provided requirements.txt file.

``` powershell
pip install -r requirements.txt
```

### 🗂️ Project Structure

The project is organized as follows:

```
project/
├── data/
│   ├── 00_raw/                     # (Optional) Compressed Wikipedia dump(s)
│   │   ├── simplewiki-latest-pages-articles.xml.bz2
│   │   └── placeholder.txt
│   ├── 10_parsed/                  # Stores title + raw Wikitext (Parquet format)
│   │   ├── articles.parquet
│   │   └── placeholder.txt
│   ├── 20_transformed/             # Stores clean_body + destination articles (Parquet format)
│   │   ├── articles.parquet
│   │   └── placeholder.txt
│   ├── 30_embedded/                # Adds embeddings to articles (Parquet format)
│   │   ├── articles.parquet
│   │   └── ...
│   ├── 31_tf_idf/                  # Stores TF-IDF features (Parquet format)
│   │   └── ...
│   ├── 40_preprocessed/            # Generates clustering and network analysis-ready dataset (Parquet format)
│   └── 50_clustered/               # Stores clustering results
│
├── src/
│   ├── 00_ingest.py                # Script to download Wikipedia dumps
│   ├── 10_parse.py                 # Script to parse XML to Parquet
│   ├── 20_transform.py             # Script to clean markup and extract links
│   ├── 30_embed.py                 # Script to generate embeddings vector of articles
│   ├── 31_tf_idf.py                # Script to generate TF-IDF features for articles
│   ├── 40_preprocess.py            # Script to clean a dataset with embeddings
│   ├── 51_kmeans.py                # Applies KMeans clustering to preprocessed data
│   ├── 52_hdbscan.py               # Applies HDBSCAN clustering to preprocessed data
│   ├── 53_louvain.py               # Applies Louvain community detection
│   ├── 61_graph_generating.py      # Script to generate graph structures for analysis
│   ├── analysis/
│   │   └── category_analysis.py    # Analysis scripts for categories and more
│   └── utils/
│       ├── stream_bz2.py           # Utility for streaming bz2 compressed files
│       ├── textclean.py            # Utility functions for text cleaning
│       └── wikiclean.py            # Utility functions for cleaning Wikitext and columns
│
├── streamlit_app/
│   ├── home.py                     # Main page for running streamlit dashboard
│   ├── pages/              
│   │   └── 1_EDA.py                # First dashboard section. Works with dataset as-is; before cleaning
│   ├── data_sample/
│   │   └── articles_step_name_sample.parquet (*) # Dataset samples used to run the Streamlit Dashboard
│   └── sampler.py                  # Used to generate the sample files required for Streamlit Dashboard
│
├── reports/
│   ├── templates/                  # LaTeX templates (*.tex files, images)
│   ├── generated/                  # Output PDFs or *.tex files
│   └── generate_report.py          # Script that generates the LaTeX document
└── README.md                       # This file
```

### 🚀 Running the Pipeline

The data processing pipeline consists of four main stages. Each stage writes its output to the data/ directory and is idempotent, meaning if the target file already exists, rerunning the script will verify the timestamp and exit without reprocessing.

#### Step 0: Ingest Data (Optional)

Download the latest Simple English Wikipedia dump (approximately 200 MB). This step only needs to be run once, or you can skip it if you plan to parse from a URL directly in the next step.

``` python
python src/00_ingest.py
```

**Output:** Potentially stores downloaded dump in `data/00_raw/` 

#### Step 1: Parse XML to Parquet

Parse the downloaded XML dump (or directly from a URL) into a Parquet file containing article titles and raw Wikitext. This stage produces a file of approximately 230 MB with around 375000 pages.

``` python
python src/10_parse.py
```

**Output:** `data/10_parsed/articles.parquet`

#### Step 2: Transform and Clean Data

Clean the Wikitext markup from the parsed data and extract outgoing links from each article. We obtain text related metadata from the `cleaned_article_body` column. These columns are:

- **char_count**: Total number of characters in the text (including spaces and punctuation).

- **word_count**: Total number of words in the text.

- **sentence_count**: Total number of sentences, split using punctuation marks like `.`, `!`, and `?`.

- **avg_word_length**: Average length of the words in the text.

- **avg_sentence_length**: Average number of words per sentence.

- **uppercase_word_count**: Number of words that are fully uppercase.

- **stopword_ratio**: Proportion of words that are stopwords (based on a predefined stopword list).

- **punctuation_ratio**: Proportion of characters that are punctuation marks.

``` python
python src/20_transform.py
```

**Output:** `data/20_transformed/articles.parquet`

> At this point, the dataset contains the title of the article, the cleaned content, the articles to which it links, the amount of sections and the categories to which it belongs. The dataset weighs around 169 MB

#### Step 3: Generate embeddings vector

Generate a set of columns from the embedding vectors for the Wikitext markup from the transformed data and export the result into a .parquet. 

In this stage, the text column `cleaned_article_body` is tokenized. After this, the embedding associated values are generated accordingly, resulting in the addition of 768 columns associated with the dimension of the embedding vector for each text.

The resulting DataFrame is exported into a .parquet file


``` python
python src/30_embed.py
```

**Output:** `data/30_embedded/articles.parquet`

> At this point, do note this dataset is considered ready for cleaning. This raw dataset weighs around 1.64 GB

#### Step 4: Preprocess the generated dataset

In this stage, the text column `cleaned_article_body` is dropped, the column `article_id` is generated to keep consistency across the outputs; and two different datasets are generated:

**Mapping dataset**
In this dataset, we keep the columns: `article_id`, `title`, and `linked_article_titles`; since this will be used for network analysis; these are the only columns required at this point

**Clustering dataset**

In this dataset 105435 rows are redirects to other articles, which are not important for content-based clustering analysis, therefore, these will be dropped since they don't provide a significative insight for content-based clustering.

A set of columns from the embedding columns which represent the reduced dimensionality of the dataset through PCA and these columns will be included in the dataset whilst the embedding columns are dropped. 

``` python
python src/40_preprocess.py
```

**Output:** 
**One dataset for Mapping** 
`data/40_preprocessed_mapping/articles.parquet`

**One dataset for Clustering**
`data/40_preprocessed_classification/articles.parquet`

The resulting DataFrames are exported into a .parquet file

> At this point, do note these datasets are considered ready for clustering and network analysis. The raw datasets weigh around 42 MB and 377 MB

#### Execution Summary

To run the entire pipeline:

``` python
# 1. Download the data (run once, or skip if parsing from URL)
python src/00_ingest.py

# 2. Parse XML to Parquet
python src/10_parse.py

# 3. Clean markup and extract outgoing links
python src/20_transform.py

# 4. Generate embbeding vector
python src/30_embed.py 

# 5. Generate clen dataset from clustering and network analysis
python src/40_preprocess.py
```

#### Streamlit Dashboard execution

Given you have the file generated by Step 3, you should be able to check first page of the Streamlit Dashboard, EDA.
In order to do so, you must run this command, while on the root project folder:

``` python
streamlit run streamlit_app/home.py
```


### 📚 Acknowledgements

This project uses the `wikitextparser` library to parse Wikitext into structured content.

Documentation up to date

