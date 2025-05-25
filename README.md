
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
    - [Step 4:](#step-4)
    - [Execution Summary](#execution-summary)
  - [📚 Acknowledgements](#-acknowledgements)


## Wikipedia Graph Analysis Project

This project provides a pipeline for downloading, parsing, and cleaning Wikipedia dumps to extract article text and outgoing links. This data can then be used for various graph analysis tasks.

### 🔧 Setup Instructions

Follow these steps to set up your project environment.

#### 1. Activate Virtual Environment (Windows PowerShell)

**Note:** Depending on your installation, you might require to use <code>py</code> instead of <code>python</code> whenever trying to run a python command presented in this documentation

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
│   ├── 00_raw/             # (Optional) Compressed Wikipedia dump(s)
│   ├── 10_parsed/          # Stores title + raw Wikitext (Parquet format)
│   ├── 20_transformed/     # Stores clean_body + destination articles (Parquet format)
│   ├── 30_embedded/        # Adds embeddings to articles (Parquet format)   
│   ├── 40_preprocesed/     # Preprocesses the dataset with embeddings and return a clustering-ready dataset (Parquet format)   
│   ├── 40_clusters/        # Store clustering results (e.g., labels, centroids)
│   └── 50_evaluation/      # Store evaluation metrics/results
├── src/
│   ├── 00_ingest.py        # Script to download Wikipedia dumps
│   ├── 10_parse.py         # Script to parse XML to Parquet
│   ├── 20_transform.py     # Script to clean markup and extract links
│   ├── 30_embed.py         # Script to generate embeddings vector of articles
│   ├── 40_preprocess.py    # Script to clean a dataset with embeddings
│   ├── 50_cluster.py       # Scripts for clustering models
│   ├── 60_evaluate.py      # Scripts for evaluation of clustering
│   ├── 70_dashboard.py     # Streamlit dashboard app
│   ├── 80_report.py        # Script to generate PDF report
│   └── utils/
│       ├── stream_bz2.py   # Utility for streaming bz2 compressed files
│       └── wikiclean.py    # Utility functions for cleaning Wikitext and columns
└── README.md           # This file
```

### 🚀 Running the Pipeline

The data processing pipeline consists of four main stages. Each stage writes its output to the data/ directory and is idempotent, meaning if the target file already exists, rerunning the script will verify the timestamp and exit without reprocessing.

#### Step 0: Ingest Data (Optional)

Download the latest Simple English Wikipedia dump (approximately 60 MB). This step only needs to be run once, or you can skip it if you plan to parse from a URL directly in the next step.

``` python
python src/00_ingest.py
```

**Output:** Potentially stores downloaded dump in <code>data/00_raw/</code> 

#### Step 1: Parse XML to Parquet

Parse the downloaded XML dump (or directly from a URL) into a Parquet file containing article titles and raw Wikitext. This stage produces a file of approximately 230 MB with around 140,000 pages.

``` python
python src/10_parse.py
```

**Output:** <code>data/10_parsed/articles.parquet</code>

#### Step 2: Transform and Clean Data

Clean the Wikitext markup from the parsed data and extract outgoing links from each article.

``` python
python src/20_transform.py
```

**Output:** <code>data/20_clean/articles.parquet</code>

> At this point, the dataset contains the title of the article, the cleaned content, the articles to which it links, the amount of sections and the categories to which it belongs. The dataset weighs around 240 MB

#### Step 3: Generate embeddings vector

Generate a set of columns from the embedding vectors for the Wikitext markup from the transformed data and export the result into a .parquet. 

In this stage, the text column <code>cleaned_article_body</code> is tokenized. After this, the embedding associated values are generated accordingly, resulting in the addition of 768 columns associated with the dimension of the embedding vector for each text.

The resulting DataFrame is exported into a .parquet file


``` python
python src/30_embed.py
```

**Output:** <code>data/30_embedded/articles.parquet</code>

> At this point, do note this dataset is considered ready for cleaning. This raw dataset weighs around 1 GB

#### Step 4:


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
```


### 📚 Acknowledgements

This project uses the <code>wikitextparser</code> library to parse Wikitext into structured content.

Documentation up to date

