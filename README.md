
## Table of Contents
- [Table of Contents](#table-of-contents)
- [Wikipedia Graph Analysis Project](#wikipedia-graph-analysis-project)
  - [🔧 Setup Instructions](#-setup-instructions)
    - [1. Activate Virtual Environment (Windows PowerShell)](#1-activate-virtual-environment-windows-powershell)
    - [2. Install Dependencies](#2-install-dependencies)
  - [🗂️ Project Structure](#️-project-structure)
  - [🚀 Running the Pipeline](#-running-the-pipeline)
    - [Step 0: Ingest Data](#step-0-ingest-data)
    - [Step 1: Parse XML to Parquet](#step-1-parse-xml-to-parquet)
    - [Step 2: Transform and Clean Data](#step-2-transform-and-clean-data)
    - [Step 3: Generate embeddings vector](#step-3-generate-embeddings-vector)
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
│   ├── 20_clean/           # Stores clean_body + destination articles (Parquet format)
│   ├── 30_embeddings/      # Adds embeddings to articles (Parquet format)   
│   ├── 40_clusters/        # Store clustering results (e.g., labels, centroids)
│   └── 50_evaluation/      # Store evaluation metrics/results
├── src/
│   ├── 00_ingest.py        # Script to download Wikipedia dumps
│   ├── 10_parse.py         # Script to parse XML to Parquet
│   ├── 20_transform.py     # Script to clean markup and extract links
│   ├── 30_embed.py         # Script to generate embeddings vector of articles
│   ├── 40_cluster.py       # Scripts for clustering models
│   ├── 50_evaluate.py      # Scripts for evaluation of clustering
│   ├── 60_dashboard.py     # Streamlit dashboard app
│   ├── 70_report.py        # Script to generate PDF report
│   └── utils/
│       ├── stream_bz2.py   # Utility for streaming bz2 compressed files
│       └── wikiclean.py    # Utility functions for cleaning Wikitext and columns
└── README.md           # This file
```

### 🚀 Running the Pipeline

The data processing pipeline consists of three main stages. Each stage writes its output to the data/ directory and is idempotent, meaning if the target file already exists, rerunning the script will verify the timestamp and exit without reprocessing.

#### Step 0: Ingest Data

Download the latest Simple English Wikipedia dump (approximately 60 MB). This step only needs to be run once, or you can skip it if you plan to parse from a URL directly in the next step.

``` python
python src/00_ingest.py
```

Output: Potentially stores downloaded dump in <code>data/00_raw/</code> 

#### Step 1: Parse XML to Parquet

Parse the downloaded XML dump (or directly from a URL) into a Parquet file containing article titles and raw Wikitext. This stage produces a file of approximately 230 MB with around 140,000 pages.

``` python
python src/10_parse.py
```

Output: <code>data/10_parsed/articles.parquet</code>

#### Step 2: Transform and Clean Data

Clean the Wikitext markup from the parsed data and extract outgoing links from each article.

``` python
python src/20_transform.py
```

Output: <code>data/20_clean/articles.parquet</code>

#### Step 3: Generate embeddings vector

Generate a column of embedding vectors for the Wikitext markup from the cleaned data and export the result into a .parquet.

``` python
python src/30_embed.py
```

#### Execution Summary

To run the entire pipeline:

``` python
# 1. Download the data (run once, or skip if parsing from URL)
python src/00_ingest.py

# 2. Parse XML to Parquet
python src/10_parse.py

# 3. Clean markup and extract outgoing links
python src/20_transform.py

# 4. Generate embbedings vector
# python src/30_embed.py 
```


### 📚 Acknowledgements

This project uses the <code>wikitextparser</code> library to parse Wikitext into structured content.

Documentation up to date

