## Wikipedia Graph Analysis Project

This project provides a pipeline for downloading, parsing, and cleaning Wikipedia dumps to extract article text and outgoing links. This data can then be used for various graph analysis tasks.

### 🔧 Setup InstructionsFollow these steps to set up your project environment.

#### 1. Activate Virtual Environment (Windows PowerShell)

It's recommended to use a virtual environment to manage project dependencies..

``` powershell
\venv\Scripts\Activate.ps1
```

##### Install Dependencies

Install the necessary Python libraries using the provided requirements.txt file.

``` powershell
pip install -r requirements.txt
```


### 🗂️ Project Structure

The project is organized as follows:

<pre lang="markdown"> 
    <code>
    ```text
     project/ ├── data/ │ ├── 00_raw/ # (Optional) Compressed Wikipedia dump(s) │ ├── 10_parsed/ # Stores title + raw Wikitext (Parquet format) │ └── 20_clean/ # Stores clean_body + destination articles (Parquet format) ├── src/ │ ├── 00_ingest.py # Script to download Wikipedia dumps │ ├── 10_parse.py # Script to parse XML to Parquet │ ├── 20_transform.py # Script to clean markup and extract links │ └── utils/ │ ├── stream_bz2.py # Utility for streaming bz2 compressed files │ └── wikiclean.py # Utility for cleaning Wikitext └── README.md # This file 
     ```
    </code> 
</pre>

project/
├── data/
│   ├── 00_raw/         # (Optional) Compressed Wikipedia dump(s)
│   ├── 10_parsed/      # Stores title + raw Wikitext (Parquet format)
│   └── 20_clean/       # Stores clean_body + destination articles (Parquet format)
├── src/
│   ├── 00_ingest.py    # Script to download Wikipedia dumps
│   ├── 10_parse.py     # Script to parse XML to Parquet
│   ├── 20_transform.py # Script to clean markup and extract links
│   └── utils/
│       ├── stream_bz2.py # Utility for streaming bz2 compressed files
│       └── wikiclean.py  # Utility for cleaning Wikitext
└── README.md           # This file


🚀 Running the PipelineThe data processing pipeline consists of three main stages. Each stage writes its output to the data/ directory and is idempotent, meaning if the target file already exists, rerunning the script will verify the timestamp and exit without reprocessing.Step 1: Ingest DataDownload the latest Simple English Wikipedia dump (approximately 60 MB). This step only needs to be run once, or you can skip it if you plan to parse from a URL directly in the next step.python src/00_ingest.py
Output: Potentially stores downloaded dump in data/00_raw/Step 2: Parse XML to ParquetParse the downloaded XML dump (or directly from a URL) into a Parquet file containing article titles and raw Wikitext. This stage produces a file of approximately 230 MB with around 140,000 pages.python src/10_parse.py
Output: data/10_parsed/articles.parquetStep 3: Transform and Clean DataClean the Wikitext markup from the parsed data and extract outgoing links from each article.python src/20_transform.py
Output: data/20_clean/articles.parquetExecution SummaryTo run the entire pipeline:# 1. Download the data (run once, or skip if parsing from URL)
python src/00_ingest.py

# 2. Parse XML to Parquet
python src/10_parse.py

# 3. Clean markup and extract outgoing links
python src/20_transform.py
