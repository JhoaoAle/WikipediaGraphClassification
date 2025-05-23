import torch
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor
import pathlib

IN_PARQUET = pathlib.Path("data/20_parsed/articles.parquet")
OUT_PARQUET = pathlib.Path("data/30_embeddings/articles.parquet")

# Config
BATCH_SIZE = 256  # RTX 3060 can handle this with FP16
MAX_LENGTH = 512  # BERT's max token limit
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize BERT with FP16 (half-precision)
model = BertModel.from_pretrained("bert-base-uncased").to(DEVICE).half()
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Pre-tokenize all text (CPU-bound, parallelized)
def preprocess_texts(texts):
    with ThreadPoolExecutor() as executor:
        return list(tqdm(
            executor.map(
                lambda x: tokenizer(x, truncation=True, max_length=MAX_LENGTH),
                texts
            ),
            desc="Tokenizing",
            total=len(texts)
        ))

# GPU-accelerated embedding generation
def generate_embeddings(batch_inputs):
    with torch.no_grad():
        # Pad the batch
        batch = tokenizer.pad(
            batch_inputs,
            padding=True,
            return_tensors="pt"
        )
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        
        outputs = model(**batch)
        return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

# Main pipeline
def main():
    # Load data
    df = pd.read_parquet(IN_PARQUET)
    texts = df["cleaned_article_body"].tolist()
    
    # Step 1: Parallel tokenization (CPU)
    tokenized = preprocess_texts(texts)
    
    # Step 2: Batch processing (GPU)
    embeddings = []
    for i in tqdm(range(0, len(tokenized), BATCH_SIZE), desc="GPU Processing"):
        batch = tokenized[i:i + BATCH_SIZE]
        embeddings.append(generate_embeddings(batch))
    
    # Save results
    df["bert_embedding"] = np.concatenate(embeddings)
    df.to_parquet(OUT_PARQUET, index=False)

if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True  # Enable Tensor Cores
    main()