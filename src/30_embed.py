import torch
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor
import pathlib
from typing import List, Dict

IN_PARQUET = pathlib.Path("data/20_transformed/articles.parquet")
OUT_PARQUET = pathlib.Path("data/30_embedded/articles.parquet")

# Config
BATCH_SIZE = 256
MAX_LENGTH = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "bert-base-uncased"

def initialize_bert() -> tuple:
    """Initialize BERT model and tokenizer"""
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertModel.from_pretrained(MODEL_NAME).to(DEVICE)
    if DEVICE.type == 'cuda':
        model = model.half()  # Convert model to FP16
    return tokenizer, model

def preprocess_texts(texts: List[str], tokenizer: BertTokenizer) -> List[Dict]:
    """Parallel tokenization of texts"""
    def tokenize(text: str):
        try:
            return tokenizer(
                text,
                truncation=True,
                max_length=MAX_LENGTH,
                return_attention_mask=True,
                padding='max_length'
            )
        except Exception as e:
            print(f"Error tokenizing text: {e}")
            return None
    
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(
            executor.map(tokenize, texts),
            desc="Tokenizing",
            total=len(texts)
        ))
    
    return [res for res in results if res is not None]


def generate_embeddings(batch: List[Dict], tokenizer: BertTokenizer, model: BertModel) -> np.ndarray:
    import torch

    model.eval()  # Ensure model is in eval mode

    with torch.no_grad():
        # Use tokenizer batching and pad+convert to tensors directly on the GPU
        batch = tokenizer.pad(
            batch,
            padding='max_length',
            return_tensors="pt"
        )
        batch = {k: v.to(DEVICE, non_blocking=True) for k, v in batch.items()}

        outputs = model(**batch)

        # Float32 pooling for precision
        token_embeddings = outputs.last_hidden_state.float()
        mask = batch["attention_mask"].unsqueeze(-1).float()
        masked_embeddings = token_embeddings * mask
        pooled = masked_embeddings.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

        return pooled.cpu().numpy()



def main():
    # Initialize model and tokenizer
    tokenizer, model = initialize_bert()
    
    # Load data
    df = pd.read_parquet(IN_PARQUET)
    texts = df["cleaned_article_body"].astype(str).tolist()
    
    # Step 1: Parallel tokenization
    tokenized = preprocess_texts(texts, tokenizer)
    
    # Step 2: Batch processing
    embeddings = []
    for i in tqdm(range(0, len(tokenized), BATCH_SIZE), desc="Generating embeddings"):
        batch = tokenized[i:i + BATCH_SIZE]
        try:
            batch_embeddings = generate_embeddings(batch, tokenizer, model)
            embeddings.append(batch_embeddings)
        except Exception as e:
            print(f"Error processing batch {i//BATCH_SIZE}: {e}")
            # Add zero vectors for failed batches
            embeddings.append(np.zeros((len(batch), 768)))
    
    # Combine results
    all_embeddings = np.concatenate(embeddings)
    # Convert embeddings to separate columns
    embedding_cols = pd.DataFrame(all_embeddings, columns=[f"emb_{i}" for i in range(all_embeddings.shape[1])])
    df = pd.concat([df.reset_index(drop=True), embedding_cols], axis=1)
        
    # Save results
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PARQUET, index=False)
    print("✓ wrote", OUT_PARQUET, len(df), "rows")

if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    main()