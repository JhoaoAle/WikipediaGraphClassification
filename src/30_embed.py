import torch
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
import pathlib
import math
from torch.cuda.amp import autocast
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

# Configuration
IN_PARQUET = pathlib.Path("data/20_transformed/articles.parquet")
OUT_PARQUET = pathlib.Path("data/30_embedded/articles.parquet")
MODEL_NAME = "all-mpnet-base-v2"
BASE_BATCH_SIZE = 256  # Starting batch size (will be adjusted dynamically)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def initialize_model() -> SentenceTransformer:
    """Initialize SentenceTransformer model with performance optimizations"""
    torch.backends.cuda.matmul.allow_tf32 = True  # Enable TensorFloat-32
    torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner
    
    model = SentenceTransformer(MODEL_NAME)
    model = model.to(DEVICE)
    model.max_seq_length = 384  # Optimal for most Sentence-BERT models
    
    print(f"✓ Loaded model '{MODEL_NAME}' on device: {DEVICE}")
    print(f"✓ Model max sequence length: {model.max_seq_length}")
    print(f"✓ Model embedding dimension: {model.get_sentence_embedding_dimension()}")
    return model

def get_optimal_batch_size(texts: list, model: SentenceTransformer) -> int:
    """Dynamically adjust batch size based on text lengths"""
    avg_len = np.mean([len(t.split()) for t in texts])
    max_len = model.max_seq_length
    
    if avg_len > 0.6 * max_len:
        return max(BASE_BATCH_SIZE // 2, 32)
    elif avg_len > 0.3 * max_len:
        return BASE_BATCH_SIZE
    return min(BASE_BATCH_SIZE * 2, 512)

def generate_embeddings(
    model: SentenceTransformer,
    texts: list,
    batch_size: int,
    show_progress: bool = True
) -> np.ndarray:
    """Generate embeddings safely with GPU utilization"""
    emb_dim = model.get_sentence_embedding_dimension()
    embeddings = np.zeros((len(texts), emb_dim), dtype=np.float32)
    
    # Sort texts by length for more efficient batching
    text_lengths = [len(t.split()) for t in texts]
    sorted_indices = np.argsort(text_lengths)
    sorted_texts = [texts[i] for i in sorted_indices]
    
    # Warm-up GPU with a small batch
    if torch.cuda.is_available():
        warmup_text = ["warmup"] * min(8, len(texts))
        _ = model.encode(warmup_text, device=DEVICE)
    
    # Process batches serially (safe, no OOM)
    for i in tqdm(
        range(0, len(sorted_texts), batch_size),
        desc=f"Generating embeddings (batch_size={batch_size})",
        disable=not show_progress
    ):
        batch = sorted_texts[i:i + batch_size]
        with torch.no_grad(), torch.amp.autocast(device_type='cuda'): # Use AMP for better perf
            batch_embeddings = model.encode(
                batch,
                batch_size=batch_size,
                device=DEVICE,
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )
        start_idx = i
        end_idx = start_idx + len(batch_embeddings)
        embeddings[start_idx:end_idx] = batch_embeddings.cpu().numpy()
        
        # Optional: clear memory between batches
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Restore original order
    original_order_embeddings = np.zeros_like(embeddings)
    original_order_embeddings[sorted_indices] = embeddings
    return original_order_embeddings

def main():
    model = initialize_model()
    
    # Load data
    print("→ Loading parquet:", IN_PARQUET)
    df = pd.read_parquet(IN_PARQUET)
    print(f"✓ Loaded {len(df):,} rows")
    texts = df["cleaned_article_body"].astype(str).tolist()
    print(f"✓ Loaded {len(texts)} texts from {IN_PARQUET}")
    
    # Determine optimal batch size
    batch_size = get_optimal_batch_size(texts, model)
    print(f"✓ Using dynamic batch size: {batch_size}")
    
    # Generate embeddings
    embeddings = generate_embeddings(model, texts, batch_size)
    
    # Step 4: Dimensionality Reduction (optional but useful)
    svd = TruncatedSVD(n_components=200, random_state=42)
    print("⏳ Reducing dimensionality of embeddings with TruncatedSVD...")
    X_reduced = svd.fit_transform(embeddings)
    X_reduced = normalize(X_reduced)
    print(f"✓ Reduced embedding shape: {X_reduced.shape}")

    # Step 5: Construct reduced embedding DataFrame
    print("⏳ Constructing reduced embedding DataFrame...")
    reduced_df = pd.DataFrame(
        tqdm(X_reduced, desc="Building DataFrame"),
        columns=[f"emb_{i}" for i in range(X_reduced.shape[1])]
    )
    df = pd.concat([df.reset_index(drop=True), reduced_df], axis=1)

    print(f"✓ Explained variance (sum): {svd.explained_variance_ratio_.sum():.3f}")

    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PARQUET, index=False)
    print(f"✓ Saved TF-IDF features to {OUT_PARQUET}")
    print(f"✓ Wrote {OUT_PARQUET} with {len(df)} rows")




if __name__ == "__main__":
    main()