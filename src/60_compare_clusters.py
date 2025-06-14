import pandas as pd
import pathlib
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_score,
    completeness_score,
    v_measure_score
)

IN_PARQUET = pathlib.Path("data/50_clustered/articles.parquet")

def compare_clusterings(df, col1="louvain_community", col2="kmeans_cluster"):
    # Filtrar NaNs
    df_clean = df.dropna(subset=[col1, col2])
    print(f"✅ Filas válidas para comparar: {len(df_clean):,} de {len(df):,}")

    # Convertir a enteros (pueden venir como floats si hubo NaNs antes)
    labels1 = df_clean[col1].astype(int)
    labels2 = df_clean[col2].astype(int)

    # Calcular métricas
    ari = adjusted_rand_score(labels1, labels2)
    nmi = normalized_mutual_info_score(labels1, labels2)
    homogeneity = homogeneity_score(labels1, labels2)
    completeness = completeness_score(labels1, labels2)
    v_measure = v_measure_score(labels1, labels2)

    # Mostrar resultados
    print("\n🔍 Comparación de Clustering:")
    print(f"Adjusted Rand Index (ARI):       {ari:.4f}")
    print(f"Normalized Mutual Information:   {nmi:.4f}")
    print(f"Homogeneity:                     {homogeneity:.4f}")
    print(f"Completeness:                    {completeness:.4f}")
    print(f"V-Measure:                       {v_measure:.4f}")

    return {
        "ari": ari,
        "nmi": nmi,
        "homogeneity": homogeneity,
        "completeness": completeness,
        "v_measure": v_measure
    }

def main():
    
    df = pd.read_parquet(IN_PARQUET)
    compare_clusterings(df)

if __name__ == "__main__":
    main()
