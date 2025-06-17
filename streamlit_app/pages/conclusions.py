import streamlit as st

st.title("📌 Conclusions: Graph Structure and Community Analysis")

st.subheader("📊 Graph Overview")
st.markdown("""
- **Nodes**: 54,924  
- **Edges**: 868,460  
- **Average Degree**: 31.62  
- **In/Out Degree**: Symmetric (15.81 avg), indicating a balanced directed graph.  
- **Connected Components**: 233 weakly connected components  
""")

st.info("""
➤ **Insight**: The network is large, sparse, and well-connected. The average degree of ~31 indicates moderate connectivity, while the existence of only 233 weak components suggests strong global cohesion — most articles are reachable through a series of links.
""")

st.subheader("⭐ Influence and Centrality")
st.markdown("**Top Nodes by PageRank (Page et al., 1999):**")
st.code("""
Node 16883: 0.0072
Node 97:    0.0035
Node 287:   0.0028
Node 17598: 0.0023
Node 76:    0.0023
""")

st.markdown("**Top Nodes by Betweenness (Freeman, 1977):**")
st.code("""
Node 16883: 225,781,399.68
Node 97:    125,710,783.86
Node 17598:  59,804,667.54
Node 531:    48,757,107.14
Node 141:    43,191,421.55
""")

st.success("""
🔍 **Insight**: The high concentration of centrality in a few nodes suggests **hub dominance**, typical of **scale-free networks**. These influential nodes serve as **critical bridges or authorities**, likely central concepts or well-linked Wikipedia articles (e.g., broad topics like "Science" or "World War II").
""")

st.subheader("🔁 Clustering and Path Structure")
st.markdown("""
- **Average Clustering Coefficient**: 0.1981  
- **Average Shortest Path Length**: 3.37  
- **Diameter**: 9  
""")

st.info("""
🌐 **Insight**: These values indicate a **small-world network** — most articles are only ~3.4 clicks apart on average, and clustering is present but not overly strong. This supports efficient navigation and localized grouping of related content.
""")

st.subheader("🧱 Community Detection")
st.markdown("**Louvain Algorithm:**")
st.code("Communities: 1923\nModularity: 0.6134")

st.markdown("**Leiden Algorithm:**")
st.code("Communities: 1928\nModularity: 0.6198")

st.success("""
🧠 **Insight**: Both Louvain and Leiden uncovered nearly 2,000 distinct communities, with **high modularity scores (>0.6)**. This confirms the graph contains **well-separated thematic clusters**, validating Wikipedia’s topical structure — distinct regions for politics, science, sports, etc.
""")

st.subheader("📊 Clustering Comparison Metrics")

st.markdown("**Louvain vs Leiden:**")
st.code("""
Adjusted Rand Index (ARI):     0.6086
Normalized Mutual Info (NMI):  0.7204
Homogeneity:                   0.7442
Completeness:                  0.6981
V-Measure:                     0.7204
""")

st.success("""
🟢 **Insight**: Louvain and Leiden yield **very consistent clusterings**, as shown by high ARI and NMI. This agreement suggests that community structure in the graph is **stable** across algorithms and not overly sensitive to method choice.
""")

st.markdown("**Louvain vs KMeans:**")
st.code("""
Adjusted Rand Index (ARI):     0.0045
Normalized Mutual Info (NMI):  0.3774
Homogeneity:                   0.7651
Completeness:                  0.2505
V-Measure:                     0.3774
""")

st.markdown("**Leiden vs KMeans:**")
st.code("""
Adjusted Rand Index (ARI):     0.0058
Normalized Mutual Info (NMI):  0.3963
Homogeneity:                   0.2674
Completeness:                  0.7657
V-Measure:                     0.3963
""")

st.warning("""
🔺 **Insight**: KMeans clusters diverge **significantly** from graph-based communities (Louvain/Leiden).  
- **Very low ARI (~0.005)** suggests near-random overlap — KMeans partitions the space in a way **incompatible with graph topology**.
- **High homogeneity but low completeness** implies that individual KMeans clusters tend to fall within single communities, but many nodes from the same community are split across clusters.

This is expected, as KMeans operates on **vector space geometry**, not structural connectivity.
""")

st.subheader("📈 KMeans Cluster Statistics")

st.markdown("""
- **Total Clusters**: 4,500  
- **Cluster Size Stats**:
    - Mean: 60.09
    - Std: 39.48
    - Min: 1
    - Max: 443
    - Median (50%): 51
    - 25%–75%: [33, 78]
""")

st.markdown("**Top 5 Largest Clusters:**")
st.code("""
Cluster 31.0:   443 points (0.16%)
Cluster 51.0:   398 points (0.15%)
Cluster 73.0:   363 points (0.13%)
Cluster 246.0:  359 points (0.13%)
Cluster 1973.0: 336 points (0.12%)
""")

st.info("""
🔍 **Insight**: KMeans produced a large number of small-to-medium clusters.  
- **Low variance** in size (~mean=60, std=39) indicates **relatively balanced partitioning**.
- The largest cluster contains only **0.16%** of points — there are **no dominant clusters**, unlike typical real-world graphs.

This suggests that KMeans, in this case, was configured to enforce **uniformity** over semantic cohesion — useful for sampling, but possibly misaligned with underlying content structure.
""")

st.subheader("🧩 Final Remarks")
st.markdown("""
- The graph exhibits real-world behaviors seen in complex systems: **scale-free**, **modular**, and **small-world**.
- Community detection via Louvain/Leiden aligns well and supports a **semantically structured corpus**.
""")

st.warning("""
🚨 **Caution**: KMeans and graph community detection may serve **different purposes**:
- **KMeans** is used when working in a vector space (e.g., embeddings).
- **Louvain/Leiden** are used when working with structural relations (e.g., links/citations).
Combining both offers **richer multilayer understanding**, but they are not to be compared against each other to see which is better, but rather, how they complement each other.
""")
