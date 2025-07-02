import pandas as pd
import networkx as nx
from community import community_louvain
import matplotlib.pyplot as plt
import itertools

def build_climate_nature_network(climate_df: pd.DataFrame, es_df: pd.DataFrame, id_col: str = "firm_id"):
    """
    Constructs a weighted network connecting climate risk drivers and ecosystem service dependencies.

    Parameters:
    - climate_df: DataFrame with firm_id + climate risk columns (e.g., 'floods', 'heat_stress')
    - es_df: DataFrame with firm_id + ES dependency columns (e.g., 'surface_water', 'flood_protection')
    - id_col: name of firm identifier column

    Returns:
    - G: NetworkX Graph object with communities detected
    """
    # Merge firm-level data
    merged = pd.merge(climate_df, es_df, on=id_col, how='inner')
    merged = merged.dropna()

    # Extract node labels
    climate_nodes = [c for c in climate_df.columns if c != id_col]
    es_nodes = [e for e in es_df.columns if e != id_col]
    all_nodes = climate_nodes + es_nodes

    # Create graph
    G = nx.Graph()
    G.add_nodes_from(all_nodes)

    # Compute edges: pairwise average product of scores
    for node_i, node_j in itertools.combinations(all_nodes, 2):
        product_avg = (merged[node_i] * merged[node_j]).mean()
        if product_avg > 0:
            G.add_edge(node_i, node_j, weight=product_avg)

    # Community detection
    partition = community_louvain.best_partition(G, weight='weight')
    nx.set_node_attributes(G, partition, 'community')

    # Optional plot
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42, k=0.3)
    colors = [partition[n] for n in G.nodes()]
    nx.draw_networkx(G, pos, node_color=colors, with_labels=True, edge_color='gray', node_size=600, cmap=plt.cm.Set3)
    plt.title("Climate–Nature Risk Network (Louvain Community Detection)")
    plt.axis("off")
    plt.show()

    return G




# Sample firm-level scores
climate_scores = pd.DataFrame({
    'firm_id': ['F1', 'F2', 'F3'],
    'floods': [0.6, 0.8, 0.2],
    'heat_stress': [0.7, 0.5, 0.4]
})

es_scores = pd.DataFrame({
    'firm_id': ['F1', 'F2', 'F3'],
    'surface_water': [0.3, 0.9, 0.5],
    'flood_protection': [0.6, 0.7, 0.4]
})

# Build and plot network
G = build_climate_nature_network(climate_scores, es_scores)
