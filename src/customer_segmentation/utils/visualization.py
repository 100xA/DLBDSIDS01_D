"""
Visualization utilities for customer segmentation.
"""
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

def get_cluster_descriptions(n_clusters):
    """
    Get descriptive labels for each cluster.
    """
    return {i: f"Cluster {i+1}" for i in range(n_clusters)}

def analyze_pca_components(X_scaled, feature_names):
    """
    Analyze and interpret PCA components.
    """
    pca = PCA(n_components=2)
    pca.fit(X_scaled)
    
    # Get component loadings
    loadings = pca.components_
    
    # Calculate explained variance
    explained_variance_ratio = pca.explained_variance_ratio_
    
    # Create loading matrix
    loading_matrix = pd.DataFrame(
        loadings.T,
        columns=['PC1', 'PC2'],
        index=feature_names
    )
    
    # Print explained variance
    print("\nPCA Komponenten Analyse:")
    print(f"Erklärte Varianz PC1: {explained_variance_ratio[0]*100:.1f}%")
    print(f"Erklärte Varianz PC2: {explained_variance_ratio[1]*100:.1f}%")
    print("\nKomponentenladungen:")
    print(loading_matrix)
    
    return loading_matrix, explained_variance_ratio

def create_visualizations(processed_df, kmeans_model, X_scaled):
    """
    Create comprehensive visualizations of customer segments.
    """
    n_clusters = len(np.unique(processed_df['Cluster']))
    cluster_descriptions = get_cluster_descriptions(n_clusters)
    
    # Analyze PCA components
    feature_names = ['Alter', 'Arbeitserfahrung', 'Familiengröße', 'Ausgabenniveau']
    loading_matrix, explained_variance = analyze_pca_components(X_scaled, feature_names)
    
    plt.style.use("seaborn-v0_8")
    
    # Create figure with more space for legend
    fig = plt.figure(figsize=(26, 10))
    
    # Create grid with more space for legends
    gs = plt.GridSpec(1, 2, figure=fig, width_ratios=[1.6, 1])
    
    # Create color palettes
    cluster_colors = plt.cm.tab20(np.linspace(0, 1, n_clusters))
    
    # 1. PCA Visualization
    ax1 = fig.add_subplot(gs[0, 0])
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Plot clusters
    for cluster in range(n_clusters):
        mask = processed_df['Cluster'] == cluster
        scatter = ax1.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                            alpha=0.7, 
                            color=cluster_colors[cluster],
                            s=100,
                            label=cluster_descriptions[cluster])
    
    # Plot centroids
    centroids_pca = pca.transform(kmeans_model.cluster_centers_)
    ax1.scatter(centroids_pca[:, 0], centroids_pca[:, 1], 
                c='black', marker='x', s=200, linewidths=3,
                label='Cluster Zentren')
    
    ax1.set_title('Kundengruppen (PCA Visualisierung)', fontsize=14, pad=20)
    ax1.set_xlabel(f'Erste Hauptkomponente ({explained_variance[0]*100:.1f}%)', fontsize=12)
    ax1.set_ylabel(f'Zweite Hauptkomponente ({explained_variance[1]*100:.1f}%)', fontsize=12)
    
    # Customize legend
    leg = ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # 2. Feature Importance Plot
    ax2 = fig.add_subplot(gs[0, 1])
    feature_importance = np.abs(loading_matrix['PC1']).sort_values(ascending=True)
    y_pos = np.arange(len(feature_importance))
    
    ax2.barh(y_pos, feature_importance, alpha=0.8)
    ax2.set_title('Feature Importance (PC1)', fontsize=14, pad=20)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(feature_importance.index)
    ax2.set_xlabel('Absolute Loading Value')
    
    plt.tight_layout()
    plt.savefig('output/figures/customer_segments.png',
                bbox_inches='tight',
                dpi=300)
    plt.close()
