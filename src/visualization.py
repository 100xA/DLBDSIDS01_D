import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

def get_cluster_descriptions(n_clusters):
    return {i: f"Cluster {i+1}" for i in range(n_clusters)}

def analyze_pca_components(X_scaled, feature_names):
    pca = PCA(n_components=3)  
    pca.fit(X_scaled)
    
    loadings = pca.components_
    
    explained_variance_ratio = pca.explained_variance_ratio_
    
    loading_matrix = pd.DataFrame(
        loadings.T,
        columns=['PC1', 'PC2', 'PC3'],
        index=feature_names
    )
    
    print("\nPCA Komponenten Analyse (3D):")
    print(f"Erklärte Varianz PC1: {explained_variance_ratio[0]*100:.1f}%")
    print(f"Erklärte Varianz PC2: {explained_variance_ratio[1]*100:.1f}%")
    print(f"Erklärte Varianz PC3: {explained_variance_ratio[2]*100:.1f}%")
    print("\nKomponentenladungen:")
    print(loading_matrix)
    
    return loading_matrix, explained_variance_ratio

def create_visualizations(processed_df, kmeans_model, X_scaled):
    n_clusters = len(np.unique(processed_df['Cluster']))
    cluster_descriptions = get_cluster_descriptions(n_clusters)
    
    feature_names = ['Age', 'Work_Experience', 'Family_Size', 'Spending_Score']
    loading_matrix, explained_variance = analyze_pca_components(X_scaled, feature_names)
    
    plt.style.use("seaborn-v0_8")
    
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    cluster_colors = plt.cm.tab20(np.linspace(0, 1, n_clusters))
    
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)

    for cluster in range(n_clusters):
        mask = processed_df['Cluster'] == cluster
        ax.scatter(X_pca[mask, 0], 
                   X_pca[mask, 1], 
                   X_pca[mask, 2],
                   alpha=0.7,
                   color=cluster_colors[cluster],
                   s=100,
                   label=cluster_descriptions[cluster])
        
    centroids_pca = pca.transform(kmeans_model.cluster_centers_)
    ax.scatter(centroids_pca[:, 0], 
              centroids_pca[:, 1], 
              centroids_pca[:, 2],
              c='black', 
              marker='*',  
              s=500,  
              linewidths=2,
              label='Cluster Zentren')
    
    for i, (x, y, z) in enumerate(centroids_pca):
        ax.text(x, y, z, f' C{i+1}', fontsize=12, fontweight='bold')
    
    ax.set_xlabel(f'PC1 ({explained_variance[0]*100:.1f}%)', fontsize=12)
    ax.set_ylabel(f'PC2 ({explained_variance[1]*100:.1f}%)', fontsize=12)
    ax.set_zlabel(f'PC3 ({explained_variance[2]*100:.1f}%)', fontsize=12)
    
    ax.set_title('3D Kundengruppen Visualisierung\nPCA mit drei Hauptkomponenten', 
                fontsize=14, pad=20)
    
    ax.legend(bbox_to_anchor=(1.15, 1), loc='upper left', fontsize=10)
    
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    feature_importance = np.abs(loading_matrix[['PC1', 'PC2', 'PC3']]).mean(axis=1).sort_values(ascending=True)
    y_pos = np.arange(len(feature_importance))
    
    ax2.barh(y_pos, feature_importance, alpha=0.8)
    ax2.set_title('Feature Importance (Durchschnitt über PC1-PC3)', fontsize=14, pad=20)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(feature_importance.index)
    ax2.set_xlabel('Absolute durchschnittliche Ladung')
    

    plt.figure(fig.number)
    plt.savefig('../output/figures/customer_segments_3d.png',
                bbox_inches='tight',
                dpi=300)
    
    plt.figure(fig2.number)
    plt.savefig('../output/figures/feature_importance_3d.png',
                bbox_inches='tight',
                dpi=300)
    
    plt.close('all')
