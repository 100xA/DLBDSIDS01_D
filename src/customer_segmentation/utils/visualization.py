"""
Visualization utilities for customer segmentation.
"""
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import seaborn as sns
import pandas as pd

def get_cluster_descriptions():
    """
    Get descriptive labels for each cluster.
    """
    return {
        0: "Familien mit mittlerem Einkommen\n(Middle Income Families)",
        1: "Kostenbewusste Rentner\n(Budget-Conscious Retirees)",
        2: "Karriereorientierte Paare\n(Career-Focused Couples)",
        3: "Junge Großfamilien\n(Young Large Families)",
        4: "Wohlhabende Best Ager\n(Affluent Best Agers)"
    }

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
    
    # Interpret components
    pc1_features = loading_matrix.PC1.abs().sort_values(ascending=False)
    pc2_features = loading_matrix.PC2.abs().sort_values(ascending=False)
    
    print("\nInterpretation der Hauptkomponenten:")
    print("PC1 (wichtigste Merkmale):")
    for feat, load in pc1_features.items():
        print(f"• {feat}: {load:.3f}")
    
    print("\nPC2 (wichtigste Merkmale):")
    for feat, load in pc2_features.items():
        print(f"• {feat}: {load:.3f}")
    
    return loading_matrix, explained_variance_ratio

def create_visualizations(processed_df, kmeans_model, X_scaled):
    """
    Create comprehensive visualizations of customer segments.
    """
    n_clusters = len(np.unique(processed_df['Cluster']))
    cluster_descriptions = get_cluster_descriptions()
    
    # Analyze PCA components
    feature_names = ['Alter', 'Arbeitserfahrung', 'Familiengröße', 'Ausgabenniveau']
    loading_matrix, explained_variance = analyze_pca_components(X_scaled, feature_names)
    
    plt.style.use("seaborn-v0_8")
    
    # Erstelle eine Figur mit mehr Platz für die Legende
    fig = plt.figure(figsize=(26, 10))
    
    # Erstelle ein Grid mit mehr Platz für die Legenden
    gs = plt.GridSpec(1, 2, figure=fig, width_ratios=[1.6, 1])
    
    # Create color palettes
    cluster_colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
    spending_cmap = plt.cm.RdYlBu_r
    
    # 1. PCA Visualization
    ax1 = fig.add_subplot(gs[0, 0])
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Plot clusters
    for cluster in range(n_clusters):
        mask = processed_df['Cluster'] == cluster
        scatter = ax1.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                            alpha=0.7, c=processed_df[mask]['Spending_Score'],
                            cmap=spending_cmap, vmin=1, vmax=3, s=100,
                            label=cluster_descriptions[cluster].split('\n')[0])
    
    # Plot centroids
    centroids_pca = pca.transform(kmeans_model.cluster_centers_)
    ax1.scatter(centroids_pca[:, 0], centroids_pca[:, 1], 
                c='red', marker='x', s=200, linewidths=3, 
                label='Zentroide')
    
    # Create more informative axis labels
    pc1_features = loading_matrix.PC1.abs().nlargest(2)
    pc2_features = loading_matrix.PC2.abs().nlargest(2)
    
    pc1_label = "Erste Hauptkomponente\n"
    pc2_label = "Zweite Hauptkomponente\n"
    
    for feat, load in pc1_features.items():
        sign = '+' if loading_matrix.PC1[feat] > 0 else '-'
        pc1_label += f"{sign}{feat} "
    
    for feat, load in pc2_features.items():
        sign = '+' if loading_matrix.PC2[feat] > 0 else '-'
        pc2_label += f"{sign}{feat} "
    
    ax1.set_xlabel(pc1_label, fontsize=12, labelpad=10)
    ax1.set_ylabel(pc2_label, fontsize=12, labelpad=10)
    ax1.set_title(f'Kundengruppen im PCA-Raum\nErklärte Varianz: {explained_variance[0]*100:.1f}% / {explained_variance[1]*100:.1f}%', 
                  fontsize=14, pad=20)
    
    # Verschiebe die Legende weiter nach rechts und vergrößere den Abstand
    legend1 = ax1.legend(bbox_to_anchor=(1.15, 1.15), 
                        loc='upper left', 
                        fontsize=11,
                        title="Kundengruppen",
                        title_fontsize=12)
    
    # Füge die Colorbar mit mehr Abstand hinzu
    cbar = plt.colorbar(scatter, ax=ax1, 
                       label='Ausgabenniveau',
                       pad=0.05)
    cbar.ax.tick_labels = ['Niedrig', 'Mittel', 'Hoch']
    
    # 2. Cluster Size and Spending Distribution
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Create stacked bars for spending distribution
    spending_data = []
    for cluster in range(n_clusters):
        cluster_data = processed_df[processed_df['Cluster'] == cluster]
        low = len(cluster_data[cluster_data['Spending_Score'] <= 1.5]) / len(cluster_data) * 100
        medium = len(cluster_data[(cluster_data['Spending_Score'] > 1.5) & 
                                (cluster_data['Spending_Score'] <= 2.5)]) / len(cluster_data) * 100
        high = len(cluster_data[cluster_data['Spending_Score'] > 2.5]) / len(cluster_data) * 100
        spending_data.append([low, medium, high])
    
    spending_data = np.array(spending_data)
    bottom = np.zeros(n_clusters)
    
    # Plot stacked bars with better colors
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    for i, (height, label) in enumerate(zip(spending_data.T, ['Niedrig', 'Mittel', 'Hoch'])):
        ax2.bar(range(n_clusters), height, bottom=bottom, label=label,
                color=colors[i], alpha=0.7)
        bottom += height
    
    # Customize the plot
    ax2.set_xlabel('Kundengruppe', fontsize=12, labelpad=10)
    ax2.set_ylabel('Ausgabenverteilung (%)', fontsize=12, labelpad=10)
    ax2.set_title('Größe und Ausgabenverteilung\nder Kundengruppen', 
                  fontsize=14, pad=20)
    
    # Rotate and align the tick labels so they look better
    ax2.set_xticks(range(n_clusters))
    ax2.set_xticklabels([desc.split('\n')[0] for desc in 
                         [cluster_descriptions[i] for i in range(n_clusters)]],
                        rotation=45, ha='right', fontsize=10)
    
    # Add size annotations with more space
    cluster_sizes = processed_df['Cluster'].value_counts().sort_index()
    for i, size in enumerate(cluster_sizes):
        percentage = size/len(processed_df)*100
        ax2.text(i, 105, f'n={size}\n({percentage:.1f}%)', 
                ha='center', va='bottom', fontsize=10)
    
    # Verschiebe die Legende weiter nach rechts und vergrößere den Abstand
    legend2 = ax2.legend(title='Ausgabenniveau',
                        bbox_to_anchor=(1.15, 1.15),
                        loc='upper left',
                        fontsize=11,
                        title_fontsize=12)
    
    # Erhöhe den oberen Rand für die Beschriftungen
    ax2.set_ylim(0, 120)
    
    # Adjust layout with more space for legends
    plt.tight_layout()
    
    # Speichere die Grafik mit extra Rand für die Legenden
    plt.savefig('output/figures/customer_segments_detailed.png', 
                dpi=300, 
                bbox_inches='tight',
                pad_inches=0.5)
