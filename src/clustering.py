"""
Clustering implementation for customer segmentation.
"""
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.utils import shuffle

def find_optimal_clusters(X_scaled, max_clusters=15):
    """
    Find the optimal number of clusters using a second-derivative elbow method
    and silhouette analysis. Returns the best silhouette k by default.
    """
    inertias = []
    silhouette_scores = []
    k_range = range(2, max_clusters + 1)
    
    print("\nAnalyzing optimal number of clusters...")
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
        score = silhouette_score(X_scaled, kmeans.labels_)
        silhouette_scores.append(score)
        print(f"K={k}: Silhouette Score = {score:.3f}")
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(k_range, inertias, 'o-', linewidth=2, markersize=6)
    plt.xlabel('Anzahl der Cluster (K)', fontsize=10)
    plt.ylabel('Inertia', fontsize=10)
    plt.title('Elbow Method', pad=20, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.2)
    plt.xticks(range(2, max_clusters + 1))
    
    rate_of_change = np.diff(np.diff(inertias))
    elbow_k = np.argmin(np.abs(rate_of_change)) + 2
    plt.annotate('Elbow Point', 
                 xy=(elbow_k, inertias[elbow_k - 2]),
                 xytext=(elbow_k + 1, inertias[elbow_k - 2] * 1.1),
                 arrowprops=dict(facecolor='red', shrink=0.05),
                 color='red')
    

    plt.subplot(1, 2, 2)
    plt.plot(k_range, silhouette_scores, 'o-', linewidth=2, markersize=6)
    plt.xlabel('Anzahl der Cluster (K)', fontsize=10)
    plt.ylabel('Silhouette Score', fontsize=10)
    plt.title('Silhouette Analysis', pad=20, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.2)
    plt.xticks(range(2, max_clusters + 1))  
    
    best_silhouette_k = np.argmax(silhouette_scores) + 2
    plt.annotate('Bester Score', 
                 xy=(best_silhouette_k, max(silhouette_scores)),
                 xytext=(best_silhouette_k + 1, max(silhouette_scores)*0.95),
                 arrowprops=dict(facecolor='red', shrink=0.05),
                 color='red')
    
    plt.tight_layout()
    plt.savefig('../output/figures/cluster_optimization.png', 
                bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"\nElbow method suggests {elbow_k} clusters")
    print(f"Best silhouette score with {best_silhouette_k} clusters")
    
    return best_silhouette_k

def perform_clustering(X_scaled, n_clusters=10):
    suggested_k = find_optimal_clusters(X_scaled)
    
    if n_clusters is None:
        n_clusters = suggested_k  

    print(f"\nUsing {n_clusters} clusters for the analysis")
    
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )
    clusters = kmeans.fit_predict(X_scaled)
    return clusters, kmeans

def analyze_clusters(processed_df, clusters):
    n_clusters = len(np.unique(clusters))
    
    cluster_descriptions = {i: f"Cluster {i+1}" for i in range(n_clusters)}
    
    print("\nDetaillierte Cluster-Analyse:\n")
    for cluster_idx in range(n_clusters):
        cluster_data = processed_df[processed_df['Cluster'] == cluster_idx]
        print(f"\n{cluster_descriptions[cluster_idx]}:")
        print(f"Anzahl Kunden: {len(cluster_data)} "
              f"({len(cluster_data)/len(processed_df)*100:.1f}%)\n")
        
        print("Demografische Merkmale:")
        print(f"• Durchschnittsalter: {cluster_data['Age'].mean():.1f} Jahre")
        print(f"• Arbeitserfahrung: {cluster_data['Work_Experience'].mean():.1f} Jahre")
        print(f"• Durchschnittliche Familiengröße: {cluster_data['Family_Size'].mean():.1f} Personen")
        
        print("\nAusgabenverhalten:")
        spending_dist = cluster_data['Spending_Score'].value_counts(normalize=True) * 100
        for score, percentage in spending_dist.items():
            spending_level = {1: 'Niedrig', 2: 'Mittel', 3: 'Hoch'}.get(score, 'Unbekannt')
            print(f"• {spending_level}: {percentage:.1f}%")

        features = ['Age', 'Work_Experience', 'Family_Size', 'Spending_Score']
        cluster_means = cluster_data[features].mean()
        all_means = processed_df[features].mean()
        all_stds = processed_df[features].std()
        
        distinctive_features = []
        for feature in features:
            z_score = (cluster_means[feature] - all_means[feature]) / all_stds[feature]
            # Threshold for "distinctive" can be tuned:
            if abs(z_score) > 0.5:
                direction = "überdurchschnittlich" if z_score > 0 else "unterdurchschnittlich"
                distinctive_features.append(f"{feature} ({direction})")
        
        if distinctive_features:
            print("\nCharakteristische Merkmale:")
            for feat in distinctive_features:
                print(f"• {feat}")
        
        print("\n" + "-"*50)

