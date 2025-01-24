"""
Clustering implementation for customer segmentation.
"""
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import numpy as np
import matplotlib.pyplot as plt

def find_optimal_clusters(X_scaled, max_clusters=15):
    """
    Find the optimal number of clusters using elbow method and silhouette analysis.
    """
    inertias = []
    silhouette_scores = []
    k_range = range(2, max_clusters + 1)
    
    print("\nAnalyzing optimal number of clusters...")
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
        print(f"K={k}: Silhouette Score = {silhouette_scores[-1]:.3f}")
    
    # Plot elbow curve
    plt.figure(figsize=(12, 5))
    
    # Inertia plot
    plt.subplot(1, 2, 1)
    plt.plot(k_range, inertias, 'o-', color='blue', linewidth=2, markersize=6)
    plt.xlabel('Anzahl der Cluster (K)', fontsize=10)
    plt.ylabel('Inertia', fontsize=10)
    plt.title('Elbow Method', pad=20, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.2)
    plt.xticks(range(2, 16))  # Show all K values from 2 to 15
    
    # Add annotation for the elbow point
    elbow_point = np.argmin(np.abs(np.diff(np.diff(inertias)))) + 2
    plt.annotate('Elbow Point', 
                xy=(elbow_point, inertias[elbow_point-2]),
                xytext=(elbow_point+1, inertias[elbow_point-2]*1.1),
                arrowprops=dict(facecolor='red', shrink=0.05),
                color='red')
    
    # Silhouette score plot
    plt.subplot(1, 2, 2)
    plt.plot(k_range, silhouette_scores, 'o-', color='blue', linewidth=2, markersize=6)
    plt.xlabel('Anzahl der Cluster (K)', fontsize=10)
    plt.ylabel('Silhouette Score', fontsize=10)
    plt.title('Silhouette Analysis', pad=20, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.2)
    plt.xticks(range(2, 16))  # Show all K values from 2 to 15
    
    # Add annotation for best silhouette score
    best_k = np.argmax(silhouette_scores) + 2
    plt.annotate(f'Bester Score', 
                xy=(best_k, max(silhouette_scores)),
                xytext=(best_k+1, max(silhouette_scores)*0.95),
                arrowprops=dict(facecolor='red', shrink=0.05),
                color='red')
    
    plt.tight_layout()
    plt.savefig('output/figures/cluster_optimization.png', 
                bbox_inches='tight',
                dpi=300)
    plt.close()
    
    # Find optimal K
    # Calculate the rate of change in inertia
    inertia_changes = np.diff(inertias)
    rate_of_change = np.diff(inertia_changes)
    
    # Find the elbow point where the rate of change stabilizes
    elbow_k = np.argmin(np.abs(rate_of_change)) + 3  # +3 because we started from k=2
    
    # Find K with best silhouette score
    best_silhouette_k = np.argmax(silhouette_scores) + 2  # +2 because we started from k=2
    
    print(f"\nElbow method suggests {elbow_k} clusters")
    print(f"Best silhouette score with {best_silhouette_k} clusters")
    
    return best_silhouette_k

def perform_clustering(X_scaled, n_clusters=10):
    """
    Perform K-means clustering on the scaled data.
    """
    # Always run the analysis to show the visualization
    suggested_k = find_optimal_clusters(X_scaled)
    
    # Use the provided n_clusters or the suggested one if none provided
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
    """
    Analyze the characteristics of each cluster.
    """
    n_clusters = len(np.unique(clusters))
    
    # Generate generic descriptions for any number of clusters
    cluster_descriptions = {}
    for i in range(n_clusters):
        cluster_descriptions[i] = f"Cluster {i+1}"
    
    print("\nDetaillierte Cluster-Analyse:\n")
    for cluster in range(n_clusters):
        cluster_data = processed_df[processed_df['Cluster'] == cluster]
        print(f"\n{cluster_descriptions[cluster]}:")
        print(f"Anzahl Kunden: {len(cluster_data)} ({len(cluster_data)/len(processed_df)*100:.1f}%)\n")
        
        print("Demografische Merkmale:")
        print(f"• Durchschnittsalter: {cluster_data['Age'].mean():.1f} Jahre")
        print(f"• Arbeitserfahrung: {cluster_data['Work_Experience'].mean():.1f} Jahre")
        print(f"• Durchschnittliche Familiengröße: {cluster_data['Family_Size'].mean():.1f} Personen")
        
        print("\nAusgabenverhalten:")
        spending_dist = cluster_data['Spending_Score'].value_counts(normalize=True) * 100
        for score, percentage in spending_dist.items():
            spending_level = {1: 'Niedrig', 2: 'Mittel', 3: 'Hoch'}[score]
            print(f"• {spending_level}: {percentage:.1f}%")
        
        # Calculate and print cluster characteristics
        features = ['Age', 'Work_Experience', 'Family_Size', 'Spending_Score']
        cluster_means = cluster_data[features].mean()
        cluster_stds = cluster_data[features].std()
        
        # Identify distinctive features (those that deviate significantly from overall mean)
        all_means = processed_df[features].mean()
        all_stds = processed_df[features].std()
        
        distinctive_features = []
        for feature in features:
            z_score = (cluster_means[feature] - all_means[feature]) / all_stds[feature]
            if abs(z_score) > 0.5:  # Threshold for considering a feature distinctive
                direction = "überdurchschnittlich" if z_score > 0 else "unterdurchschnittlich"
                distinctive_features.append(f"{feature} ({direction})")
        
        if distinctive_features:
            print("\nCharakteristische Merkmale:")
            for feature in distinctive_features:
                print(f"• {feature}")
        
        print("\n" + "-"*50)

def evaluate_clustering(X_scaled, kmeans_model):
    """
    Evaluate the effectiveness of K-means clustering.
    """
    labels = kmeans_model.labels_
    
    # Calculate clustering metrics
    silhouette = silhouette_score(X_scaled, labels)
    calinski = calinski_harabasz_score(X_scaled, labels)
    davies = davies_bouldin_score(X_scaled, labels)
    
    # Calculate cluster separation
    centroids = kmeans_model.cluster_centers_
    centroid_distances = []
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            dist = np.linalg.norm(centroids[i] - centroids[j])
            centroid_distances.append(dist)
    avg_separation = np.mean(centroid_distances)
    
    print("\nCluster-Validierungsmetriken:")
    print(f"• Silhouette Score: {silhouette:.3f} (höher ist besser, max 1.0)")
    print(f"• Calinski-Harabasz Index: {calinski:.1f} (höher ist besser)")
    print(f"• Davies-Bouldin Index: {davies:.3f} (niedriger ist besser)")
    print(f"• Durchschnittliche Zentroid-Separation: {avg_separation:.3f}")
    
    # Compare with random baseline
    from sklearn.utils import shuffle
    random_labels = shuffle(labels)
    random_silhouette = silhouette_score(X_scaled, random_labels)
    improvement = ((silhouette - random_silhouette) / abs(random_silhouette)) * 100
    
    print(f"\nVergleich mit Zufallsbaseline:")
    print(f"• Zufälliges Clustering Silhouette Score: {random_silhouette:.3f}")
    print(f"• Verbesserung gegenüber Zufall: {improvement:.1f}%")
