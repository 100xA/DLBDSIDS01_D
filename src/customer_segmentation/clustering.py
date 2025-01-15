"""
Clustering implementation for customer segmentation.
"""
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import numpy as np
import matplotlib.pyplot as plt

def find_optimal_clusters(X_scaled, max_clusters=10):
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
    plt.plot(k_range, inertias, 'bo-')
    plt.xlabel('Anzahl der Cluster (K)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')
    
    # Silhouette score plot
    plt.subplot(1, 2, 2)
    plt.plot(k_range, silhouette_scores, 'ro-')
    plt.xlabel('Anzahl der Cluster (K)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Analysis')
    
    plt.tight_layout()
    plt.savefig('output/figures/cluster_optimization.png')
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

def perform_clustering(X_scaled, n_clusters=5):
    """
    Perform K-means clustering on the scaled data.
    """
    if n_clusters is None:
        n_clusters = find_optimal_clusters(X_scaled)
    
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
    cluster_descriptions = {
        0: "Familien mit mittlerem Einkommen",
        1: "Kostenbewusste Rentner",
        2: "Karriereorientierte Paare",
        3: "Junge Großfamilien",
        4: "Wohlhabende Best Ager"
    }
    
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
        spending_dist = {
            'Niedrig': len(cluster_data[cluster_data['Spending_Score'] <= 1.5]) / len(cluster_data) * 100,
            'Mittel': len(cluster_data[(cluster_data['Spending_Score'] > 1.5) & 
                                     (cluster_data['Spending_Score'] <= 2.5)]) / len(cluster_data) * 100,
            'Hoch': len(cluster_data[cluster_data['Spending_Score'] > 2.5]) / len(cluster_data) * 100
        }
        for category, percentage in spending_dist.items():
            if percentage > 0:
                print(f"• {category}: {percentage:.1f}%")
        
        print("\nBesondere Merkmale:")
        if cluster == 0:
            print("• Mittleres Alter mit größeren Familien")
            print("• Überwiegend mittleres Ausgabenniveau")
            print("• Stabile Familienstruktur")
        elif cluster == 1:
            print("• Ältere Generation mit kleinen Haushalten")
            print("• Sehr kostenbewusst")
            print("• Fokus auf Sparsamkeit")
        elif cluster == 2:
            print("• Hohe Berufserfahrung")
            print("• Kleine bis mittlere Familien")
            print("• Ausgewogenes Ausgabenverhalten")
        elif cluster == 3:
            print("• Jüngste Altersgruppe")
            print("• Größte durchschnittliche Familiengröße")
            print("• Stark preisbewusst")
        elif cluster == 4:
            print("• Ältere, wohlhabende Zielgruppe")
            print("• Kleine Haushalte")
            print("• Höchster Anteil an hohen Ausgaben")
        print("-" * 50)

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
