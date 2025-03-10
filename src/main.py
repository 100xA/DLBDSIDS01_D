from preprocessing import load_and_preprocess_data
from visualization import create_visualizations
from clustering import perform_clustering, analyze_clusters
import os

def main():
    processed_df, X_scaled = load_and_preprocess_data()

    clusters, kmeans_model = perform_clustering(X_scaled)
    processed_df['Cluster'] = clusters

    output_dir = '../data/processed'
    os.makedirs(output_dir, exist_ok=True)

    analyze_clusters(processed_df, clusters)

    create_visualizations(processed_df, kmeans_model, X_scaled)

if __name__ == "__main__":
    main()
