"""
Main script for customer segmentation analysis.
"""
from utils.preprocessing import load_and_preprocess_data
from utils.visualization import create_visualizations
from clustering import perform_clustering, analyze_clusters, evaluate_clustering

def main():
    """
    Main function to run the customer segmentation analysis.
    """
    # Load and preprocess data
    processed_df, X_scaled = load_and_preprocess_data()
    
    # Perform clustering
    clusters, kmeans_model = perform_clustering(X_scaled)
    processed_df['Cluster'] = clusters
    
    # Analyze clusters
    analyze_clusters(processed_df, clusters)
    evaluate_clustering(X_scaled, kmeans_model)
    
    # Create visualizations
    create_visualizations(processed_df, kmeans_model, X_scaled)

if __name__ == "__main__":
    main()
