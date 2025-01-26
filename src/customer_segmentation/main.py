"""
Main script for customer segmentation analysis.
"""
from utils.preprocessing import load_and_preprocess_data
from utils.visualization import create_visualizations
from clustering import perform_clustering, analyze_clusters, evaluate_clustering
import os

def main():
    """
    Main function to run the customer segmentation analysis.
    """
    # Load and preprocess data
    processed_df, X_scaled = load_and_preprocess_data()
    
    # Perform clustering
    clusters, kmeans_model = perform_clustering(X_scaled)
    processed_df['Cluster'] = clusters
    
    # Save segmentation results
    output_dir = 'data/processed'
    os.makedirs(output_dir, exist_ok=True)
    
    # Add cluster descriptions
    cluster_descriptions = {
        0: "Premium Familien",
        1: "Junge Sparfamilien",
        2: "Erfahrene Singles",
        3: "Sparsame Senioren",
        4: "Mittlere Ausgeber",
        5: "Sparsame Singles",
        6: "Erfahrene Ausgeber",
        7: "Junge Erfahrene",
        8: "Wohlhabende Senioren",
        9: "Große Jungfamilien"
    }
    
    processed_df['Cluster_Beschreibung'] = processed_df['Cluster'].map(cluster_descriptions)
    
    # Save to CSV
    output_file = os.path.join(output_dir, 'customer_segments.csv')
    processed_df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\nSegmentierungsergebnisse wurden gespeichert in: {output_file}")
    
    # Analyze clusters
    analyze_clusters(processed_df, clusters)
    evaluate_clustering(X_scaled, kmeans_model)
    
    # Create visualizations
    create_visualizations(processed_df, kmeans_model, X_scaled)

if __name__ == "__main__":
    main()
