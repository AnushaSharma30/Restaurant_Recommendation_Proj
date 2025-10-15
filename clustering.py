import pandas as pd
import numpy as np
import hdbscan
import gc

def perform_clustering():
    """
    Loads the data with embeddings, aggregates them by user,
    and performs HDBSCAN clustering to find 'taste tribes'.
    """
    print("--- Final Step: Clustering Users into 'Taste Tribes' ---")
    
    try:
        df = pd.read_csv('data_with_embeddings.csv')
        print("Successfully loaded 'data_with_embeddings.csv'.")
    except FileNotFoundError:
        print("ERROR: 'data_with_embeddings.csv' not found. Please run the full script first.")
        return

    # --- 1. Aggregate Taste DNA by User ---
    # We want to cluster users, so we need a single 'Taste DNA' profile per user.
    # A simple average is a robust way to do this.
    print("Aggregating taste DNA embeddings for each unique user...")
    embedding_cols = [col for col in df.columns if 'taste_dna' in col]
    user_dna_profiles = df.groupby('user_id')[embedding_cols].mean().reset_index()

    print(f"Created {len(user_dna_profiles)} unique user profiles.")

    # --- 2. Perform HDBSCAN Clustering ---
    # Extract just the embedding data for the clustering algorithm
    data_to_cluster = user_dna_profiles[embedding_cols].values
    
    print("Performing HDBSCAN clustering... (This may take a few minutes)")
    # HDBSCAN parameters:
    # min_cluster_size: The smallest group of users we'll consider a 'tribe'.
    # min_samples: How conservative to be when clustering. Higher = more noise points.
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=100,
        min_samples=15,
        metric='euclidean',
        core_dist_n_jobs=-1 # Use all available CPU cores
    )
    
    clusters = clusterer.fit_predict(data_to_cluster)
    
    # --- 3. Analyze and Save the Results ---
    print("\n--- Clustering Results ---")
    
    # Add the cluster labels back to our user profiles
    user_dna_profiles['cluster_id'] = clusters
    
    num_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    num_noise_points = np.sum(clusters == -1)
    
    print(f"Number of 'Taste Tribes' discovered: {num_clusters}")
    print(f"Number of users not assigned to any tribe (noise): {num_noise_points}")
    
    print("\nCluster Distribution:")
    print(user_dna_profiles['cluster_id'].value_counts())
    
    # Save the final results to a new CSV file
    OUTPUT_FILENAME = 'user_clusters.csv'
    user_dna_profiles[['user_id', 'cluster_id']].to_csv(OUTPUT_FILENAME, index=False)
    
    print(f"\n✅ Clustering complete. User cluster assignments saved to '{OUTPUT_FILENAME}'.")
    print("You can now use these cluster IDs to make recommendations.")

if __name__ == '__main__':
    perform_clustering()