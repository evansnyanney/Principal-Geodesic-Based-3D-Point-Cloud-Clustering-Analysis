# Importing Libraries
import numpy as np
import pandas as pd
import os
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import sys

# Configure logging
logging.basicConfig(
    filename="clustering_variance_analysis.log",
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s",
)

# ============================
# ===== HELPER FUNCTIONS =====
# ============================


def load_csv(file_path, has_header=True, x_col="X", y_col="Y", z_col="Z"):
    """
    Load point cloud data from a CSV file.
    """
    try:
        if has_header:
            df = pd.read_csv(file_path)
            points = df[[x_col, y_col, z_col]].values.astype(np.float32)
        else:
            df = pd.read_csv(file_path, header=None)
            points = df.iloc[:, [x_col, y_col, z_col]].values.astype(np.float32)
        return points
    except Exception as e:
        logging.error(f"Error loading CSV file '{file_path}': {e}")
        sys.exit(1)


def cluster_point_cloud_spectral(points, n_clusters=5, n_neighbors=10):
    """
    Cluster the point cloud using Spectral Clustering with a nearest neighbors affinity matrix.
    """
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(points)
    connectivity = nbrs.kneighbors_graph(points, mode="connectivity")
    affinity_matrix = 0.5 * (connectivity + connectivity.T)
    spectral = SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=42,
    )
    cluster_labels = spectral.fit_predict(affinity_matrix)
    return cluster_labels


# ============================
# ====== MAIN EXECUTION ======
# ============================


def main():
    """
    Main function to execute the clustering and variance analysis.
    """
    # === Step 1: Define the paths ===
    scans_folder_path = "C:/Users/en596624/OneDrive - Ohio University/Desktop/landmarks3D/Half Ball Landmarks"  # Replace with your scans folder path
    variance_file_path = "C:/Users/en596624/OneDrive - Ohio University/Desktop/landmarks3D/Half Ball Pointwise Variance.csv"  # Replace with your variance file path

    # === Step 2: Load one scan for clustering ===
    print("Loading a scan for clustering...")
    scan_files = sorted(
        [f for f in os.listdir(scans_folder_path) if f.endswith(".csv")]
    )
    if not scan_files:
        logging.error("No scan files found in the specified folder.")
        sys.exit(1)

    # For simplicity, use the first scan
    scan_file_path = os.path.join(scans_folder_path, scan_files[0])
    points = load_csv(scan_file_path, has_header=True, x_col="X", y_col="Y", z_col="Z")
    num_landmarks = points.shape[0]

    # === Step 3: Load the variance data ===
    print("Loading pointwise variance data...")
    variance_df = pd.read_csv(variance_file_path)
    variance_columns = [
        "Pointwise Variance X",
        "Pointwise Variance Y",
        "Pointwise Variance Z",
    ]
    if not all(col in variance_df.columns for col in variance_columns):
        logging.error("Variance file does not contain the required variance columns.")
        sys.exit(1)
    variance_data = variance_df[variance_columns].values.astype(np.float32)

    if variance_data.shape[0] != num_landmarks:
        logging.error("Number of variance entries does not match number of landmarks.")
        sys.exit(1)

    # === Step 4: Perform clustering ===
    print("Clustering the landmarks...")
    k_clusters = 5
    n_neighbors = 10
    cluster_labels = cluster_point_cloud_spectral(
        points, n_clusters=k_clusters, n_neighbors=n_neighbors
    )

    # === Step 5: Associate variance data with clusters ===
    print("Associating variance data with clusters...")
    variance_df["Cluster"] = cluster_labels

    # === Step 6: Calculate mean and std of variance per cluster ===
    print("Calculating mean and standard deviation of variance per cluster...")
    cluster_stats = variance_df.groupby("Cluster").agg(
        {
            "Pointwise Variance X": ["mean", "std"],
            "Pointwise Variance Y": ["mean", "std"],
            "Pointwise Variance Z": ["mean", "std"],
        }
    )
    print(cluster_stats)

    # === Step 7: Create overlaid histograms ===
    print("Creating overlaid histograms...")
    cluster_colors = sns.color_palette("hls", k_clusters)
    variance_components = [
        "Pointwise Variance X",
        "Pointwise Variance Y",
        "Pointwise Variance Z",
    ]

    for variance_component in variance_components:
        plt.figure(figsize=(10, 6))
        for cluster_id in sorted(variance_df["Cluster"].unique()):
            # Subset data for the cluster
            cluster_data = variance_df[variance_df["Cluster"] == cluster_id]
            # Plot histogram
            sns.histplot(
                cluster_data[variance_component],
                bins=30,
                kde=False,
                color=cluster_colors[cluster_id],
                label=f"Cluster {cluster_id}",
                alpha=0.5,
            )
        plt.title(f"Distribution of {variance_component} Across Clusters")
        plt.xlabel(f"{variance_component}")
        plt.ylabel("Frequency")
        plt.legend()
        plt.show()

    # === Step 8: Save cluster assignments if needed ===
    # variance_df.to_csv('variance_with_clusters.csv', index=False)
    # print("Variance data with cluster assignments saved to 'variance_with_clusters.csv'.")

    print("Clustering and variance analysis completed successfully.")


if __name__ == "__main__":
    main()
