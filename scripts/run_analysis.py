import os
import sys
import time
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm

from pga_pipeline.config import NORMALIZATION_METHOD, CLUSTER_COLORS
from pga_pipeline.data_loader import load_all_scans, load_pointwise_variance
from pga_pipeline.geometry import normalize_points, compute_frechet_mean, compute_euclidean_mean, compute_pga, compute_frechet_mean_per_cluster
from pga_pipeline.clustering import cluster_point_cloud_spectral, assess_clustering_consistency
from pga_pipeline.stats import associate_variance_with_clusters, perform_statistical_tests
from pga_pipeline.visualization import (
    plot_variance_distributions_single_figure,
    plot_clusters_visualization,
    plot_average_variance,
    plot_pga_results,
    plot_means
)

logging.basicConfig(filename='outputs/local_pga_validation.log',
                    level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

def main():
    try:
        # Create results directory
        os.makedirs("results", exist_ok=True)
        
        # NOTE: Modify these paths as needed, or accept as arguments
        scans_folder_path = "data/Freeform Landmarks"  
        variance_file_path = "data/Freeform Pointwise Variance.csv"  

        print("Loading all scans...")
        all_scans = load_all_scans(scans_folder_path, has_header=True, x_col='X', y_col='Y', z_col='Z', max_scans=None)
        if len(all_scans) == 0:
            logging.error("No scans loaded. Exiting script.")
            sys.exit(1)

        num_landmarks = all_scans[0].shape[0]
        for idx, scan in enumerate(all_scans, start=1):
            if scan.shape[0] != num_landmarks:
                logging.error(f"Scan {idx} has {scan.shape[0]} landmarks, expected {num_landmarks}.")
                sys.exit(1)

        k_clusters = 10
        n_neighbors = 25

        scans_cluster_labels = []
        combined_variance_data = []
        cluster_pga_stats = {}

        print("Loading pointwise variance data...")
        variance_data = load_pointwise_variance(variance_file_path, 
                                                variance_columns=['Pointwise Variance X', 'Pointwise Variance Y', 'Pointwise Variance Z'], 
                                                expected_landmarks=num_landmarks,
                                                variance_scale_factor=0.1)

        for scan_idx, scan in enumerate(tqdm(all_scans, desc="Processing scans")):
            start_time = time.time()
            print(f"\nProcessing Scan {scan_idx + 1}/{len(all_scans)}...")
            normalized_scan = normalize_points(scan, method=NORMALIZATION_METHOD)
            
            print("Clustering the scan...")
            cluster_labels = cluster_point_cloud_spectral(normalized_scan, n_clusters=k_clusters, n_neighbors=n_neighbors)
            scans_cluster_labels.append(cluster_labels)
            
            print("Associating variance with clusters...")
            cluster_variance_stats = associate_variance_with_clusters(cluster_labels, variance_data)
            
            print("Performing statistical tests...")
            statistical_results = perform_statistical_tests(cluster_variance_stats)
            print(f"Statistical Results: {statistical_results}")
            
            print("Validating variances...")
            for cluster_id, stats in cluster_variance_stats.items():
                logging.info(f"Scan {scan_idx + 1}, Cluster {cluster_id}:")
                logging.info(f"Mean Variance X: {stats['mean_variance'][0]:.6f}, Y: {stats['mean_variance'][1]:.6f}, Z: {stats['mean_variance'][2]:.6f}")
                logging.info(f"Std Dev Variance X: {stats['std_variance'][0]:.6f}, Y: {stats['std_variance'][1]:.6f}, Z: {stats['std_variance'][2]:.6f}")
                
            print("Performing Principal Geodesic Analysis (PGA) for each cluster...")
            for cluster_id, stats in cluster_variance_stats.items():
                cluster_points = normalized_scan[cluster_labels == cluster_id]
                if NORMALIZATION_METHOD == 'minmax':
                    cluster_mean = compute_frechet_mean(cluster_points)
                else:
                    cluster_mean = compute_euclidean_mean(cluster_points)
                pga_results = compute_pga(cluster_points, cluster_mean, method=NORMALIZATION_METHOD, n_components=3)
                cluster_pga_stats[cluster_id] = {
                    'principal_directions': pga_results['principal_directions'],
                    'eigenvalues': pga_results['eigenvalues'],
                    'mean': cluster_mean
                }
                
            print("Plotting cluster visualizations...")
            plot_clusters_visualization(normalized_scan, cluster_labels, scan_idx + 1, CLUSTER_COLORS, output_folder='results/cluster_visualizations')
            
            for cluster_id, stats in cluster_variance_stats.items():
                combined_variance_data.append({
                    'Scan': scan_idx + 1,
                    'Cluster': cluster_id,
                    'Variance_X': stats['mean_variance'][0],
                    'Variance_Y': stats['mean_variance'][1],
                    'Variance_Z': stats['mean_variance'][2],
                    'Std_Deviation_X': stats['std_variance'][0],
                    'Std_Deviation_Y': stats['std_variance'][1],
                    'Std_Deviation_Z': stats['std_variance'][2]
                })
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Finished processing Scan {scan_idx + 1} in {elapsed_time:.2f} seconds.")

        combined_variance_df = pd.DataFrame(combined_variance_data)
        n_clusters = len(np.unique(scans_cluster_labels[0]))
        
        print("\nComputing means for each cluster across all scans...")
        means = compute_frechet_mean_per_cluster(all_scans, scans_cluster_labels, n_clusters, method=NORMALIZATION_METHOD)
        mean_type = 'Fréchet' if NORMALIZATION_METHOD == 'unit_sphere' else 'Euclidean'
        means_df = pd.DataFrame({
            'Cluster': np.arange(n_clusters),
            'Mean_X': means[:, 0],
            'Mean_Y': means[:, 1],
            'Mean_Z': means[:, 2]
        })
        print(f"\n{mean_type} Means for Each Cluster:")
        print(means_df)
        
        print("\nAssessing clustering consistency across scans...")
        consistency_metrics = assess_clustering_consistency(scans_cluster_labels)
        print(f"Clustering Consistency Metrics: {consistency_metrics}")
        
        print("\nSaving combined variance data to CSV...")
        combined_variance_df.to_csv('results/combined_within_cluster_variances.csv', index=False)
        print("Variance data saved to 'results/combined_within_cluster_variances.csv'.")
        
        print("\nGenerating variance distribution plots...")
        plot_variance_distributions_single_figure(cluster_variance_stats, n_clusters, CLUSTER_COLORS, output_folder='results/variance_distributions')
        
        print(f"\nGenerating {mean_type} means visualizations...")
        plot_means(means, method=mean_type, n_clusters=n_clusters, cluster_colors=CLUSTER_COLORS, output_folder='results/means_visualizations')
        
        print(f"\nGenerating {mean_type} average variance visualizations...")
        first_scan = all_scans[0]
        first_scan_labels = scans_cluster_labels[0]
        plot_average_variance(first_scan, first_scan_labels, cluster_variance_stats, method=mean_type, n_clusters=n_clusters, cluster_colors=CLUSTER_COLORS, output_folder='outputs/average_variance_plots')
        
        print("\nGenerating Principal Geodesic Analysis (PGA) visualizations...")
        plot_pga_results(cluster_pga_stats, output_folder='outputs/pga_results')
        
        print("\nLocal PGA Validation Script Completed Successfully.")
    except Exception as e:
        logging.error(f"An unexpected error occurred in the main execution: {e}")
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
