import numpy as np
from scipy.stats import kruskal

def associate_variance_with_clusters(cluster_labels, variance_data):
    unique_clusters = np.unique(cluster_labels)
    cluster_variance_stats = {}
    for cluster in unique_clusters:
        cluster_variance = variance_data[cluster_labels == cluster]
        if len(cluster_variance) == 0:
            continue
        mean_variance = cluster_variance.mean(axis=0)
        std_variance = cluster_variance.std(axis=0)
        cluster_variance_stats[cluster] = {
            'mean_variance': mean_variance,
            'std_variance': std_variance,
            'all_variances': cluster_variance
        }
    return cluster_variance_stats

def perform_statistical_tests(cluster_variance_stats):
    variance_x = [stats['all_variances'][:, 0] for stats in cluster_variance_stats.values()]
    variance_y = [stats['all_variances'][:, 1] for stats in cluster_variance_stats.values()]
    variance_z = [stats['all_variances'][:, 2] for stats in cluster_variance_stats.values()]
    
    stat_x, p_x = kruskal(*variance_x)
    stat_y, p_y = kruskal(*variance_y)
    stat_z, p_z = kruskal(*variance_z)
    
    results = {
        'Variance_X': {'H-statistic': stat_x, 'p-value': p_x},
        'Variance_Y': {'H-statistic': stat_y, 'p-value': p_y},
        'Variance_Z': {'H-statistic': stat_z, 'p-value': p_z}
    }
    return results
