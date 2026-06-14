import numpy as np
import logging
import sys
from sklearn.decomposition import PCA

def normalize_points(points, method='none'):
    if method == 'none':
        return points
    elif method == 'standard':
        mean = points.mean(axis=0)
        std = points.std(axis=0)
        std = np.where(std == 0, 1e-10, std)
        normalized = (points - mean) / std
        return normalized
    elif method == 'minmax':
        min_vals = points.min(axis=0)
        max_vals = points.max(axis=0)
        ranges = max_vals - min_vals
        ranges = np.where(ranges == 0, 1e-10, ranges)
        normalized = (points - min_vals) / ranges
        return normalized
    elif method == 'unit_sphere':
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-10, norms)
        normalized = points / norms
        return normalized
    else:
        logging.error(f"Invalid normalization method: {method}.")
        sys.exit(1)

def compute_frechet_mean(points, max_iter=50, tol=1e-5):
    mean = points.mean(axis=0)
    mean /= np.linalg.norm(mean)
    for i in range(max_iter):
        prev_mean = mean.copy()
        tangent_vectors = log_map(points, mean)
        mean_shift = tangent_vectors.mean(axis=0)
        mean = exp_map(mean_shift.reshape(1, -1), mean)[0]
        if np.linalg.norm(mean - prev_mean) < tol:
            break
    return mean

def compute_euclidean_mean(points):
    return points.mean(axis=0)

def log_map(points, base_point):
    dot_product = np.einsum('ij,j->i', points, base_point)
    dot_product = np.clip(dot_product, -1.0, 1.0)
    angles = np.arccos(dot_product)
    angles = np.where(angles == 0, 1e-10, angles)
    sin_angles = np.sin(angles)
    sin_angles = np.where(sin_angles == 0, 1e-10, sin_angles)
    tangent_vectors = (points - np.outer(dot_product, base_point)) / sin_angles[:, np.newaxis]
    tangent_vectors *= angles[:, np.newaxis]
    return tangent_vectors

def exp_map(tangent_vectors, base_point):
    angles = np.linalg.norm(tangent_vectors, axis=1, keepdims=True)
    sin_angles = np.sin(angles)
    sin_angles = np.where(angles == 0, 1.0, sin_angles / angles)
    new_points = base_point * np.cos(angles) + tangent_vectors * sin_angles
    return normalize_points(new_points, method='unit_sphere')

def compute_pga(cluster_points, mean, method='minmax', n_components=3):
    if method == 'unit_sphere':
        tangent_vectors = log_map(cluster_points, mean)
    else:
        tangent_vectors = cluster_points - mean
    pca = PCA(n_components=n_components)
    pca.fit(tangent_vectors)
    principal_directions = pca.components_
    eigenvalues = pca.explained_variance_
    return {
        'principal_directions': principal_directions,
        'eigenvalues': eigenvalues
    }

def compute_frechet_mean_per_cluster(all_scans, scans_cluster_labels, n_clusters, method='minmax'):
    cluster_points = {cluster: [] for cluster in range(n_clusters)}
    for scan_idx in range(len(all_scans)):
        scan = all_scans[scan_idx]
        labels = scans_cluster_labels[scan_idx]
        for point, label in zip(scan, labels):
            if label != -1:
                cluster_points[label].append(point)
    means = []
    for cluster in range(n_clusters):
        points = np.array(cluster_points[cluster])
        if len(points) == 0:
            logging.warning(f"No points found for Cluster {cluster}.")
            means.append(np.array([0.0, 0.0, 0.0]))
            continue
        if method == 'unit_sphere':
            mean = compute_frechet_mean(points)
        else:
            mean = compute_euclidean_mean(points)
        means.append(mean)
    return np.array(means)
