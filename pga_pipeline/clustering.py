import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    confusion_matrix,
)
from scipy.optimize import linear_sum_assignment


def cluster_point_cloud_spectral(points, n_clusters=5, n_neighbors=20):
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


def align_cluster_labels(reference_labels, target_labels):
    cm = confusion_matrix(reference_labels, target_labels)
    row_ind, col_ind = linear_sum_assignment(-cm)
    label_mapping = {col_ind[i]: row_ind[i] for i in range(len(row_ind))}
    aligned_target_labels = np.array(
        [label_mapping.get(label, -1) for label in target_labels]
    )
    return aligned_target_labels


def assess_clustering_consistency(scans_cluster_labels):
    reference_labels = scans_cluster_labels[0]
    ari_scores = []
    nmi_scores = []
    for target_labels in scans_cluster_labels[1:]:
        aligned_target_labels = align_cluster_labels(reference_labels, target_labels)
        ari = adjusted_rand_score(reference_labels, aligned_target_labels)
        nmi = normalized_mutual_info_score(reference_labels, aligned_target_labels)
        ari_scores.append(ari)
        nmi_scores.append(nmi)
    consistency_metrics = {
        "Adjusted Rand Index": ari_scores,
        "Normalized Mutual Information": nmi_scores,
    }
    return consistency_metrics
