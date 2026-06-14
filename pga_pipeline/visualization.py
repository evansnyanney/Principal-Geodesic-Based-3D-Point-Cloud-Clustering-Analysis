import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap, BoundaryNorm

# Global plot settings
sns.set(style="whitegrid")
sns.set_context("talk")

plt.rcParams.update({
    'font.size': 24,
    'axes.titlesize': 30,
    'axes.labelsize': 28,
    'xtick.labelsize': 26,
    'ytick.labelsize': 26,
    'legend.fontsize': 24,
    'figure.titlesize': 32
})

def get_discrete_colormap(cluster_colors_subset):
    cmap = ListedColormap(cluster_colors_subset)
    return cmap

def plot_variance_distributions_single_figure(cluster_variance_stats, n_clusters, cluster_colors, output_folder='outputs/variance_distributions'):
    os.makedirs(output_folder, exist_ok=True)
    
    data = []
    for cluster, stats in cluster_variance_stats.items():
        df = pd.DataFrame(stats['all_variances'], columns=['Variance_X', 'Variance_Y', 'Variance_Z'])
        df['Cluster'] = cluster
        data.append(df)
    combined_df = pd.concat(data, ignore_index=True)
    
    palette = cluster_colors[:n_clusters]
    
    fig, axes = plt.subplots(1, 3, figsize=(32, 10))
    variance_axes = ['Variance_X', 'Variance_Y', 'Variance_Z']
    titles = ['Variance in X Across Clusters', 'Variance in Y Across Clusters', 'Variance in Z Across Clusters']
    
    for ax, var, title in zip(axes, variance_axes, titles):
        for cluster in range(n_clusters):
            cluster_data = combined_df[combined_df['Cluster'] == cluster][var]
            sns.kdeplot(cluster_data, color=palette[cluster], label=f'Cluster {cluster}', shade=False, linewidth=3, ax=ax)
        ax.set_title(title, fontsize=40, fontweight='bold')
        ax.set_xlabel(var, fontsize=36, fontweight='bold')
        ax.set_ylabel('Density', fontsize=36, fontweight='bold')
        ax.tick_params(axis='both', labelsize=36)
        if var == "Variance_Z":
            ax.legend(title='Clusters', fontsize=32, title_fontsize=34, loc='upper left', bbox_to_anchor=(1.05, 1))
        else:
            ax.legend(title='Clusters', fontsize=32, title_fontsize=34)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'Variance_Distributions.png'), dpi=300)
    plt.close()
    print(f"Variance distribution plots saved in '{output_folder}'.")

def plot_clusters_visualization(points, cluster_labels, scan_idx, cluster_colors, output_folder='outputs/cluster_visualizations'):
    os.makedirs(output_folder, exist_ok=True)
    
    unique_clusters = np.unique(cluster_labels)
    n_clusters = len(unique_clusters)
    palette = cluster_colors[:n_clusters]
    cmap = ListedColormap(palette)
    norm = BoundaryNorm(np.arange(-0.5, n_clusters + 0.5, 1), ncolors=n_clusters)
    
    labels_numeric = cluster_labels.copy()
    if -1 in unique_clusters:
        labels_numeric[cluster_labels == -1] = n_clusters
        adjusted_cmap = ListedColormap(palette + ['grey'])
        adjusted_norm = BoundaryNorm(np.arange(-0.5, n_clusters + 1.5, 1), ncolors=n_clusters + 1)
    else:
        adjusted_cmap = cmap
        adjusted_norm = norm
    
    fig = plt.figure(figsize=(28, 20))
    
    # 2D Plot: X vs Y
    ax1 = fig.add_subplot(2, 2, 1)
    scatter1 = ax1.scatter(points[:, 0], points[:, 1], c=labels_numeric, cmap=adjusted_cmap, norm=adjusted_norm, s=40, alpha=0.7)
    ax1.set_title(f'Scan {scan_idx}: Cluster Visualization X vs Y', fontsize=32, fontweight='bold')
    ax1.set_xlabel('X Coordinate', fontsize=28, fontweight='bold')
    ax1.set_ylabel('Y Coordinate', fontsize=28, fontweight='bold')
    ax1.tick_params(axis='both', labelsize=28)
    cbar1 = fig.colorbar(scatter1, ax=ax1, ticks=np.arange(n_clusters + (1 if -1 in unique_clusters else 0)))
    cbar1.set_label('Cluster ID', fontsize=28, fontweight='bold')
    cbar1.ax.tick_params(labelsize=28)
    
    # 2D Plot: Y vs Z
    ax2 = fig.add_subplot(2, 2, 2)
    scatter2 = ax2.scatter(points[:, 1], points[:, 2], c=labels_numeric, cmap=adjusted_cmap, norm=adjusted_norm, s=40, alpha=0.7)
    ax2.set_title(f'Scan {scan_idx}: Cluster Visualization Y vs Z', fontsize=32, fontweight='bold')
    ax2.set_xlabel('Y Coordinate', fontsize=28, fontweight='bold')
    ax2.set_ylabel('Z Coordinate', fontsize=28, fontweight='bold')
    ax2.tick_params(axis='both', labelsize=28)
    cbar2 = fig.colorbar(scatter2, ax=ax2, ticks=np.arange(n_clusters + (1 if -1 in unique_clusters else 0)))
    cbar2.set_label('Cluster ID', fontsize=28, fontweight='bold')
    cbar2.ax.tick_params(labelsize=28)
    
    # 2D Plot: X vs Z
    ax3 = fig.add_subplot(2, 2, 3)
    scatter3 = ax3.scatter(points[:, 0], points[:, 2], c=labels_numeric, cmap=adjusted_cmap, norm=adjusted_norm, s=40, alpha=0.7)
    ax3.set_title(f'Scan {scan_idx}: Cluster Visualization X vs Z', fontsize=32, fontweight='bold')
    ax3.set_xlabel('X Coordinate', fontsize=28, fontweight='bold')
    ax3.set_ylabel('Z Coordinate', fontsize=28, fontweight='bold')
    ax3.tick_params(axis='both', labelsize=28)
    cbar3 = fig.colorbar(scatter3, ax=ax3, ticks=np.arange(n_clusters + (1 if -1 in unique_clusters else 0)))
    cbar3.set_label('Cluster ID', fontsize=28, fontweight='bold')
    cbar3.ax.tick_params(labelsize=28)
    
    # 3D Plot
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    scatter4 = ax4.scatter(points[:, 0], points[:, 1], points[:, 2],
                           c=labels_numeric, cmap=adjusted_cmap, norm=adjusted_norm, s=40, alpha=0.7)
    ax4.set_title(f'Scan {scan_idx}: 3D Cluster Visualization', fontsize=32, fontweight='bold')
    ax4.set_xlabel('X Coordinate', fontsize=28, fontweight='bold', labelpad=20)
    ax4.set_ylabel('Y Coordinate', fontsize=28, fontweight='bold', labelpad=20)
    ax4.set_zlabel('Z Coordinate', fontsize=28, fontweight='bold', labelpad=40)
    ax4.tick_params(axis='both', labelsize=28)
    cbar4 = fig.colorbar(scatter4, ax=ax4, shrink=0.6, aspect=20, pad=0.15,
                         ticks=np.arange(n_clusters + (1 if -1 in unique_clusters else 0)))
    cbar4.set_label('Cluster ID', fontsize=28, fontweight='bold')
    cbar4.ax.tick_params(labelsize=28)
    
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(os.path.join(output_folder, f'Cluster_Visualization_Scan_{scan_idx}.png'), dpi=300)
    plt.close()
    print(f"Cluster visualizations for Scan {scan_idx} saved in '{output_folder}'.")

def plot_average_variance(points, cluster_labels, average_variance_stats, method='frechet_mean', n_clusters=7, cluster_colors=None, output_folder='outputs/average_variance_plots'):
    os.makedirs(output_folder, exist_ok=True)
    
    unique_clusters = np.unique(cluster_labels)
    palette = cluster_colors[:n_clusters]
    cmap = ListedColormap(palette)
    norm = BoundaryNorm(np.arange(-0.5, n_clusters + 0.5, 1), ncolors=n_clusters)
    
    labels_numeric = cluster_labels.copy()
    if -1 in unique_clusters:
        labels_numeric[cluster_labels == -1] = n_clusters
        adjusted_cmap = ListedColormap(palette + ['grey'])
        adjusted_norm = BoundaryNorm(np.arange(-0.5, n_clusters + 1.5, 1), ncolors=n_clusters + 1)
    else:
        adjusted_cmap = cmap
        adjusted_norm = norm
    
    fig = plt.figure(figsize=(28, 20))
    
    # 2D Plot: X vs Y
    ax1 = fig.add_subplot(2, 2, 1)
    scatter1 = ax1.scatter(points[:, 0], points[:, 1], c=labels_numeric, cmap=adjusted_cmap, norm=adjusted_norm, s=40, alpha=0.7)
    ax1.set_title(f'{method} Average Variance Visualization X vs Y', fontsize=32, fontweight='bold')
    ax1.set_xlabel('X Coordinate', fontsize=28, fontweight='bold')
    ax1.set_ylabel('Y Coordinate', fontsize=28, fontweight='bold')
    ax1.tick_params(axis='both', labelsize=28)
    cbar1 = fig.colorbar(scatter1, ax=ax1, ticks=np.arange(n_clusters + (1 if -1 in unique_clusters else 0)))
    cbar1.set_label('Cluster ID', fontsize=28, fontweight='bold')
    cbar1.ax.tick_params(labelsize=28)
    
    # 2D Plot: Y vs Z
    ax2 = fig.add_subplot(2, 2, 2)
    scatter2 = ax2.scatter(points[:, 1], points[:, 2], c=labels_numeric, cmap=adjusted_cmap, norm=adjusted_norm, s=40, alpha=0.7)
    ax2.set_title(f'{method} Average Variance Visualization Y vs Z', fontsize=32, fontweight='bold')
    ax2.set_xlabel('Y Coordinate', fontsize=28, fontweight='bold')
    ax2.set_ylabel('Z Coordinate', fontsize=28, fontweight='bold')
    ax2.tick_params(axis='both', labelsize=28)
    cbar2 = fig.colorbar(scatter2, ax=ax2, ticks=np.arange(n_clusters + (1 if -1 in unique_clusters else 0)))
    cbar2.set_label('Cluster ID', fontsize=28, fontweight='bold')
    cbar2.ax.tick_params(labelsize=28)
    
    # 2D Plot: X vs Z
    ax3 = fig.add_subplot(2, 2, 3)
    scatter3 = ax3.scatter(points[:, 0], points[:, 2], c=labels_numeric, cmap=adjusted_cmap, norm=adjusted_norm, s=40, alpha=0.7)
    ax3.set_title(f'{method} Average Variance Visualization X vs Z', fontsize=32, fontweight='bold')
    ax3.set_xlabel('X Coordinate', fontsize=28, fontweight='bold')
    ax3.set_ylabel('Z Coordinate', fontsize=28, fontweight='bold')
    ax3.tick_params(axis='both', labelsize=28)
    cbar3 = fig.colorbar(scatter3, ax=ax3, ticks=np.arange(n_clusters + (1 if -1 in unique_clusters else 0)))
    cbar3.set_label('Cluster ID', fontsize=28, fontweight='bold')
    cbar3.ax.tick_params(labelsize=28)
    
    # 3D Plot
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    scatter4 = ax4.scatter(points[:, 0], points[:, 1], points[:, 2],
                           c=labels_numeric, cmap=adjusted_cmap, norm=adjusted_norm, s=40, alpha=0.7)
    ax4.set_title(f'{method} Average Variance Visualization (3D)', fontsize=32, fontweight='bold')
    ax4.set_xlabel('X Coordinate', fontsize=28, fontweight='bold', labelpad=20)
    ax4.set_ylabel('Y Coordinate', fontsize=28, fontweight='bold', labelpad=20)
    ax4.set_zlabel('Z Coordinate', fontsize=28, fontweight='bold', labelpad=40)
    ax4.tick_params(axis='both', labelsize=28)
    cbar4 = fig.colorbar(scatter4, ax=ax4, shrink=0.6, aspect=20, pad=0.15,
                         ticks=np.arange(n_clusters + (1 if -1 in unique_clusters else 0)))
    cbar4.set_label('Cluster ID', fontsize=28, fontweight='bold')
    cbar4.ax.tick_params(labelsize=28)
    
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(os.path.join(output_folder, f'{method}_Average_Variance_Visualization.png'), dpi=300)
    plt.close()
    print(f"{method} average variance visualizations saved in '{output_folder}'.")

def plot_pga_results(cluster_pga_stats, output_folder='outputs/pga_results'):
    os.makedirs(output_folder, exist_ok=True)
    for cluster, stats in cluster_pga_stats.items():
        principal_directions = stats['principal_directions']
        eigenvalues = stats['eigenvalues']
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        mean = stats['mean']
        for i, direction in enumerate(principal_directions):
            ax.quiver(mean[0], mean[1], mean[2],
                      direction[0], direction[1], direction[2],
                      length=eigenvalues[i]*2, color='r', arrow_length_ratio=0.1, linewidth=2)
        ax.set_title(f'Cluster {cluster}: Principal Geodesic Directions', fontsize=32, fontweight='bold')
        ax.set_xlabel('X', fontsize=28, fontweight='bold', labelpad=20)
        ax.set_ylabel('Y', fontsize=28, fontweight='bold', labelpad=20)
        ax.set_zlabel('Z', fontsize=28, fontweight='bold', labelpad=40)
        ax.tick_params(axis='both', labelsize=28)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f'Cluster_{cluster}_PGA.png'), dpi=300)
        plt.close()
        print(f"PGA visualization for Cluster {cluster} saved in '{output_folder}'.")

def plot_means(means, method='Euclidean', n_clusters=7, cluster_colors=None, output_folder='outputs/means_visualizations'):
    os.makedirs(output_folder, exist_ok=True)
    palette = cluster_colors[:n_clusters]
    cmap = ListedColormap(palette)
    norm = BoundaryNorm(np.arange(-0.5, n_clusters + 0.5, 1), ncolors=n_clusters)
    cluster_labels = np.arange(n_clusters)
    fig = plt.figure(figsize=(30, 20))
    
    # 2D Plot: X vs Y
    ax1 = fig.add_subplot(2, 2, 1)
    scatter1 = ax1.scatter(means[:, 0], means[:, 1], c=cluster_labels, cmap=cmap, norm=norm, s=400, edgecolors='k')
    ax1.set_title(f'{method} Means X vs Y', fontsize=32, fontweight='bold')
    ax1.set_xlabel('Mean X', fontsize=28, fontweight='bold')
    ax1.set_ylabel('Mean Y', fontsize=28, fontweight='bold')
    ax1.tick_params(axis='both', labelsize=28)
    cbar1 = fig.colorbar(scatter1, ax=ax1, ticks=range(n_clusters))
    cbar1.set_label('Cluster ID', fontsize=28, fontweight='bold')
    cbar1.ax.tick_params(labelsize=28)
    
    # 2D Plot: Y vs Z
    ax2 = fig.add_subplot(2, 2, 2)
    scatter2 = ax2.scatter(means[:, 1], means[:, 2], c=cluster_labels, cmap=cmap, norm=norm, s=400, edgecolors='k')
    ax2.set_title(f'{method} Means Y vs Z', fontsize=32, fontweight='bold')
    ax2.set_xlabel('Mean Y', fontsize=28, fontweight='bold')
    ax2.set_ylabel('Mean Z', fontsize=28, fontweight='bold')
    ax2.tick_params(axis='both', labelsize=28)
    cbar2 = fig.colorbar(scatter2, ax=ax2, ticks=range(n_clusters))
    cbar2.set_label('Cluster ID', fontsize=28, fontweight='bold')
    cbar2.ax.tick_params(labelsize=28)
    
    # 2D Plot: X vs Z
    ax3 = fig.add_subplot(2, 2, 3)
    scatter3 = ax3.scatter(means[:, 0], means[:, 2], c=cluster_labels, cmap=cmap, norm=norm, s=400, edgecolors='k')
    ax3.set_title(f'{method} Means X vs Z', fontsize=32, fontweight='bold')
    ax3.set_xlabel('Mean X', fontsize=28, fontweight='bold')
    ax3.set_ylabel('Mean Z', fontsize=28, fontweight='bold')
    ax3.tick_params(axis='both', labelsize=28)
    cbar3 = fig.colorbar(scatter3, ax=ax3, ticks=range(n_clusters))
    cbar3.set_label('Cluster ID', fontsize=28, fontweight='bold')
    cbar3.ax.tick_params(labelsize=28)
    
    # 3D Plot
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    scatter4 = ax4.scatter(means[:, 0], means[:, 1], means[:, 2],
                           c=cluster_labels, cmap=cmap, norm=norm, s=400, edgecolors='k')
    ax4.set_title(f'{method} Means (3D)', fontsize=32, fontweight='bold')
    ax4.set_xlabel('Mean X', fontsize=28, fontweight='bold', labelpad=20)
    ax4.set_ylabel('Mean Y', fontsize=28, fontweight='bold', labelpad=20)
    ax4.set_zlabel('Mean Z', fontsize=28, fontweight='bold', labelpad=40)
    ax4.tick_params(axis='both', labelsize=28)
    ax4.set_xlim([means[:, 0].min() - 1, means[:, 0].max() + 1])
    ax4.set_ylim([means[:, 1].min() - 1, means[:, 1].max() + 1])
    ax4.set_zlim([means[:, 2].min() - 1, means[:, 2].max() + 1])
    
    cbar4 = fig.colorbar(scatter4, ax=ax4, shrink=0.6, aspect=20, pad=0.15, ticks=range(n_clusters))
    cbar4.set_label('Cluster ID', fontsize=28, fontweight='bold')
    cbar4.ax.tick_params(labelsize=28)
    
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(os.path.join(output_folder, f'{method}_Means_Visualization.png'), dpi=300)
    plt.close()
    print(f"{method} means visualizations saved in '{output_folder}'.")
