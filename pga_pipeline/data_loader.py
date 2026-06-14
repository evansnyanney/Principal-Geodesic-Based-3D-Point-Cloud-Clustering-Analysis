import pandas as pd
import numpy as np
import logging
import sys
import os
from tqdm import tqdm


def load_csv(file_path, has_header=True, x_col="X", y_col="Y", z_col="Z"):
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


def load_all_scans(
    folder_path, has_header=True, x_col="X", y_col="Y", z_col="Z", max_scans=None
):
    try:
        scan_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".csv")])
        if max_scans:
            scan_files = scan_files[:max_scans]
        all_scans = []
        for file in tqdm(scan_files, desc="Loading scans"):
            file_path = os.path.join(folder_path, file)
            scan_points = load_csv(
                file_path, has_header=has_header, x_col=x_col, y_col=y_col, z_col=z_col
            )
            all_scans.append(scan_points)
        return all_scans
    except Exception as e:
        logging.error(f"Error loading scans: {e}")
        sys.exit(1)


def load_pointwise_variance(
    file_path, variance_columns=None, expected_landmarks=None, variance_scale_factor=1.0
):
    if variance_columns is None:
        variance_columns = ["Variance_X", "Variance_Y", "Variance_Z"]
    try:
        df = pd.read_csv(file_path)
        variance_data = df[variance_columns].values.astype(np.float32)
        # Scale the variance data using the provided factor
        variance_data = variance_data * variance_scale_factor
        if expected_landmarks and variance_data.shape[0] != expected_landmarks:
            logging.error("Mismatch in number of variance entries and landmarks.")
            sys.exit(1)
        return variance_data
    except Exception as e:
        logging.error(f"Error loading variance CSV file: {e}")
        sys.exit(1)
