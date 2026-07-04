"""
preprocess.py
Converts raw trajectory DataFrame (frame_number, object_id, x, y, ...)
into normalized (x, y) sequences per vehicle, ready for the LSTM model.

Also supports segmenting a single long real GPS trajectory (from I2WDD)
into fixed-length windows, since real I2WDD data is one continuous trace
rather than multiple objects like the synthetic data.
"""

import numpy as np
import pandas as pd


def extract_sequences(df: pd.DataFrame) -> dict:
    """
    Groups the DataFrame by object_id and extracts (x, y) sequences.

    Returns a dict: {object_id: np.array of shape (num_frames, 2)}
    """
    sequences = {}
    for obj_id, group in df.groupby("object_id"):
        group_sorted = group.sort_values("frame_number")
        xy = group_sorted[["x", "y"]].to_numpy()
        sequences[obj_id] = xy
    return sequences


def normalize_sequences(sequences: dict, frame_width: int, frame_height: int) -> dict:
    """
    Scales all (x, y) values to the range [0, 1] based on frame dimensions.
    This keeps the LSTM's inputs in a stable, consistent range.
    """
    normalized = {}
    for obj_id, seq in sequences.items():
        seq_norm = seq.copy().astype(float)
        seq_norm[:, 0] /= frame_width   # normalize x
        seq_norm[:, 1] /= frame_height  # normalize y
        normalized[obj_id] = seq_norm
    return normalized


def pad_sequences(sequences: dict, max_len: int = None) -> tuple:
    """
    Pads all sequences to the same length so they can be batched together.
    Shorter sequences are padded with zeros at the end.

    Returns:
    - padded_array: shape (num_objects, max_len, 2)
    - lengths: original length of each sequence (needed later to ignore padding in loss)
    """
    lengths = [len(seq) for seq in sequences.values()]
    if max_len is None:
        max_len = max(lengths)

    num_objects = len(sequences)
    padded_array = np.zeros((num_objects, max_len, 2))

    for i, (obj_id, seq) in enumerate(sequences.items()):
        length = min(len(seq), max_len)
        padded_array[i, :length, :] = seq[:length]

    return padded_array, np.array(lengths)


def preprocess_pipeline(df: pd.DataFrame, frame_width: int = 1920, frame_height: int = 1080) -> tuple:
    """
    Full preprocessing pipeline for SYNTHETIC data: raw DataFrame -> padded, normalized array.
    """
    sequences = extract_sequences(df)
    normalized = normalize_sequences(sequences, frame_width, frame_height)
    padded_array, lengths = pad_sequences(normalized)
    return padded_array, lengths


def segment_and_normalize_gps(trajectory: np.ndarray, window_size: int = 100, stride: int = 50) -> tuple:
    """
    Takes a single long real GPS trajectory (from load_i2wdd_gps) and splits it
    into multiple fixed-length overlapping windows, similar in spirit to having
    multiple "objects" for training - even though it's really one continuous ride.

    Args:
        trajectory: array of shape (num_points, 2) -> (lat, long)
        window_size: number of points per window/segment
        stride: how far to move the window each step (smaller stride = more overlap = more windows)

    Returns:
        padded_array: shape (num_windows, window_size, 2) - normalized (lat, long)
        lengths: array of real lengths per window (all equal to window_size here,
                 since we only keep full windows - no partial/padded windows)
    """
    num_points = trajectory.shape[0]

    # Normalize based on this trajectory's own min/max range (GPS has no fixed frame size)
    lat_min, lat_max = trajectory[:, 0].min(), trajectory[:, 0].max()
    long_min, long_max = trajectory[:, 1].min(), trajectory[:, 1].max()

    normalized = trajectory.copy().astype(float)
    normalized[:, 0] = (normalized[:, 0] - lat_min) / (lat_max - lat_min + 1e-12)
    normalized[:, 1] = (normalized[:, 1] - long_min) / (long_max - long_min + 1e-12)

    # Slice into overlapping windows
    windows = []
    for start in range(0, num_points - window_size + 1, stride):
        window = normalized[start:start + window_size]
        windows.append(window)

    padded_array = np.stack(windows, axis=0)  # (num_windows, window_size, 2)
    lengths = np.full(padded_array.shape[0], window_size)  # all windows are full length

    return padded_array, lengths


if __name__ == "__main__":
    from data_loader import load_trajectories, load_i2wdd_gps

    # --- Test synthetic pipeline (unchanged) ---
    df = load_trajectories(source="synthetic", num_objects=10)
    padded_array, lengths = preprocess_pipeline(df)
    print(f"Synthetic - Padded array shape: {padded_array.shape}")
    print(f"Synthetic - Sequence lengths: {lengths}")
    print(f"Synthetic - Sample sequence (first object, first 5 frames):\n{padded_array[0][:5]}")

    # --- Test real GPS pipeline (new) ---
    print("\n--- Testing real I2WDD GPS preprocessing ---")
    gps_path = "data/raw/i2wdd/18_01_25/IMU/GPS_1.csv"
    real_traj = load_i2wdd_gps(gps_path)

    real_padded, real_lengths = segment_and_normalize_gps(real_traj, window_size=100, stride=50)
    print(f"Real GPS - Padded array shape: {real_padded.shape}")
    print(f"Real GPS - Number of windows: {len(real_lengths)}")
    print(f"Real GPS - Sample window (first 5 points, normalized):\n{real_padded[0][:5]}")