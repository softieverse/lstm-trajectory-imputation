import numpy as np
import pandas as pd


def extract_sequences(df: pd.DataFrame) -> dict:

    sequences = {}
    for obj_id, group in df.groupby("object_id"):
        group_sorted = group.sort_values("frame_number")
        xy = group_sorted[["x", "y"]].to_numpy()
        sequences[obj_id] = xy
    return sequences


def normalize_sequences(sequences: dict, frame_width: int, frame_height: int) -> dict:

    normalized = {}
    for obj_id, seq in sequences.items():
        seq_norm = seq.copy().astype(float)
        seq_norm[:, 0] /= frame_width   # normalize x
        seq_norm[:, 1] /= frame_height  # normalize y
        normalized[obj_id] = seq_norm
    return normalized


def pad_sequences(sequences: dict, max_len: int = None) -> tuple:
 
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

    sequences = extract_sequences(df)
    normalized = normalize_sequences(sequences, frame_width, frame_height)
    padded_array, lengths = pad_sequences(normalized)
    return padded_array, lengths


def segment_and_normalize_gps(trajectory: np.ndarray, window_size: int = 100, stride: int = 50) -> tuple:

    num_points = trajectory.shape[0]

    lat_min, lat_max = trajectory[:, 0].min(), trajectory[:, 0].max()
    long_min, long_max = trajectory[:, 1].min(), trajectory[:, 1].max()

    normalized = trajectory.copy().astype(float)
    normalized[:, 0] = (normalized[:, 0] - lat_min) / (lat_max - lat_min + 1e-12)
    normalized[:, 1] = (normalized[:, 1] - long_min) / (long_max - long_min + 1e-12)

    windows = []
    for start in range(0, num_points - window_size + 1, stride):
        window = normalized[start:start + window_size]
        windows.append(window)

    padded_array = np.stack(windows, axis=0)  # (num_windows, window_size, 2)
    lengths = np.full(padded_array.shape[0], window_size)  # all windows are full length

    return padded_array, lengths


if __name__ == "__main__":
    from data_loader import load_trajectories, load_i2wdd_gps

    df = load_trajectories(source="synthetic", num_objects=10)
    padded_array, lengths = preprocess_pipeline(df)
    print(f"Synthetic - Padded array shape: {padded_array.shape}")
    print(f"Synthetic - Sequence lengths: {lengths}")
    print(f"Synthetic - Sample sequence (first object, first 5 frames):\n{padded_array[0][:5]}")

    print("\n--- Testing real I2WDD GPS preprocessing ---")
    gps_path = "data/raw/i2wdd/18_01_25/IMU/GPS_1.csv"
    real_traj = load_i2wdd_gps(gps_path)

    real_padded, real_lengths = segment_and_normalize_gps(real_traj, window_size=100, stride=50)
    print(f"Real GPS - Padded array shape: {real_padded.shape}")
    print(f"Real GPS - Number of windows: {len(real_lengths)}")
    print(f"Real GPS - Sample window (first 5 points, normalized):\n{real_padded[0][:5]}")