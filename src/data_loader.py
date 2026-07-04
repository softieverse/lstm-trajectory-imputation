"""
data_loader.py
Generates synthetic vehicle trajectory data mimicking I2WDD's structure:
frame_number, object_id, x, y, width, height, category

Also supports loading REAL GPS trajectory data from the I2WDD dataset
(note: I2WDD tracks only the EGO two-wheeler, so it gives ONE trajectory
per GPS file, not multiple interacting vehicles like SkyEye would have).
"""

import numpy as np
import pandas as pd


def generate_synthetic_trajectories(
    num_objects: int = 50,
    min_frames: int = 80,
    max_frames: int = 150,
    frame_width: int = 1920,
    frame_height: int = 1080,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generates synthetic vehicle-like trajectories.
    Each object moves with smooth, semi-random motion (not pure random walk)
    to mimic real vehicle behavior.

    Returns a DataFrame with columns:
    frame_number, object_id, x, y, width, height, category
    """
    rng = np.random.default_rng(seed)
    categories = ["car", "motorbike", "auto-rickshaw", "bus", "truck"]

    records = []

    for obj_id in range(num_objects):
        num_frames = rng.integers(min_frames, max_frames)
        category = rng.choice(categories, p=[0.3, 0.4, 0.2, 0.05, 0.05])

        # Random starting position
        x = rng.uniform(0, frame_width)
        y = rng.uniform(0, frame_height)

        # Random initial velocity
        vx = rng.uniform(-5, 5)
        vy = rng.uniform(-5, 5)

        # Object size depends loosely on category
        base_w, base_h = {
            "car": (60, 40),
            "motorbike": (30, 25),
            "auto-rickshaw": (45, 35),
            "bus": (90, 50),
            "truck": (80, 45),
        }[category]

        for frame in range(num_frames):
            # Smooth motion: velocity drifts slightly each frame (not pure noise)
            vx += rng.normal(0, 0.5)
            vy += rng.normal(0, 0.5)

            x += vx
            y += vy

            # Keep within frame bounds (simple clamp)
            x = np.clip(x, 0, frame_width)
            y = np.clip(y, 0, frame_height)

            records.append({
                "frame_number": frame,
                "object_id": obj_id,
                "x": x,
                "y": y,
                "width": base_w + rng.normal(0, 2),
                "height": base_h + rng.normal(0, 2),
                "category": category,
            })

    df = pd.DataFrame(records)
    return df


def load_i2wdd_gps(filepath: str) -> np.ndarray:
    """
    Loads a single GPS_X.csv file from the real I2WDD dataset
    and extracts (lat, long) as a trajectory.

    NOTE: I2WDD tracks the EGO vehicle only (the bike doing the recording),
    not multiple surrounding vehicles. So this returns ONE trajectory
    (a single object's path), not multiple interacting objects like SkyEye would.

    Args:
        filepath: path to a GPS_X.csv file (e.g. "data/raw/i2wdd/18_01_25/IMU/GPS_1.csv")

    Returns:
        array of shape (num_points, 2) -> columns are (lat, long)
    """
    df = pd.read_csv(filepath)
    trajectory = df[["GPS (Lat.) [deg]", "GPS (Long.) [deg]"]].to_numpy()
    return trajectory


def load_trajectories(source: str = "synthetic", **kwargs) -> pd.DataFrame:
    """
    Main entry point for loading synthetic trajectory data (multi-object).
    For real I2WDD GPS data, use load_i2wdd_gps() directly instead,
    since it returns a single trajectory rather than a multi-object DataFrame.
    """
    if source == "synthetic":
        return generate_synthetic_trajectories(**kwargs)
    else:
        raise NotImplementedError(f"Data source '{source}' not implemented yet.")


if __name__ == "__main__":
    # --- Test synthetic data ---
    df = load_trajectories(source="synthetic", num_objects=10)
    print(df.head(10))
    print(f"\nTotal objects: {df['object_id'].nunique()}")
    print(f"Total rows: {len(df)}")

    # --- Test real I2WDD GPS data ---
    print("\n--- Testing real I2WDD GPS data ---")
    gps_path = "data/raw/i2wdd/18_01_25/IMU/GPS_1.csv"
    real_traj = load_i2wdd_gps(gps_path)
    print(f"Real trajectory shape: {real_traj.shape}")
    print(f"First 5 points:\n{real_traj[:5]}")