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
    rng = np.random.default_rng(seed)
    categories = ["car", "motorbike", "auto-rickshaw", "bus", "truck"]

    records = []

    for obj_id in range(num_objects):
        num_frames = rng.integers(min_frames, max_frames)
        category = rng.choice(categories, p=[0.3, 0.4, 0.2, 0.05, 0.05])

        x = rng.uniform(0, frame_width)
        y = rng.uniform(0, frame_height)

        vx = rng.uniform(-5, 5)
        vy = rng.uniform(-5, 5)

        base_w, base_h = {
            "car": (60, 40),
            "motorbike": (30, 25),
            "auto-rickshaw": (45, 35),
            "bus": (90, 50),
            "truck": (80, 45),
        }[category]

        for frame in range(num_frames):
            vx += rng.normal(0, 0.5)
            vy += rng.normal(0, 0.5)

            x += vx
            y += vy

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

    df = pd.read_csv(filepath)
    trajectory = df[["GPS (Lat.) [deg]", "GPS (Long.) [deg]"]].to_numpy()
    return trajectory


def load_trajectories(source: str = "synthetic", **kwargs) -> pd.DataFrame:

    if source == "synthetic":
        return generate_synthetic_trajectories(**kwargs)
    else:
        raise NotImplementedError(f"Data source '{source}' not implemented yet.")


if __name__ == "__main__":
    df = load_trajectories(source="synthetic", num_objects=10)
    print(df.head(10))
    print(f"\nTotal objects: {df['object_id'].nunique()}")
    print(f"Total rows: {len(df)}")

    print("\n--- Testing real I2WDD GPS data ---")
    gps_path = "data/raw/i2wdd/18_01_25/IMU/GPS_1.csv"
    real_traj = load_i2wdd_gps(gps_path)
    print(f"Real trajectory shape: {real_traj.shape}")
    print(f"First 5 points:\n{real_traj[:5]}")