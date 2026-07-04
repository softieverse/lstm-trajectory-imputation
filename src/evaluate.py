import torch
import numpy as np
import matplotlib.pyplot as plt

from data_loader import load_trajectories, load_i2wdd_gps
from preprocess import preprocess_pipeline, segment_and_normalize_gps
from masking import apply_masking
from model import LSTMImputer


def compute_rmse(predictions: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor,
                  frame_width: int = 1920, frame_height: int = 1080) -> dict:

    mask_expanded = mask.unsqueeze(-1).expand_as(predictions).bool()

    pred_masked = predictions[mask_expanded].view(-1, 2)
    target_masked = targets[mask_expanded].view(-1, 2)

    pred_pixels = pred_masked.clone()
    pred_pixels[:, 0] *= frame_width
    pred_pixels[:, 1] *= frame_height

    target_pixels = target_masked.clone()
    target_pixels[:, 0] *= frame_width
    target_pixels[:, 1] *= frame_height

    rmse_x = torch.sqrt(torch.mean((pred_pixels[:, 0] - target_pixels[:, 0]) ** 2))
    rmse_y = torch.sqrt(torch.mean((pred_pixels[:, 1] - target_pixels[:, 1]) ** 2))
    rmse_overall = torch.sqrt(torch.mean((pred_pixels - target_pixels) ** 2))

    return {
        "rmse_x_pixels": rmse_x.item(),
        "rmse_y_pixels": rmse_y.item(),
        "rmse_overall_pixels": rmse_overall.item(),
        "num_points_evaluated": mask.sum().item()
    }


def compute_rmse_gps_meters(predictions: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor,
                              lat_min: float, lat_max: float, long_min: float, long_max: float,
                              ref_lat: float) -> dict:
    mask_expanded = mask.unsqueeze(-1).expand_as(predictions).bool()

    pred_masked = predictions[mask_expanded].view(-1, 2)
    target_masked = targets[mask_expanded].view(-1, 2)

    pred_latlong = pred_masked.clone()
    pred_latlong[:, 0] = pred_latlong[:, 0] * (lat_max - lat_min) + lat_min
    pred_latlong[:, 1] = pred_latlong[:, 1] * (long_max - long_min) + long_min

    target_latlong = target_masked.clone()
    target_latlong[:, 0] = target_latlong[:, 0] * (lat_max - lat_min) + lat_min
    target_latlong[:, 1] = target_latlong[:, 1] * (long_max - long_min) + long_min

    meters_per_deg_lat = 111320.0
    meters_per_deg_long = 111320.0 * np.cos(np.radians(ref_lat))

    lat_error_m = (pred_latlong[:, 0] - target_latlong[:, 0]) * meters_per_deg_lat
    long_error_m = (pred_latlong[:, 1] - target_latlong[:, 1]) * meters_per_deg_long

    rmse_lat_m = torch.sqrt(torch.mean(lat_error_m ** 2))
    rmse_long_m = torch.sqrt(torch.mean(long_error_m ** 2))
    rmse_overall_m = torch.sqrt(torch.mean(lat_error_m ** 2 + long_error_m ** 2))

    return {
        "rmse_lat_meters": rmse_lat_m.item(),
        "rmse_long_meters": rmse_long_m.item(),
        "rmse_overall_meters": rmse_overall_m.item(),
        "num_points_evaluated": mask.sum().item()
    }


def plot_trajectory_comparison(original: np.ndarray, masked: np.ndarray, predicted: np.ndarray,
                                 mask: np.ndarray, real_length: int, object_id: int = 0,
                                 title: str = "Trajectory Imputation",
                                 xlabel: str = "X position (normalized)",
                                 ylabel: str = "Y position (normalized)",
                                 save_path: str = "results/trajectory_comparison.png"):

    orig = original[:real_length]
    pred = predicted[:real_length]
    m = mask[:real_length]

    plt.figure(figsize=(8, 6))

    plt.plot(orig[:, 0], orig[:, 1], 'g-', label="True trajectory", linewidth=2, alpha=0.7)

    observed = ~m
    plt.scatter(orig[observed, 0], orig[observed, 1], c='blue', s=30, label="Observed points", zorder=3)
    plt.scatter(orig[m, 0], orig[m, 1], c='red', marker='x', s=60, label="Missing (true value)", zorder=3)
    plt.scatter(pred[m, 0], pred[m, 1], c='orange', marker='o', s=40,
                facecolors='none', edgecolors='orange', linewidths=2,
                label="Model's imputed value", zorder=4)

    plt.title(f"{title} — Sample {object_id}")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Plot saved to {save_path}")
    plt.close()


def evaluate_model(model_path: str = "results/lstm_imputer.pth", num_objects: int = 100,
                    missing_rate: float = 0.2, seed: int = 99):

    df = load_trajectories(source="synthetic", num_objects=num_objects, seed=seed)
    padded_array, lengths = preprocess_pipeline(df)
    masked_array, mask_array = apply_masking(
        padded_array, lengths, method="random", missing_rate=missing_rate, seed=seed
    )

    original = torch.tensor(padded_array, dtype=torch.float32)
    masked = torch.tensor(masked_array, dtype=torch.float32)
    mask = torch.tensor(mask_array, dtype=torch.bool)

    model = LSTMImputer(input_size=2, hidden_size=64, num_layers=2, dropout=0.2)  # CHANGED
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        predictions = model(masked, mask)

    metrics = compute_rmse(predictions, original, mask)

    print("=== Evaluation Results (Synthetic data, unseen test vehicles) ===")
    print(f"Points evaluated: {metrics['num_points_evaluated']}")
    print(f"RMSE (X): {metrics['rmse_x_pixels']:.2f} pixels")
    print(f"RMSE (Y): {metrics['rmse_y_pixels']:.2f} pixels")
    print(f"RMSE (Overall): {metrics['rmse_overall_pixels']:.2f} pixels")

    plot_trajectory_comparison(
        original=original[0].numpy(), masked=masked[0].numpy(), predicted=predictions[0].numpy(),
        mask=mask[0].numpy(), real_length=lengths[0], object_id=0,
        title="Synthetic Trajectory Imputation",
        save_path="results/trajectory_comparison_synthetic.png"
    )

    return metrics


def evaluate_model_real_gps(gps_filepath: str, model_path: str = "results/lstm_imputer_real_gps.pth",
                             window_size: int = 100, stride: int = 50,
                             missing_rate: float = 0.2, seed: int = 99):

    real_traj = load_i2wdd_gps(gps_filepath)

    lat_min, lat_max = real_traj[:, 0].min(), real_traj[:, 0].max()
    long_min, long_max = real_traj[:, 1].min(), real_traj[:, 1].max()
    ref_lat = real_traj[:, 0].mean()  # used for lat/long-to-meters conversion

    padded_array, lengths = segment_and_normalize_gps(real_traj, window_size=window_size, stride=stride)
    masked_array, mask_array = apply_masking(
        padded_array, lengths, method="random", missing_rate=missing_rate, seed=seed
    )

    original = torch.tensor(padded_array, dtype=torch.float32)
    masked = torch.tensor(masked_array, dtype=torch.float32)
    mask = torch.tensor(mask_array, dtype=torch.bool)

    model = LSTMImputer(input_size=2, hidden_size=64, num_layers=2, dropout=0.2)  # CHANGED
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        predictions = model(masked, mask)

    metrics = compute_rmse_gps_meters(predictions, original, mask, lat_min, lat_max, long_min, long_max, ref_lat)

    print("=== Evaluation Results (Real I2WDD GPS data, unseen windows) ===")
    print(f"Points evaluated: {metrics['num_points_evaluated']}")
    print(f"RMSE (Latitude): {metrics['rmse_lat_meters']:.2f} meters")
    print(f"RMSE (Longitude): {metrics['rmse_long_meters']:.2f} meters")
    print(f"RMSE (Overall): {metrics['rmse_overall_meters']:.2f} meters")

    plot_trajectory_comparison(
        original=original[0].numpy(), masked=masked[0].numpy(), predicted=predictions[0].numpy(),
        mask=mask[0].numpy(), real_length=lengths[0], object_id=0,
        title="Real GPS Trajectory Imputation",
        xlabel="Latitude (normalized)", ylabel="Longitude (normalized)",
        save_path="results/trajectory_comparison_real_gps.png"
    )

    return metrics


if __name__ == "__main__":
    gps_path = "data/raw/i2wdd/18_01_25/IMU/GPS_1.csv"
    evaluate_model_real_gps(gps_filepath=gps_path)