import torch
import torch.nn as nn
import numpy as np

from data_loader import load_trajectories, load_i2wdd_gps
from preprocess import preprocess_pipeline, segment_and_normalize_gps
from masking import apply_masking
from model import LSTMImputer


def masked_mse_loss(predictions: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:

    mask_expanded = mask.unsqueeze(-1).expand_as(predictions).float()

    squared_error = (predictions - targets) ** 2
    masked_error = squared_error * mask_expanded

    loss = masked_error.sum() / (mask_expanded.sum() + 1e-8)
    return loss


def prepare_data(num_objects: int = 100, missing_rate: float = 0.2, seed: int = 42):

    df = load_trajectories(source="synthetic", num_objects=num_objects, seed=seed)
    padded_array, lengths = preprocess_pipeline(df)

    masked_array, mask_array = apply_masking(
        padded_array, lengths, method="random", missing_rate=missing_rate, seed=seed
    )

    original = torch.tensor(padded_array, dtype=torch.float32)
    masked = torch.tensor(masked_array, dtype=torch.float32)
    mask = torch.tensor(mask_array, dtype=torch.bool)

    return original, masked, mask, lengths


def train_val_split(original, masked, mask, val_fraction: float = 0.2, seed: int = 42):

    num_objects = original.shape[0]
    rng = np.random.default_rng(seed)
    indices = rng.permutation(num_objects)

    val_size = int(num_objects * val_fraction)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    train_data = (original[train_idx], masked[train_idx], mask[train_idx])
    val_data = (original[val_idx], masked[val_idx], mask[val_idx])

    return train_data, val_data


def train_model(num_epochs: int = 100, num_objects: int = 100, missing_rate: float = 0.2, lr: float = 0.001):

    original, masked, mask, lengths = prepare_data(num_objects=num_objects, missing_rate=missing_rate)
    (train_orig, train_masked, train_mask), (val_orig, val_masked, val_mask) = train_val_split(original, masked, mask)

    model = LSTMImputer(input_size=2, hidden_size=64, num_layers=2, dropout=0.2)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"Train objects: {train_orig.shape[0]}, Val objects: {val_orig.shape[0]}")
    print(f"Starting training for {num_epochs} epochs...\n")

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        predictions = model(train_masked, train_mask)
        loss = masked_mse_loss(predictions, train_orig, train_mask)

        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_predictions = model(val_masked, val_mask)
            val_loss = masked_mse_loss(val_predictions, val_orig, val_mask)

        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {loss.item():.6f} | Val Loss: {val_loss.item():.6f}")

    print("\nTraining complete.")
    return model, (val_orig, val_masked, val_mask)


def train_model_real_gps(gps_filepath: str, num_epochs: int = 100, window_size: int = 100,
                          stride: int = 50, missing_rate: float = 0.2, lr: float = 0.001):

    real_traj = load_i2wdd_gps(gps_filepath)
    padded_array, lengths = segment_and_normalize_gps(real_traj, window_size=window_size, stride=stride)

    masked_array, mask_array = apply_masking(
        padded_array, lengths, method="random", missing_rate=missing_rate, seed=42
    )

    original = torch.tensor(padded_array, dtype=torch.float32)
    masked = torch.tensor(masked_array, dtype=torch.float32)
    mask = torch.tensor(mask_array, dtype=torch.bool)

    (train_orig, train_masked, train_mask), (val_orig, val_masked, val_mask) = train_val_split(original, masked, mask)

    model = LSTMImputer(input_size=2, hidden_size=64, num_layers=2, dropout=0.2)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"Real GPS - Train windows: {train_orig.shape[0]}, Val windows: {val_orig.shape[0]}")
    print(f"Starting training for {num_epochs} epochs...\n")

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        predictions = model(train_masked, train_mask)
        loss = masked_mse_loss(predictions, train_orig, train_mask)

        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_predictions = model(val_masked, val_mask)
            val_loss = masked_mse_loss(val_predictions, val_orig, val_mask)

        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {loss.item():.6f} | Val Loss: {val_loss.item():.6f}")

    print("\nTraining complete.")
    torch.save(model.state_dict(), "results/lstm_imputer_real_gps.pth")
    print("Model saved to results/lstm_imputer_real_gps.pth")
    return model, (val_orig, val_masked, val_mask)


if __name__ == "__main__":

    gps_path = "data/raw/i2wdd/18_01_25/IMU/GPS_1.csv"
    model, val_data = train_model_real_gps(gps_filepath=gps_path, num_epochs=300, missing_rate=0.2)