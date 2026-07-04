LSTM Trajectory Imputation

This project implements an LSTM Autoencoder to reconstruct missing points in vehicle trajectory data.

The model is trained on GPS trajectory sequences where some points are intentionally hidden. It learns the movement patterns of the vehicle and predicts the missing locations based on the surrounding trajectory.

The project serves as a baseline implementation before developing a more advanced LSTM-GAIN model for trajectory imputation.

The workflow includes:

Loading and preprocessing trajectory data
Simulating missing trajectory points
Training an LSTM Autoencoder
Evaluating the predicted trajectories using RMSE

The current implementation uses the I2WDD (Indian 2-Wheeler Driving Dataset), which contains real GPS trajectories collected from a motorcycle ride.