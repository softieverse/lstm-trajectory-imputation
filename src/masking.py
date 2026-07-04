"""
masking.py
Simulates missing data in trajectory sequences for the imputation task.
Given a clean sequence, produces a masked version + a record of what was hidden,
so the model has something to learn to fill back in.
"""

import numpy as np


def random_point_mask(sequence: np.ndarray, missing_rate: float = 0.2, seed: int = None) -> tuple:
    """
    Randomly hides individual (x, y) points scattered across the sequence.
    Mimics sensor glitches / random detection failures.

    Args:
        sequence: array of shape (seq_len, 2)
        missing_rate: fraction of points to hide (0.2 = 20%)

    Returns:
        masked_sequence: copy of sequence with missing points set to 0
        mask: boolean array, same shape as sequence's first dim (seq_len,)
              True = missing, False = observed
    """
    rng = np.random.default_rng(seed)
    seq_len = sequence.shape[0]

    mask = rng.random(seq_len) < missing_rate  # True where we'll hide data

    masked_sequence = sequence.copy()
    masked_sequence[mask] = 0.0  # zero out the missing points

    return masked_sequence, mask


def block_mask(sequence: np.ndarray, num_blocks: int = 2, block_size_range: tuple = (5, 15), seed: int = None) -> tuple:
    """
    Hides continuous chunks (blocks) of frames.
    Mimics occlusion — e.g., a vehicle briefly hidden behind another vehicle.

    Args:
        sequence: array of shape (seq_len, 2)
        num_blocks: how many separate gaps to create
        block_size_range: (min, max) length of each gap in frames

    Returns:
        masked_sequence, mask (same format as random_point_mask)
    """
    rng = np.random.default_rng(seed)
    seq_len = sequence.shape[0]

    mask = np.zeros(seq_len, dtype=bool)

    for _ in range(num_blocks):
        block_size = rng.integers(block_size_range[0], block_size_range[1])
        if block_size >= seq_len:
            continue  # skip if block is too big for this sequence
        start = rng.integers(0, seq_len - block_size)
        mask[start:start + block_size] = True

    masked_sequence = sequence.copy()
    masked_sequence[mask] = 0.0

    return masked_sequence, mask


def apply_masking(padded_array: np.ndarray, lengths: np.ndarray, method: str = "random", **kwargs) -> tuple:
    """
    Applies masking across a whole batch of sequences (from preprocess.py's output).

    Args:
        padded_array: shape (num_objects, max_len, 2)
        lengths: real (unpadded) length of each sequence
        method: "random" or "block"

    Returns:
        masked_array: same shape as padded_array, with missing points zeroed
        mask_array: shape (num_objects, max_len), True = missing (only within real length)
    """
    num_objects, max_len, _ = padded_array.shape
    masked_array = padded_array.copy()
    mask_array = np.zeros((num_objects, max_len), dtype=bool)

    for i in range(num_objects):
        real_len = lengths[i]
        real_seq = padded_array[i, :real_len, :]

        if method == "random":
            masked_seq, mask = random_point_mask(real_seq, **kwargs)
        elif method == "block":
            masked_seq, mask = block_mask(real_seq, **kwargs)
        else:
            raise ValueError(f"Unknown masking method: {method}")

        masked_array[i, :real_len, :] = masked_seq
        mask_array[i, :real_len] = mask
        # padded region (beyond real_len) stays mask=False since it's not real data anyway

    return masked_array, mask_array


if __name__ == "__main__":
    from data_loader import load_trajectories
    from preprocess import preprocess_pipeline

    df = load_trajectories(source="synthetic", num_objects=5)
    padded_array, lengths = preprocess_pipeline(df)

    masked_array, mask_array = apply_masking(padded_array, lengths, method="random", missing_rate=0.2, seed=1)

    print(f"Original (first object, first 10 frames):\n{padded_array[0][:10]}")
    print(f"\nMasked (first object, first 10 frames):\n{masked_array[0][:10]}")
    print(f"\nMask (True = missing) for first object, first 10 frames:\n{mask_array[0][:10]}")
    print(f"\nTotal missing points for object 0: {mask_array[0].sum()} out of {lengths[0]} real frames")