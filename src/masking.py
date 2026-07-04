import numpy as np


def random_point_mask(sequence: np.ndarray, missing_rate: float = 0.2, seed: int = None) -> tuple:

    rng = np.random.default_rng(seed)
    seq_len = sequence.shape[0]

    mask = rng.random(seq_len) < missing_rate  

    masked_sequence = sequence.copy()
    masked_sequence[mask] = 0.0  

    return masked_sequence, mask


def block_mask(sequence: np.ndarray, num_blocks: int = 2, block_size_range: tuple = (5, 15), seed: int = None) -> tuple:

    rng = np.random.default_rng(seed)
    seq_len = sequence.shape[0]

    mask = np.zeros(seq_len, dtype=bool)

    for _ in range(num_blocks):
        block_size = rng.integers(block_size_range[0], block_size_range[1])
        if block_size >= seq_len:
            continue  
        start = rng.integers(0, seq_len - block_size)
        mask[start:start + block_size] = True

    masked_sequence = sequence.copy()
    masked_sequence[mask] = 0.0

    return masked_sequence, mask


def apply_masking(padded_array: np.ndarray, lengths: np.ndarray, method: str = "random", **kwargs) -> tuple:

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