from torch.utils.data import WeightedRandomSampler
import torch
import numpy as np

def make_weighted_sampler_from_labels(train_labels):
    """
    Build a WeightedRandomSampler given *train fold* labels only.

    Parameters
    ----------
    train_labels : 1D array-like (list/np.array/torch.Tensor)
        Labels for the training subset (already sliced).
    """
    labels = torch.as_tensor(train_labels, dtype=torch.long)

    # count how many samples per class in THIS fold
    class_sample_count = torch.bincount(labels).float()

    # inverse-frequency weights
    weight_per_class = 1.0 / class_sample_count
    weights = weight_per_class[labels]

    sampler = WeightedRandomSampler(
        weights,
        num_samples=len(weights),
        replacement=True,
    )
    return sampler
