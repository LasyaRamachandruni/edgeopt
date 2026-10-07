"""CIFAR-10 loading shared by eval, calibration and fine-tuning.

Tries torchvision first (downloads from cs.toronto.edu), then the Hugging Face
dataset "uoft-cs/cifar10". Images are resized to the model input size and
normalised with ImageNet statistics, matching how the checkpoint was trained.
"""

import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, Subset

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def cifar10_transform(img_size=224, train=False):
    ops = [T.Resize((img_size, img_size))]
    if train:
        ops.append(T.RandomHorizontalFlip())
    ops += [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return T.Compose(ops)


class _HFCifar10(Dataset):
    def __init__(self, split, transform, cache_dir=None):
        from datasets import load_dataset

        self.ds = load_dataset("uoft-cs/cifar10", split=split, cache_dir=cache_dir)
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        row = self.ds[i]
        return self.transform(row["img"].convert("RGB")), row["label"]


def load_cifar10(train=False, img_size=224, root="./data", augment=False):
    transform = cifar10_transform(img_size, train=train and augment)
    errors = []
    try:
        import torchvision

        return torchvision.datasets.CIFAR10(root=root, train=train, download=True, transform=transform)
    except Exception as e:  # network blocked, mirror down, ...
        errors.append(f"torchvision: {e}")
    try:
        return _HFCifar10("train" if train else "test", transform, cache_dir=f"{root}/hf")
    except Exception as e:
        errors.append(f"huggingface uoft-cs/cifar10: {e}")
    raise RuntimeError("Could not load CIFAR-10.\n" + "\n".join(errors))


def fixed_subset(dataset, n, seed=0):
    """Deterministic random subset of n images (same indices every run)."""
    if n is None or n >= len(dataset):
        return dataset
    idx = np.random.RandomState(seed).permutation(len(dataset))[:n]
    return Subset(dataset, sorted(idx.tolist()))


def numpy_batches(dataset, batch_size=64):
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for images, labels in loader:
        yield images.numpy(), labels.numpy()
