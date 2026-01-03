from pathlib import Path
from typing import List, Tuple, Optional, Union, Callable
import numpy as np
import torch
from torch.utils.data import Dataset


class FullNpyDataset(Dataset):
    """
    Loads .npy tiles from a block-based structure like:

      root/
        dataset_0000/
          NoRisk/*.npy
          Risk/*.npy
        dataset_0001/
          NoRisk/*.npy
          Risk/*.npy
        ...

    or, if there are no dataset_XXXX folders, falls back to:

      root/
        NoRisk/*.npy
        Risk/*.npy

    Args:
        root: OUT_ROOT folder.

        split:
            "all"   -> use all dataset_XXXX blocks
            "train" -> use all blocks except the fixed test range
            "test"  -> use only blocks in TEST_BLOCK_RANGE (default 0–9)

            NOTE: if `blocks` is provided, it takes precedence over `split`.

        blocks:
            Optional explicit list of block names, e.g. ["dataset_0000", "dataset_0001"].
            If given, this overrides `split` and uses exactly these blocks.

        channels:
            "rgb"       -> return 3-channel tensors (first 3 channels)
            "rgb_ndvi"  -> return all channels as stored (e.g. 4: RGB+NDVI)

        transform:
            Optional callable applied to the image tensor x (C,H,W) before returning.
            Use this for normalization / data augmentation.
    """

    # Default fixed range for spatial test split (by dataset index)
    TEST_BLOCK_RANGE: Tuple[int, int] = (0, 9)  # dataset_0000–dataset_0009

    def __init__(
        self,
        root: Union[str, Path],
        split: str = "all",
        blocks: Optional[List[str]] = None,
        channels: str = "rgb_ndvi",
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        self.root = Path(root)
        self.samples: List[Tuple[Path, int]] = []
        self.channels = channels
        self.transform = transform
        assert self.channels in ("rgb", "rgb_ndvi"), "channels must be 'rgb' or 'rgb_ndvi'"
        assert split in ("all", "train", "test"), "split must be 'all', 'train' or 'test'"
        self.split = split

        # ----- Detect block structure: dataset_XXXX subfolders -----
        block_dirs = [
            d for d in self.root.iterdir()
            if d.is_dir() and d.name.startswith("dataset_")
        ]

        # Helper to get numeric index from "dataset_XXXX"
        def block_index(d: Path) -> int:
            return int(d.name.split("_")[-1])

        if block_dirs:
            # If explicit blocks are provided, they override split logic
            if blocks is not None:
                allowed = set(blocks)
                block_dirs = [d for d in block_dirs if d.name in allowed]
                if not block_dirs:
                    raise RuntimeError(f"No matching blocks found for: {blocks}")
            else:
                # Apply split-based selection
                if self.split == "all":
                    # use all block_dirs as-is
                    pass
                elif self.split == "test":
                    lo, hi = self.TEST_BLOCK_RANGE
                    block_dirs = [
                        d for d in block_dirs
                        if lo <= block_index(d) <= hi
                    ]
                elif self.split == "train":
                    lo_t, hi_t = self.TEST_BLOCK_RANGE
                    block_dirs = [
                        d for d in block_dirs
                        if not (lo_t <= block_index(d) <= hi_t)
                    ]

                if not block_dirs:
                    raise RuntimeError(f"No block directories selected for split='{self.split}'")

            # Collect all class names across blocks
            class_names = sorted({
                sub.name
                for b in block_dirs
                for sub in b.iterdir()
                if sub.is_dir()
            })
            self.class_to_idx = {c: i for i, c in enumerate(class_names)}

            # Build sample list: (path, label_idx)
            for b in block_dirs:
                for cls_name, idx in self.class_to_idx.items():
                    cls_dir = b / cls_name
                    if not cls_dir.exists():
                        continue
                    for f in cls_dir.glob("*.npy"):
                        self.samples.append((f, idx))

        else:
            # ----- Fallback: simple layout root/class/*.npy -----
            classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
            self.class_to_idx = {c: i for i, c in enumerate(classes)}

            for cls_name, idx in self.class_to_idx.items():
                cls_dir = self.root / cls_name
                for f in cls_dir.glob("*.npy"):
                    self.samples.append((f, idx))

        if not self.samples:
            raise RuntimeError(f"No .npy files found under {self.root} for split='{self.split}'")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        arr = np.load(path)

        # Ensure (C,H,W)
        if arr.ndim == 2:
            # (H,W) -> (1,H,W)
            x_np = arr[None, ...]
        elif arr.ndim == 3:
            # Either (H,W,C) or (C,H,W)
            if arr.shape[0] in (1, 3, 4) and arr.shape[0] <= arr.shape[-1]:
                # Assume (C,H,W)
                x_np = arr
            elif arr.shape[-1] in (1, 3, 4):
                # Assume (H,W,C) -> (C,H,W)
                x_np = np.transpose(arr, (2, 0, 1))
            else:
                raise ValueError(f"Unexpected array shape {arr.shape} in {path}")
        else:
            raise ValueError(f"Unexpected array shape {arr.shape} in {path}")

        # ---- Channel selection: RGB vs RGB+NDVI ----
        if self.channels == "rgb":
            if x_np.shape[0] < 3:
                raise ValueError(f"Requested RGB (3 ch) but only {x_np.shape[0]} channels in {path}")
            x_np = x_np[:3, ...]      # take first 3 channels (assumed RGB)
        elif self.channels == "rgb_ndvi":
            # keep all channels as they are (e.g. 4: R,G,B,NDVI)
            pass

        x = torch.from_numpy(x_np).float()
        y = torch.tensor(label, dtype=torch.long)

        # ---- Apply optional transform / augmentation ----
        if self.transform is not None:
            x = self.transform(x)

        return x, y

    def num_classes(self) -> int:
        return len(self.class_to_idx)

    def num_channels(self) -> int:
        x, _ = self[0]
        return x.shape[0]
