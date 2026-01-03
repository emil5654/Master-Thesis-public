import sys
from collections import Counter
from pathlib import Path
from typing import Tuple

import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
from optuna.importance import get_param_importances
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.transforms.functional as TF
import copy
import json


from mtvision.data.datasets2 import FullNpyDataset
from mtvision.models.factory import create_model
from mtvision.training.samplers import make_weighted_sampler_from_labels
from mtvision.training.loops import train_one_epoch

# ---------------- GLOBAL CONFIG ----------------
DATA_ROOT = Path(
    "path/to/your/data/root"  # <-- CHANGE THIS TO YOUR DATA PATH
)
N_SPLITS_CV = 3
MAX_EPOCHS = 40 # found in hyperparameter tuning aswell
PATIENCE = 5
MIN_DELTA = 1e-4
MAX_DIST_BATCHES = 100
VAL_FRACTION = 0.1

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[INFO] Device:", DEVICE)


# ---------------- DATASET ----------------
def build_dataset(channels: str = "rgb"):
    """
    Build full tile dataset + labels + block groups for CV.

    Uses only the TRAIN pool:
      - FullNpyDataset(root, split="train") excludes the fixed test blocks
        (dataset_0000–dataset_0009) based on the TEST_BLOCK_RANGE inside
        FullNpyDataset.

    groups[i] = dataset_XXXX block name so we can use StratifiedGroupKFold
    to avoid spatial leakage (all tiles from a block stay in the same fold).

    Parameters
    ----------
    channels : {"rgb"}
        Which input channels to load.
    """
    dataset = FullNpyDataset(DATA_ROOT, split="train", channels=channels)

    # labels from stored samples list: (path, label_idx)
    labels = np.array([lbl for _, lbl in dataset.samples], dtype=int)

    # derive block name from path:
    # .../Datasets_224x224_stride128/dataset_0007/Risk/patch_xxx.npy
    block_names = []
    for path, _ in dataset.samples:
        # parent = class dir, parent.parent = dataset_XXXX (if present)
        if path.parent.parent.name.startswith("dataset_"):
            block_names.append(path.parent.parent.name)
        else:
            # fallback if no dataset_XXXX structure is present
            block_names.append("all_tiles")

    groups = np.array(block_names)

    print(f"[INFO] Loaded TRAIN dataset from {DATA_ROOT} (channels={channels})")
    print(f"[INFO] Total samples (train pool): {len(dataset)}")
    print(f"[INFO] Classes: {dataset.class_to_idx}")
    print(f"[INFO] Channels per sample: {dataset.num_channels()}")
    #print(f"[INFO] Unique groups (blocks) in train pool: {sorted(set(groups))}")

    return dataset, labels, groups


NUM_CLASSES = 2  # fixed
RISK_INDEX = 1


# ---------------- FIXED BEST CNN HYPERPARAMETERS ---------------- -> ALSO REMEMBER TO LOSEN LR FOR PRETUNING MODELS 
BEST_WEIGHT_DECAY = 0 #not yet found
BEST_OPTIMIZER_NAME = 'none' #not yet found
BEST_SAMPLER_TYPE = "none" #not yet found
BEST_AUGMENTATION_MODE = "none" #not yet found
BEST_LOSS_TYPE = "none" #not yet found
BEST_FOCAL_GAMMA = 0 #not yet found
BEST_ALPHA_RATIO = 0 #not yet found


# ---------------- NORMALIZATION TRANSFORMS ----------------
class NormalizeTransform:
    """
    Normalizes input tensors using ImageNet mean/std for RGB.
    Works for both CNNs and ViT models.

    x is expected to be a tensor with shape [C, H, W].
    """

    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __init__(self, channels="rgb"):
        self.channels = channels

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()

        # Ensure [0,1] range
        if x.max() > 1.5:
            x = x / 255.0

        if self.channels == "rgb":
            return (x - self.IMAGENET_MEAN.to(x.device)) / self.IMAGENET_STD.to(x.device)

        elif self.channels == "rgb_ndvi":
            # Normalize RGB channels, leave NDVI as-is or apply custom NDVI normalization
            rgb = (x[:3] - self.IMAGENET_MEAN.to(x.device)) / self.IMAGENET_STD.to(x.device)
            ndvi = x[3:].clamp(-1.0, 1.0)   # typical NDVI range
            return torch.cat([rgb, ndvi], dim=0)

        else:
            raise ValueError(f"Unknown channel mode: {self.channels}")
            

class SimpleScaleTransform:
    """
    Scales uint8 tiles to float32 in the [0,1] range.
    No normalization is applied.
    """
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        if x.max() > 1.5:
            x = x / 255.0
        return x
    
class AugmentAndNormalizeTransform:
    """
    Applies basic data augmentation to ALL samples, then normalizes using
    ImageNet mean/std for RGB (and leaves NDVI untouched if present).

    Expected input: x with shape [C, H, W], values in [0,1] or [0,255].
    """
    def __init__(
        self,
        channels: str = "rgb",
        p_hflip: float = 0.5,
        p_vflip: float = 0.5,
        p_rot90: float = 0.5,
        p_color_jitter: float = 0.3,
        brightness: float = 0.1,
        contrast: float = 0.1,
        saturation: float = 0.1,
        hue: float = 0.05,
    ):
        self.channels = channels
        self.p_hflip = p_hflip
        self.p_vflip = p_vflip
        self.p_rot90 = p_rot90
        self.p_color_jitter = p_color_jitter
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

        # Reuse your existing normalizer
        self.normalizer = NormalizeTransform(channels=channels)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()

        # Ensure [0,1]
        if x.max() > 1.5:
            x = x / 255.0

        # --- geometric augmentations (apply to all channels) ---
        # x is [C, H, W]
        if torch.rand(1) < self.p_hflip:
            x = torch.flip(x, dims=[2])  # flip width axis

        if torch.rand(1) < self.p_vflip:
            x = torch.flip(x, dims=[1])  # flip height axis

        if torch.rand(1) < self.p_rot90:
            # Rotate 90 degrees: transpose H and W, then flip one axis
            x = x.transpose(1, 2).flip(2)

        # --- color jitter only on RGB channels ---
        if self.channels == "rgb" and torch.rand(1) < self.p_color_jitter:
            # TF.* expect [C, H, W] in [0,1]
            x = TF.adjust_brightness(x, 1.0 + (torch.empty(1).uniform_(-self.brightness, self.brightness)).item())
            x = TF.adjust_contrast(x, 1.0 + (torch.empty(1).uniform_(-self.contrast, self.contrast)).item())
            x = TF.adjust_saturation(x, 1.0 + (torch.empty(1).uniform_(-self.saturation, self.saturation)).item())
            x = TF.adjust_hue(x, (torch.empty(1).uniform_(-self.hue, self.hue)).item())

        # For "rgb_ndvi", we keep the same geometric transforms (already applied),
        # but we *do not* do color jitter on NDVI.

        # Finally, normalize (ImageNet stats for RGB, NDVI handled inside)
        x = self.normalizer(x)
        return x

    

# ---------------- TRANSFORM WRAPPER ----------------
class TransformDataset(Dataset):
    """
    Wraps a base dataset and applies a transform to each sample.
    """
    def __init__(self, base_dataset, transform=None):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        x, y = self.base_dataset[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x, y


# ---------------- FOCAL LOSS ----------------
class FocalLoss(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        gamma: float = 2.0,
        alpha=None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.reduction = reduction

        if alpha is None:
            self.alpha = None
        else:
            alpha = torch.tensor(alpha, dtype=torch.float32)
            assert alpha.numel() == num_classes
            self.alpha = alpha

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(inputs, dim=1)
        probs = log_probs.exp()

        targets = targets.view(-1, 1)
        log_pt = log_probs.gather(1, targets).squeeze(1)
        pt = probs.gather(1, targets).squeeze(1)

        if self.alpha is not None:
            alpha_t = self.alpha.to(inputs.device).gather(
                0, targets.squeeze(1)
            )
        else:
            alpha_t = 1.0

        loss = -alpha_t * (1 - pt) ** self.gamma * log_pt

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
    
def eval_loss(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    return total_loss / total_samples



# ---------------- UTILS ----------------
def inspect_loader_distribution(
    loader: DataLoader, max_batches: int | None = None
) -> Counter:
    counter = Counter()
    for i, (_, yb) in enumerate(loader):
        counter.update(yb.tolist())
        if max_batches is not None and (i + 1) >= max_batches:
            break
    return counter


@torch.no_grad()
def eval_metrics(
    model: nn.Module, loader: DataLoader
) -> Tuple[float, float, float, float]:
    model.eval()
    total = correct = tp = fp = fn = 0

    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        preds = model(xb).argmax(dim=1)

        total += yb.size(0)
        correct += (preds == yb).sum().item()

        risk_true = yb == RISK_INDEX
        risk_pred = preds == RISK_INDEX

        tp += (risk_true & risk_pred).sum().item()
        fp += (~risk_true & risk_pred).sum().item()
        fn += (risk_true & ~risk_pred).sum().item()

    acc = correct / total if total > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return acc, recall, precision, f1


def init_classifier_bias_with_priors(model: nn.Module, class_counts: np.ndarray):
    total = class_counts.sum()
    priors = class_counts / total
    priors = np.clip(priors, 1e-6, 1.0)
    log_priors = torch.log(torch.tensor(priors, dtype=torch.float32))

    last_linear = None
    for m in model.modules():
        if isinstance(m, nn.Linear) and m.out_features == len(priors):
            last_linear = m

    if last_linear is not None and last_linear.bias is not None:
        with torch.no_grad():
            last_linear.bias.data = log_priors.to(last_linear.bias.device)
        print("[INFO] Initialized classifier bias with log class priors:", priors)
    else:
        print("[WARN] Could not find final Linear layer to init bias.")

def collect_probs_and_labels(model: nn.Module, loader: DataLoader, device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():   # <---- prevents grad tracking globally
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            probs = F.softmax(logits, dim=1)[:, RISK_INDEX]  # P(class == Risk)

            all_probs.append(probs.detach().cpu().numpy())
            all_labels.append(yb.detach().cpu().numpy())

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    return all_probs, all_labels
# ---------------- FINAL TRAIN + EARLY STOP + TEST ----------------
def train_full_and_eval_test(
    model_name: str = "resnet18_whole_backbone",
    channels: str = "rgb",
    save_path: str = "resnet18_whole_backbone_rgb_final.pth",
    results_path: str = "resnet18_whole_backbone_rgb_final_results.json",
):
    """
    1) Split TRAIN pool into train_final + val_final (block-wise) for early stopping.
    2) Train a single model on train_final with early stopping on val_final.
    3) Evaluate once on TEST split (split='test') using:
       - argmax (threshold = 0.5)
       - manual_threshold (from CV)
    4) Save model + threshold + results.
    """
    # ---------------- BUILD TRAIN DATASET ----------------
    train_dataset_full, all_labels, groups = build_dataset(channels=channels)
    num_classes = NUM_CLASSES
    in_channels = train_dataset_full.num_channels()

    print(f"[INFO] Full TRAIN pool size: {len(train_dataset_full)}")
    print(f"[INFO] Label distribution in TRAIN: {np.bincount(all_labels)}")

    # ---------------- TRAIN/VAL SPLIT (GROUP-BASED) ----------------
    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=VAL_FRACTION,
        random_state=SEED,
    )
    train_idx, val_idx = next(gss.split(
        np.zeros_like(all_labels), all_labels, groups
    ))

    train_labels = all_labels[train_idx]
    val_labels = all_labels[val_idx]

    print(f"[INFO] Train_final size: {len(train_idx)}")
    print(f"[INFO] Val_final size  : {len(val_idx)}")
    print(f"[INFO] Train label dist: {np.bincount(train_labels)}")
    print(f"[INFO] Val label dist  : {np.bincount(val_labels)}")

    train_subset = Subset(train_dataset_full, train_idx)
    val_subset = Subset(train_dataset_full, val_idx)

    # ---------------- FIXED HYPERPARAMETERS ----------------
    lr_head =  0.0002401096843418529
    lr_last_blocks = 9.190576652999549e-05
    dropout_p =  0.2887606627213892
    batch_size = 8
    weight_decay =0.00607070151418308
    optimizer_name = "Adam"
    augmentation_mode = "basic"
    sampler_type = "weighted"
    label_smoothing = 0.0
    use_bias_init = False

    # ---------------- MODEL ----------------
    model = create_model(
        model_name=model_name,
        num_classes=num_classes,
        in_channels=in_channels,
        dropout_p=dropout_p,
    ).to(DEVICE)

    if use_bias_init:
        class_counts = np.bincount(train_labels, minlength=num_classes).astype(float)
        init_classifier_bias_with_priors(model, class_counts)
    else:
        print("[INFO] Skipping classifier bias init.")

    # ---------------- DATA TRANSFORMS + LOADERS ----------------
    if augmentation_mode == "basic":
        train_transform = AugmentAndNormalizeTransform(channels=channels)
    else:
        train_transform = NormalizeTransform(channels=channels)
    val_transform = NormalizeTransform(channels=channels)

    train_dataset = TransformDataset(train_subset, transform=train_transform)
    val_dataset = TransformDataset(val_subset, transform=val_transform)

    if sampler_type == "weighted":
        sampler = make_weighted_sampler_from_labels(train_labels)
        shuffle_flag = False
    else:
        sampler = None
        shuffle_flag = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle_flag if sampler is None else False,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    if model_name == "resnet18_whole_backbone":
                    # Only fine-tune last residual block + classifier
                param_groups = [
                    {"params": model.parameters(),     "lr": lr_head},
                ]
                print("One using one LR")
    else:
        # Only fine-tune last residual block + classifier
        param_groups = [
            {"params": model.backbone.fc.parameters(),     "lr": lr_head},
            {"params": model.backbone.layer4.parameters(), "lr": lr_last_blocks},
            {"params": model.backbone.layer3.parameters(), "lr": lr_last_blocks}
        ]

    


    # ---- Optimizer selection (per fold) ---- now works with multiple Lrs
    if optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=weight_decay,
        )
    elif optimizer_name == "Adam":
        optimizer = torch.optim.Adam(
            param_groups,
            weight_decay=weight_decay,
        )
    elif optimizer_name == "SGD":
        optimizer = torch.optim.SGD(
            param_groups,
            weight_decay=weight_decay,
            momentum=0.9,
            nesterov=True,
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # ---------------- TRAINING LOOP WITH EARLY STOPPING ----------------
    print("\n[TRAIN] Starting final training with early stopping")
    best_val_f1 = -float("inf")
    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    no_improve_obj = 0
    no_improve_loss = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, DEVICE
        )

        acc_val, rec_risk, prec_risk, f1_risk = eval_metrics(model, val_loader)
        val_loss = eval_loss(model, val_loader, criterion, DEVICE)

        print(
            f"[TRAIN] Epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"TrainLoss={train_loss:.4f} | TrainAcc={train_acc:.3f} | "
            f"ValAcc={acc_val:.3f} | ValLoss={val_loss:.4f} | "
            f"ValF1_Risk={f1_risk:.3f}"
        )

        improved_obj = f1_risk > best_val_f1 + MIN_DELTA
        improved_loss = val_loss < best_val_loss - MIN_DELTA

        if improved_obj:
            best_val_f1 = f1_risk
            best_state = copy.deepcopy(model.state_dict())
            no_improve_obj = 0
        else:
            no_improve_obj += 1

        if improved_loss:
            best_val_loss = val_loss
            no_improve_loss = 0
        else:
            no_improve_loss += 1

        if no_improve_obj >= PATIENCE and no_improve_loss >= PATIENCE:
            print(f"[TRAIN] Early stopping at epoch {epoch}")
            break

    print("[TRAIN] Loading best validation weights...")
    model.load_state_dict(best_state)
    print("[TRAIN] Finished training with early stopping.")

    # ---------------- BUILD TEST DATASET ----------------
    test_dataset_base = FullNpyDataset(DATA_ROOT, split="test", channels=channels)
    print(f"[INFO] TEST pool size: {len(test_dataset_base)}")

    test_transform = NormalizeTransform(channels=channels)
    test_dataset = TransformDataset(test_dataset_base, transform=test_transform)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # ---------------- EVALUATE ARGMAX (THRESHOLD 0.5) ----------------
    acc_argmax, recall_argmax, prec_argmax, f1_argmax = eval_metrics(model, test_loader)

    # ---------------- EVALUATE WITH MANUAL THRESHOLD ----------------
    p_risk_test, y_true_test = collect_probs_and_labels(model, test_loader, DEVICE)



    print("\n========== TEST PERFORMANCE ==========")
    print("---- Argmax (threshold = 0.5) ----")
    print(f"Acc          : {acc_argmax:.3f}")
    print(f"Recall(Risk) : {recall_argmax:.3f}")
    print(f"Precision    : {prec_argmax:.3f}")
    print(f"F1_risk      : {f1_argmax:.3f}")

    # ---------------- COLLECT RESULTS ----------------
    results = {
        "model_name": model_name,
        "channels": channels,
        "val_fraction": VAL_FRACTION,
        "test_metrics_argmax": {
            "accuracy": float(acc_argmax),
            "recall_risk": float(recall_argmax),
            "precision_risk": float(prec_argmax),
            "f1_risk": float(f1_argmax),
        }}

    # ---------------- SAVE MODEL + THRESHOLD + RESULTS ----------------
    print(f"\n[SAVE] Saving model + threshold + results to: {save_path}")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "results": results,
        },
        save_path,
    )

    print(f"[SAVE] Saving results JSON to: {results_path}")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    print("[DONE] Final model training + test evaluation complete.")
    return results

    
if __name__ == "__main__":
    results = train_full_and_eval_test(
        model_name="resnet18_last2",
        channels="rgb",
        save_path="resnet18_last2_11_12.pth",
        results_path="resnet18_last2_11_12_final_results.json",
    )
    print(results)
