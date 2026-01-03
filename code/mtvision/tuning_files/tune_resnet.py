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
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.transforms.functional as TF



from mtvision.data.datasets2 import FullNpyDataset
from mtvision.models.factory import create_model
from mtvision.training.samplers import make_weighted_sampler_from_labels
from mtvision.training.loops import train_one_epoch


# ---------------- GLOBAL CONFIG ----------------
DATA_ROOT = Path(
    "Path/to/dataset"
)
N_SPLITS_CV = 3
MAX_EPOCHS = 40
PATIENCE = 5
MIN_DELTA = 1e-4
MAX_DIST_BATCHES = 100

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


def make_objective(model_name: str, channels: str = "rgb"):
    """
    Create an Optuna objective function for a given model_name.
    Works for:
      - "cnn"
      - "resnet18", "resnet50"
      - "resnet18_last5", "resnet50_last5"
      - "vit_small", "vit_small_last2"
    as long as `create_model(...)` supports them.
    """

    def objective(trial: optuna.Trial) -> float:
        # ---- Build dataset for this channels setting ----
        dataset, labels, groups = build_dataset(channels=channels)
        num_classes = NUM_CLASSES
        in_channels = dataset.num_channels()
        lr_last_blocks = 0

        # ---- Hyperparameters to tune: lr, dropout, batch size, weight_decay, optimizer ----
        # Learning rate for classifier head
        lr_head = trial.suggest_float(
            "lr_head",
            1e-5,
            5e-4,
            log=True
        )
        if model_name != "resnet18_whole_backbone":
            # Learning rate for last residual block
            lr_last_blocks = trial.suggest_float(
                "lr_last_block",
                1e-6,
                1e-4,
                log=True
            )

        dropout_p = trial.suggest_float("dropout_p", 0.0, 0.6)
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "SGD"])
        augmentation_mode = trial.suggest_categorical(
            "augmentation_mode", ["none", "basic"]
        )

        imbalance_strategy = "weighted_ce" #found to be the best model in CNN search
        
        # Strategy: weighted sampler + plain CE (no class weights, no smoothing)
        if imbalance_strategy == "weighted_ce":
            sampler_type = "weighted"
            loss_type = "ce"
            use_bias_init = False
            label_smoothing = 0.0

        focal_gamma = 0 #never used
        alpha_ratio = 0 #never used

        # ---- Alpha vector for focal loss (even if unused in CE branch) ----
        if loss_type == "focal":
            alpha0 = 2.0 / (1.0 + alpha_ratio)
            alpha1 = 2.0 * alpha_ratio / (1.0 + alpha_ratio)
            alpha_vec = np.array([alpha0, alpha1], dtype=np.float32)
        else:
            alpha_vec = None

        print(
        f"\n[TRIAL {trial.number}] "
        f"model={model_name}, channels={channels}, "
        f"lr_head={lr_head:.3e}, lr_last={lr_last_blocks:.3e}, "
        f"wd={weight_decay:.3e}, dropout={dropout_p:.2f}, "
        f"bs={batch_size}, opt={optimizer_name}, sampler={sampler_type}, "
        f"aug={augmentation_mode}, loss={loss_type}, "
        f"label_smoothing={label_smoothing:.3f}, "
        f"imbalance_strategy={imbalance_strategy}"
    )

        # ---- StratifiedGroupKFold: avoid spatial leakage by grouping on block ----
        sgkf = StratifiedGroupKFold(
            n_splits=N_SPLITS_CV, shuffle=True, random_state=SEED
        )
        fold_objs, fold_accs, fold_recalls, fold_epochs = [], [], [], []

        for fold, (train_idx, val_idx) in enumerate(
            sgkf.split(np.zeros_like(labels), labels, groups),
            start=1,
        ):
            print(
                f"[TRIAL {trial.number}] Fold {fold}/{N_SPLITS_CV} "
                f"starting ({len(train_idx)} train, {len(val_idx)} val)"
            )
            base_step = (fold - 1) * MAX_EPOCHS

            # ---- Compute class stats ONLY on training fold ----
            train_labels = labels[train_idx]
            class_counts = np.bincount(train_labels, minlength=num_classes).astype(float)
            print("[INFO] Class counts (train fold):", class_counts)

            # ---- Build model ----
            model = create_model(
                model_name=model_name,
                num_classes=num_classes,
                in_channels=in_channels,
                dropout_p=dropout_p,
            ).to(DEVICE)

            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)

            # --- Build transforms (aug vs no-aug) ---
            if augmentation_mode == "basic":
                train_transform = AugmentAndNormalizeTransform(channels=channels)
            else:
                # No augmentation, but still normalize (ImageNet stats)
                train_transform = NormalizeTransform(channels=channels)

            # Validation: NEVER augment, ALWAYS normalize
            val_transform = NormalizeTransform(channels=channels)

            # Wrap datasets
            train_dataset = TransformDataset(
                base_dataset=train_subset,
                transform=train_transform,
            )

            val_dataset = TransformDataset(
                base_dataset=val_subset,
                transform=val_transform,
            )

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

            dist = inspect_loader_distribution(
                train_loader, max_batches=MAX_DIST_BATCHES
            )
            print(
                f"[TRIAL {trial.number}] Fold {fold} sampled label "
                f"distribution (first {MAX_DIST_BATCHES} batches): {dict(dist)}"
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


            # ---- Loss function ----
            if loss_type == "focal":
                criterion = FocalLoss(
                    num_classes=num_classes,
                    gamma=focal_gamma,
                    alpha=alpha_vec,
                )
            else:
                criterion = nn.CrossEntropyLoss(
                    label_smoothing=label_smoothing,
                )

            #### TRAINGING
            best_fold_obj = -float("inf")
            best_fold_acc = 0.0
            best_fold_recall = 0.0
            best_epoch = 0

            best_val_loss = float("inf")
            no_improve_obj = 0
            no_improve_loss = 0

            for epoch in range(1, MAX_EPOCHS + 1):

                # ---------- TRAIN ----------
                train_loss, train_acc = train_one_epoch(
                    model, train_loader, optimizer, criterion, DEVICE
                )

                # ---------- EVAL ----------
                acc_val, rec_risk, prec_risk, f1_risk = eval_metrics(model, val_loader)
                val_loss = eval_loss(model, val_loader, criterion, DEVICE)

                obj = f1_risk  # objective is F1-risk

                print(
                    f"    [Fold {fold}] Epoch {epoch:02d}/{MAX_EPOCHS} | "
                    f"TrainLoss={train_loss:.4f} | TrainAcc={train_acc:.3f} | "
                    f"ValAcc={acc_val:.3f} | ValLoss={val_loss:.4f} | "
                    f"ValF1_Risk={f1_risk:.3f} | Obj={obj:.3f}"
                )

                # ---------- Optuna pruning ----------
                step = base_step + epoch
                trial.report(obj, step=step)
                if trial.should_prune():
                    print(f"    [Fold {fold}] Trial pruned at epoch {epoch}")
                    raise optuna.TrialPruned()

                # ---------- EARLY STOPPING ----------
                improved_obj = obj > best_fold_obj + MIN_DELTA
                improved_loss = val_loss < best_val_loss - MIN_DELTA

                if improved_obj:
                    best_fold_obj = obj
                    best_fold_acc = acc_val
                    best_fold_recall = rec_risk
                    best_epoch = epoch
                    no_improve_obj = 0
                else:
                    no_improve_obj += 1

                if improved_loss:
                    best_val_loss = val_loss
                    no_improve_loss = 0
                else:
                    no_improve_loss += 1

                # stop only if BOTH metrics are stale
                if no_improve_obj >= PATIENCE and no_improve_loss >= PATIENCE:
                    print(
                        f"    [Fold {fold}] Early stopping at epoch {epoch} "
                        f"(no obj or loss improvement for {PATIENCE} epochs)"
                    )
                    break

            # ---------- fold summary ----------
            print(
                f"[TRIAL {trial.number}] Fold {fold} done | "
                f"Best obj={best_fold_obj:.3f} | Best acc={best_fold_acc:.3f} | "
                f"Best recall={best_fold_recall:.3f} | Best epoch={best_epoch}"
            )

            fold_objs.append(best_fold_obj)
            fold_accs.append(best_fold_acc)
            fold_recalls.append(best_fold_recall)
            fold_epochs.append(best_epoch)

        mean_obj = float(np.mean(fold_objs))
        mean_acc = float(np.mean(fold_accs))
        mean_risk_recall = float(np.mean(fold_recalls))
        mean_best_epoch = float(np.mean(fold_epochs))

        print(
            f"[TRIAL {trial.number}] DONE (model={model_name}) | "
            f"Mean F1_risk={mean_obj:.3f} | "
            f"Mean Acc={mean_acc:.3f} | "
            f"Mean Risk Recall={mean_risk_recall:.3f} | "
            f"Mean Best Epoch={mean_best_epoch:.1f}"
        )

        trial.set_user_attr("mean_acc", mean_acc)
        trial.set_user_attr("mean_risk_recall", mean_risk_recall)
        trial.set_user_attr("mean_best_epoch", mean_best_epoch)

        return mean_obj

    return objective


def main():
    # ---------------- CHOOSE MODEL + CHANNELS ----------------
    # model_name must match what create_model(...) expects.
   
    model_name = "resnet18_last2"

 
    channels = "rgb"

    # ---------------- CREATE OBJECTIVE ----------------
    objective = make_objective(model_name=model_name, channels=channels)

    # ---------------- OPTUNA STUDY CONFIG ----------------
    storage_path = "sqlite:///resnet18_last2_10_12_tuning.db"   
    study = optuna.create_study(
        study_name=f"{model_name}_tuning",
        direction="maximize",
        storage=storage_path,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.NopPruner(),  # <--- no pruning, for cnn
    )

    # ---------------- RUN OPTIMIZATION ----------------
    n_trials = 20  # e.g. 15–20 for CNN as we discussed
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # ---------------- REPORT RESULTS ----------------
    print("\n========== BEST TRIAL ==========")
    print("Best objective (mean val F1_risk):", study.best_value)
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # Save all trials to CSV for later analysis
    df = study.trials_dataframe()
    df.to_csv(f"{model_name}_tuning.csv", index=False)
    print(f"[INFO] Saved results to {model_name}_tuning.csv")
    print("[INFO] Study DB:", storage_path)

    # Optional: hyperparameter importance
    print("\n========== HYPERPARAMETER IMPORTANCE ==========")
    importances = get_param_importances(study)
    for name, val in importances.items():
        print(f"{name}: {val:.4f}")


if __name__ == "__main__":
    main()
