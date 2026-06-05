from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm.auto import tqdm

# -------------------------
# Config
# -------------------------

DATASET_ROOT = "chlorophyll_dataset"

IMAGE_SIZE = (512, 1024)
BATCH_SIZE = 2
EPOCHS = 50
LR = 1e-4
TEST_RATIO = 0.2
SEED = 42
PATIENCE = 12

MODES = {
    "rgb_only": 3,
    # "spectral_only": 2,
    # "all_5_bands": 5,
    "three_branch_3_1_1": None,
    "three_branch_attention_3_1_1": None,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
np.random.seed(SEED)


# -------------------------
# Dataset
# -------------------------


class ChlorophyllDataset(Dataset):
    def __init__(self, root, mode, image_size=(512, 1024)):
        self.root = Path(root)
        self.df = pd.read_csv(self.root / "metadata.csv")
        self.mode = mode
        self.image_size = image_size

    def __len__(self):
        return len(self.df)

    def resize(self, x):
        return F.interpolate(
            x.unsqueeze(0),
            size=self.image_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        data = np.load(self.root / row["sample"])

        image = torch.from_numpy(data["image"]).float()  # [5, H, W]
        y = torch.tensor(data["label"], dtype=torch.float32)

        rgb = image[:3]
        spectral = image[3:5]

        band_660 = image[3]
        band_735 = image[4]
        ndci = (band_735 - band_660) / (band_735 + band_660 + 1e-6)
        ndci = ndci.unsqueeze(0)

        if self.mode == "rgb_only":
            x = rgb

        elif self.mode == "spectral_only":
            x = spectral

        elif self.mode == "all_5_bands":
            x = image

        elif self.mode == "ndci_only":
            x = ndci

        elif self.mode == "spectral_plus_ndci":
            x = torch.cat([spectral, ndci], dim=0)

        elif self.mode == "all_plus_ndci":
            x = torch.cat([image, ndci], dim=0)

        elif self.mode in {
            "three_branch_3_1_1",
            "three_branch_attention_3_1_1",
        }:
            return (
                self.resize(rgb),
                self.resize(spectral[0:1]),
                self.resize(spectral[1:2]),
            ), y

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        return self.resize(x), y


# -------------------------
# Models
# -------------------------


def make_norm(channels):
    return nn.GroupNorm(num_groups=4, num_channels=channels)


def make_encoder(in_channels):
    def block(cin, cout):
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            make_norm(cout),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    return nn.Sequential(
        block(in_channels, 16),
        block(16, 32),
        block(32, 64),
        nn.Conv2d(64, 128, 3, padding=1, bias=False),
        make_norm(128),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
    )


class ChlorophyllCNN(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1, bias=False),
            make_norm(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1, bias=False),
            make_norm(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            make_norm(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            make_norm(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.regressor(self.features(x)).squeeze(1)


class ChlorophyllThreeBranchCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.rgb_branch = make_encoder(3)
        self.spectral_660 = make_encoder(1)
        self.spectral_735 = make_encoder(1)

        self.regressor = nn.Sequential(
            nn.Linear(128 * 3, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x_rgb, x_660, x_735 = x

        f_rgb = self.rgb_branch(x_rgb)
        f_660 = self.spectral_660(x_660)
        f_735 = self.spectral_735(x_735)

        f = torch.cat([f_rgb, f_660, f_735], dim=1)

        return self.regressor(f).squeeze(1)


class ChlorophyllThreeBranchAttentionCNN(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4):
        super().__init__()

        self.rgb_branch = make_encoder(3)
        self.spectral_660 = make_encoder(1)
        self.spectral_735 = make_encoder(1)

        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True,
        )

        self.attn_norm = nn.LayerNorm(embed_dim)

        self.regressor = nn.Sequential(
            nn.Linear(embed_dim * 3, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x_rgb, x_660, x_735 = x

        f_rgb = self.rgb_branch(x_rgb)  # [B, 128]
        f_660 = self.spectral_660(x_660)  # [B, 128]
        f_735 = self.spectral_735(x_735)  # [B, 128]

        tokens = torch.stack([f_rgb, f_660, f_735], dim=1)  # [B, 3, 128]

        attn_out, _ = self.attn(tokens, tokens, tokens)  # [B, 3, 128]
        tokens = self.attn_norm(tokens + attn_out)  # residual + norm

        fused = tokens.flatten(1)  # [B, 384]

        return self.regressor(fused).squeeze(1)


# -------------------------
# Train / eval helpers
# -------------------------


def move_to_device(x, device):
    if isinstance(x, (tuple, list)):
        return tuple(t.to(device) for t in x)
    return x.to(device)


def train_one_epoch(
    model, loader, optimizer, loss_fn, device, desc="Training"
):
    model.train()
    total_loss = 0.0

    pbar = tqdm(loader, desc=desc, leave=False)

    for batch_idx, (x, y) in enumerate(pbar, start=1):
        x = move_to_device(x, device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)

        pred = model(x)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        avg_loss = total_loss / (batch_idx * loader.batch_size)

        pbar.set_postfix(
            {
                "batch_loss": f"{loss.item():.4f}",
                "avg_loss": f"{avg_loss:.4f}",
            }
        )

    return total_loss / len(loader.dataset)


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()

    y_true = []
    y_pred = []

    for x, y in loader:
        x = move_to_device(x, device)
        pred = model(x).cpu().numpy()

        y_pred.extend(pred)
        y_true.extend(y.numpy())

    return np.array(y_true), np.array(y_pred)


def regression_metrics(y_true, y_pred, eps=1e-6):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    abs_error = np.abs(y_pred - y_true)

    mae = abs_error.mean()
    mse = np.mean((y_pred - y_true) ** 2)
    rmse = np.sqrt(mse)

    mape = np.mean(abs_error / (np.abs(y_true) + eps)) * 100
    mape_accuracy = 100 - mape

    mean_true = np.mean(np.abs(y_true)) + eps
    mean_normalized_accuracy = 100 * (1 - mae / mean_true)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2) + eps
    r2 = 1 - ss_res / ss_tot

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "mape_%": mape,
        "mape_accuracy_%": mape_accuracy,
        "mean_normalized_accuracy_%": mean_normalized_accuracy,
        "r2": r2,
    }


@torch.no_grad()
def evaluate_metrics(model, loader, device):
    y_true, y_pred = collect_predictions(model, loader, device)
    metrics = regression_metrics(y_true, y_pred)
    metrics["y_true"] = y_true
    metrics["y_pred"] = y_pred
    return metrics


def mean_baseline_metrics(train_dataset, test_dataset):
    y_train = np.array(
        [float(train_dataset[i][1]) for i in range(len(train_dataset))]
    )
    y_test = np.array(
        [float(test_dataset[i][1]) for i in range(len(test_dataset))]
    )

    train_mean = y_train.mean()
    pred = np.full_like(y_test, train_mean, dtype=np.float32)

    metrics = regression_metrics(y_test, pred)
    metrics["train_mean_chla"] = train_mean

    return metrics


def print_label_stats(dataset):
    labels = np.array([float(dataset[i][1]) for i in range(len(dataset))])

    print("\nLabel statistics:")
    print(f"min:  {labels.min():.4f}")
    print(f"max:  {labels.max():.4f}")
    print(f"mean: {labels.mean():.4f}")
    print(f"std:  {labels.std():.4f}")


def build_model(mode, in_channels):
    if mode == "three_branch_3_1_1":
        return ChlorophyllThreeBranchCNN().to(device)

    if mode == "three_branch_attention_3_1_1":
        return ChlorophyllThreeBranchAttentionCNN(
            embed_dim=128,
            num_heads=4,
        ).to(device)

    return ChlorophyllCNN(in_channels=in_channels).to(device)


def run_experiment(mode, in_channels):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    dataset = ChlorophyllDataset(
        DATASET_ROOT,
        mode=mode,
        image_size=IMAGE_SIZE,
    )

    print_label_stats(dataset)

    test_size = int(len(dataset) * TEST_RATIO)
    train_size = len(dataset) - test_size

    train_dataset, test_dataset = random_split(
        dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(SEED),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(mode, in_channels)

    loss_fn = nn.SmoothL1Loss(beta=1.0)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=1e-4,
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
    )

    best_mae = float("inf")
    best_state = None
    best_epoch = 0
    final_mae = None
    bad_epochs = 0

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            desc=f"{mode} | Epoch {epoch + 1:03d}/{EPOCHS}",
        )

        current_metrics = evaluate_metrics(model, test_loader, device)
        test_mae = current_metrics["mae"]
        final_mae = test_mae

        scheduler.step(test_mae)

        if test_mae < best_mae:
            best_mae = test_mae
            best_epoch = epoch + 1
            bad_epochs = 0
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
        else:
            bad_epochs += 1

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"{mode:30s} | "
            f"Epoch {epoch + 1:03d}/{EPOCHS} | "
            f"Train loss: {train_loss:.4f} | "
            f"Test MAE: {test_mae:.4f} | "
            f"RMSE: {current_metrics['rmse']:.4f} | "
            f"R2: {current_metrics['r2']:.3f} | "
            f"Accuracy: {current_metrics['mean_normalized_accuracy_%']:.2f}% | "
            f"LR: {current_lr:.2e}"
        )

        if bad_epochs >= PATIENCE:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate_metrics(model, test_loader, device)
    baseline_metrics = mean_baseline_metrics(train_dataset, test_dataset)

    print(
        f"\nBest epoch: {best_epoch} | "
        f"Best MAE: {best_mae:.4f} | "
        f"Final MAE before restore: {final_mae:.4f}"
    )

    if mode == "three_branch_3_1_1":
        channel_label = "3 + 1 + 1"
    elif mode == "three_branch_attention_3_1_1":
        channel_label = "3 + 1 + 1 + attention"
    else:
        channel_label = in_channels

    return {
        "experiment": mode,
        "in_channels": channel_label,
        "final_mae": final_mae,
        "best_epoch": best_epoch,
        "best_mae": best_mae,
        "test_mae": test_metrics["mae"],
        "test_mse": test_metrics["mse"],
        "test_rmse": test_metrics["rmse"],
        "test_mape_%": test_metrics["mape_%"],
        "test_mape_accuracy_%": test_metrics["mape_accuracy_%"],
        "test_mean_normalized_accuracy_%": test_metrics[
            "mean_normalized_accuracy_%"
        ],
        "test_r2": test_metrics["r2"],
        "mean_baseline_mae": baseline_metrics["mae"],
        "mean_baseline_mape_%": baseline_metrics["mape_%"],
        "mean_baseline_accuracy_%": baseline_metrics[
            "mean_normalized_accuracy_%"
        ],
        "mean_baseline_r2": baseline_metrics["r2"],
        "train_mean_chla": baseline_metrics["train_mean_chla"],
        "improvement_vs_mean_%": 100
        * (baseline_metrics["mae"] - test_metrics["mae"])
        / baseline_metrics["mae"],
        "y_true": test_metrics["y_true"],
        "y_pred": test_metrics["y_pred"],
        "model": model,
    }


# -------------------------
# Run experiments
# -------------------------

results = []

for mode, in_channels in MODES.items():
    print("\n" + "=" * 90)
    print(f"Running experiment: {mode}")
    print("=" * 90)

    result = run_experiment(mode, in_channels)
    results.append(result)


# -------------------------
# Results table
# -------------------------

results_df = pd.DataFrame(
    [
        {
            "experiment": r["experiment"],
            "in_channels": r["in_channels"],
            "best_epoch": r["best_epoch"],
            "test_mae": r["test_mae"],
            "test_rmse": r["test_rmse"],
            "test_mape_%": r["test_mape_%"],
            "test_mape_accuracy_%": r["test_mape_accuracy_%"],
            "test_mean_normalized_accuracy_%": r[
                "test_mean_normalized_accuracy_%"
            ],
            "test_r2": r["test_r2"],
            "mean_baseline_mae": r["mean_baseline_mae"],
            "mean_baseline_accuracy_%": r["mean_baseline_accuracy_%"],
            "mean_baseline_r2": r["mean_baseline_r2"],
            "train_mean_chla": r["train_mean_chla"],
            "improvement_vs_mean_%": r["improvement_vs_mean_%"],
        }
        for r in results
    ]
).sort_values("test_mae")

print("\nFinal comparison:")
print(results_df)


# -------------------------
# Plot best experiment
# -------------------------

best_result = min(results, key=lambda r: r["test_mae"])

y_true = best_result["y_true"]
y_pred = best_result["y_pred"]

plt.figure(figsize=(7, 7))
plt.scatter(y_true, y_pred, alpha=0.7)

min_v = min(y_true.min(), y_pred.min())
max_v = max(y_true.max(), y_pred.max())

plt.plot([min_v, max_v], [min_v, max_v], "r--")
plt.xlabel("True CHL-a")
plt.ylabel("Predicted CHL-a")
plt.title(
    f"Best model: {best_result['experiment']} | "
    f"MAE: {best_result['test_mae']:.3f} | "
    f"R2: {best_result['test_r2']:.3f}"
)
plt.grid(True)
plt.show()


# -------------------------
# Save results
# -------------------------

results_df.to_csv("chlorophyll_experiment_results.csv", index=False)

torch.save(
    best_result["model"].state_dict(),
    f"best_{best_result['experiment']}.pth",
)

print("\nSaved results to chlorophyll_experiment_results.csv")
print(f"Saved best model to best_{best_result['experiment']}.pth")
