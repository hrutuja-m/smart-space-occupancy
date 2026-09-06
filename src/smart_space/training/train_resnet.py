"""Train the ResNet18 count regressor."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.utils.data import DataLoader

from smart_space.config import (
    PROCESSED_DIR,
    RESNET_RUN_DIR,
    data_config,
    resolve_device,
    train_config,
)
from smart_space.data.dataset import OccupancyCountDataset
from smart_space.training.common import mlflow_run, seed_everything


@torch.no_grad()
def _evaluate(model, loader, device) -> tuple[float, float, float]:
    model.eval()
    abs_e, sq_e, exact, n = 0.0, 0.0, 0, 0
    for x, y in loader:
        p = torch.round(model(x.to(device)).squeeze(1)).cpu()
        abs_e += torch.abs(p - y).sum().item()
        sq_e += torch.square(p - y).sum().item()
        exact += int((p == y).sum())
        n += y.numel()
    return abs_e / n, math.sqrt(sq_e / n), 100.0 * exact / n


def train() -> str:
    dcfg, tcfg = data_config(), train_config("resnet.yaml")
    device = resolve_device()
    seed_everything(dcfg.seed)

    train_ds = OccupancyCountDataset(PROCESSED_DIR / "train", dcfg.img_size, train=True)
    val_ds = OccupancyCountDataset(PROCESSED_DIR / "val", dcfg.img_size)
    train_loader = DataLoader(train_ds, batch_size=tcfg.batch_size, shuffle=True, num_workers=tcfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=tcfg.batch_size, num_workers=tcfg.num_workers)

    from smart_space.models.resnet_regressor import build_regressor

    model = build_regressor(pretrained=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tcfg.epochs)
    loss_fn = nn.SmoothL1Loss()

    RESNET_RUN_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = RESNET_RUN_DIR / "resnet18_regressor.pt"
    best_mae = float("inf")

    with mlflow_run(tcfg.mlflow_experiment, "resnet18_regressor", vars(tcfg) | vars(dcfg)) as run:
        for epoch in range(tcfg.epochs):
            model.train()
            running = 0.0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                loss = loss_fn(model(x).squeeze(1), y)
                loss.backward()
                opt.step()
                running += loss.item() * y.size(0)
            sched.step()

            mae, rmse, exact = _evaluate(model, val_loader, device)
            run.log_metrics(
                {"train_loss": running / len(train_ds), "val_mae": mae, "val_rmse": rmse, "val_exact": exact},
                step=epoch,
            )
            print(
                f"epoch {epoch+1:02d}/{tcfg.epochs}  loss {running/len(train_ds):.3f}  "
                f"val_MAE {mae:.3f}  exact {exact:.1f}%"
            )
            if mae < best_mae:
                best_mae = mae
                torch.save(model.state_dict(), ckpt)

        run.log_metric("best_val_mae", best_mae)
        run.log_artifact(str(ckpt))

    print(f"Best val MAE {best_mae:.3f}  ->  {ckpt}")
    return str(ckpt)


if __name__ == "__main__":
    train()
