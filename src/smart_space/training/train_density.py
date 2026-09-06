"""Train CSRNet-lite density estimator."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.utils.data import DataLoader

from smart_space.config import (
    DENSITY_RUN_DIR,
    PROCESSED_DIR,
    data_config,
    resolve_device,
    train_config,
)
from smart_space.data.dataset import OccupancyDensityDataset
from smart_space.models.density_net import CSRNetLite
from smart_space.training.common import mlflow_run, seed_everything


@torch.no_grad()
def _evaluate(model, loader, device) -> tuple[float, float]:
    model.eval()
    abs_e, sq_e, n = 0.0, 0.0, 0
    for x, dmap in loader:
        pred_count = model.predict_count(x.to(device)).cpu()
        true_count = dmap.sum(dim=(1, 2, 3))
        abs_e += torch.abs(pred_count - true_count).sum().item()
        sq_e += torch.square(pred_count - true_count).sum().item()
        n += true_count.numel()
    return abs_e / n, math.sqrt(sq_e / n)


def train() -> str:
    dcfg, tcfg = data_config(), train_config("density.yaml")
    device = resolve_device()
    seed_everything(dcfg.seed)

    train_ds = OccupancyDensityDataset(PROCESSED_DIR / "train", dcfg.img_size, dcfg.density_sigma, train=True)
    val_ds = OccupancyDensityDataset(PROCESSED_DIR / "val", dcfg.img_size, dcfg.density_sigma)
    train_loader = DataLoader(train_ds, batch_size=tcfg.batch_size, shuffle=True, num_workers=tcfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=tcfg.batch_size, num_workers=tcfg.num_workers)

    model = CSRNetLite(pretrained=True).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    # Sum-MSE on the density map (CSRNet convention): scale up so gradients aren't tiny.
    loss_fn = nn.MSELoss(reduction="sum")

    DENSITY_RUN_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = DENSITY_RUN_DIR / "csrnet_lite.pt"
    best_mae = float("inf")

    with mlflow_run(tcfg.mlflow_experiment, "csrnet_lite", vars(tcfg) | vars(dcfg)) as run:
        for epoch in range(tcfg.epochs):
            model.train()
            running = 0.0
            for x, dmap in train_loader:
                x, dmap = x.to(device), dmap.to(device)
                opt.zero_grad()
                loss = loss_fn(model(x), dmap) / x.size(0)
                loss.backward()
                opt.step()
                running += loss.item() * x.size(0)

            mae, rmse = _evaluate(model, val_loader, device)
            run.log_metrics({"train_loss": running / len(train_ds), "val_mae": mae, "val_rmse": rmse}, step=epoch)
            print(f"epoch {epoch+1:02d}/{tcfg.epochs}  loss {running/len(train_ds):.2f}  val_MAE {mae:.3f}")
            if mae < best_mae:
                best_mae = mae
                torch.save(model.state_dict(), ckpt)

        run.log_metric("best_val_mae", best_mae)
        run.log_artifact(str(ckpt))

    print(f"Best val MAE {best_mae:.3f}  ->  {ckpt}")
    return str(ckpt)


if __name__ == "__main__":
    train()
