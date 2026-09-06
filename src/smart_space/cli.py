"""``smartspace`` command-line entrypoint.

Examples
--------
    smartspace data ingest
    smartspace data split
    smartspace train yolo
    smartspace train all
    smartspace evaluate
    smartspace predict path/to/image.jpg --model xgboost
    smartspace video office.mp4 --mode flow --line 0,240,640,240
    smartspace export --model resnet
"""

from __future__ import annotations

import typer
from rich import print as rprint

app = typer.Typer(add_completion=False, help="Smart Space Occupancy Monitoring toolkit")
data_app = typer.Typer(help="Dataset preparation")
train_app = typer.Typer(help="Model training")
app.add_typer(data_app, name="data")
app.add_typer(train_app, name="train")


# ----------------------------- data ------------------------------------
@data_app.command("ingest")
def data_ingest(max_samples: int = typer.Option(None, help="cap; 0 = all")):
    from smart_space.data.ingest import ingest

    ingest(max_samples=max_samples)


@data_app.command("split")
def data_split():
    from smart_space.data.split import split

    split()


@data_app.command("prepare")
def data_prepare():
    """ingest + split in one go."""
    from smart_space.data.ingest import ingest
    from smart_space.data.split import split

    ingest()
    split()


# ----------------------------- train ----------------------------------
@train_app.command("yolo")
def train_yolo():
    from smart_space.training.train_yolo import train

    train()


@train_app.command("resnet")
def train_resnet():
    from smart_space.training.train_resnet import train

    train()


@train_app.command("xgboost")
def train_xgboost():
    from smart_space.training.train_xgboost import train

    train()


@train_app.command("density")
def train_density():
    from smart_space.training.train_density import train

    train()


@train_app.command("all")
def train_all():
    for name in ("yolo", "resnet", "xgboost", "density"):
        rprint(f"[bold cyan]── training {name} ──[/]")
        __import__(f"smart_space.training.train_{name}", fromlist=["train"]).train()


# --------------------------- evaluate --------------------------------
@app.command("evaluate")
def evaluate(plots: bool = typer.Option(True, help="also render diagnostic figures")):
    from smart_space.evaluation.evaluate import run

    run()
    if plots:
        from smart_space.evaluation.plots import generate_all

        generate_all()
    from smart_space.evaluation.report import main as render_report

    render_report()


# --------------------------- predict ---------------------------------
@app.command("predict")
def predict(image: str, model: str = typer.Option("xgboost")):
    from smart_space.inference.predictor import OccupancyPredictor

    result = OccupancyPredictor.load(model).predict(image)
    rprint(f"[bold green]{result.model}[/] -> occupancy = [bold]{result.rounded()}[/] (raw {result.count:.2f})")


@app.command("video")
def video(
    path: str,
    mode: str = typer.Option("instant", help="instant | flow"),
    line: str = typer.Option(None, help="x1,y1,x2,y2 doorway line for flow mode"),
    sample_every: int = typer.Option(15),
    out_csv: str = typer.Option(None),
):
    from smart_space.inference.video import VideoConfig, analyze_video

    line_t = None
    if line:
        x1, y1, x2, y2 = (float(v) for v in line.split(","))
        line_t = ((x1, y1), (x2, y2))
    res = analyze_video(path, VideoConfig(sample_every=sample_every, mode=mode, line=line_t))
    rprint(res.meta, f"peak={res.peak} mean={res.mean:.1f} entries={res.entries} exits={res.exits}")
    if out_csv:
        res.series.to_csv(out_csv, index=False)
        rprint(f"series -> {out_csv}")


@app.command("export")
def export(model: str = typer.Option("resnet", help="resnet | density")):
    from smart_space.inference.export_onnx import export as _export

    _export(model)


if __name__ == "__main__":
    app()
