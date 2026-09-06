"""Export the ResNet regressor / CSRNet-lite to ONNX and benchmark CPU latency.

The edge story: an occupancy sensor runs on a Raspberry Pi / Jetson class device,
on-device, discarding frames. This script produces the ONNX artifact and a
latency/throughput table (p50/p95 over N runs) that quantifies whether that is
realistic.
"""

from __future__ import annotations

import json
import time

import numpy as np
import torch

from smart_space.config import DENSITY_RUN_DIR, EXPORT_DIR, RESNET_RUN_DIR, data_config


def _load(model_name: str):
    if model_name == "resnet":
        from smart_space.models.resnet_regressor import build_regressor

        net = build_regressor(pretrained=False)
        net.load_state_dict(torch.load(RESNET_RUN_DIR / "resnet18_regressor.pt", map_location="cpu"))
        return net.eval()
    if model_name == "density":
        from smart_space.models.density_net import CSRNetLite

        net = CSRNetLite(pretrained=False)
        net.load_state_dict(torch.load(DENSITY_RUN_DIR / "csrnet_lite.pt", map_location="cpu"))
        return net.eval()
    raise ValueError("model_name must be 'resnet' or 'density'")


def export(model_name: str = "resnet", n_bench: int = 100) -> dict:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    img_size = data_config().img_size
    net = _load(model_name)
    dummy = torch.randn(1, 3, img_size, img_size)

    onnx_path = EXPORT_DIR / f"{model_name}.onnx"
    export_kwargs = dict(
        input_names=["image"],
        output_names=["output"],
        dynamic_axes={"image": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
    )
    try:
        # Legacy TorchScript exporter — no onnxscript dependency.
        torch.onnx.export(net, dummy, str(onnx_path), dynamo=False, **export_kwargs)
    except TypeError:
        # Older torch without the `dynamo` kwarg.
        torch.onnx.export(net, dummy, str(onnx_path), **export_kwargs)

    result: dict = {"model": model_name, "onnx_path": str(onnx_path), "img_size": img_size}

    # torch CPU latency
    with torch.no_grad():
        for _ in range(10):
            net(dummy)
        t = []
        for _ in range(n_bench):
            s = time.perf_counter()
            net(dummy)
            t.append((time.perf_counter() - s) * 1000)
    result["torch_cpu_ms"] = {"p50": float(np.percentile(t, 50)), "p95": float(np.percentile(t, 95))}

    # onnxruntime latency
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        feed = {"image": dummy.numpy()}
        for _ in range(10):
            sess.run(None, feed)
        t = []
        for _ in range(n_bench):
            s = time.perf_counter()
            sess.run(None, feed)
            t.append((time.perf_counter() - s) * 1000)
        result["onnxruntime_cpu_ms"] = {"p50": float(np.percentile(t, 50)), "p95": float(np.percentile(t, 95))}
        result["throughput_fps"] = round(1000.0 / np.percentile(t, 50), 1)
    except ImportError:
        result["onnxruntime_cpu_ms"] = "install smart-space-occupancy[edge]"

    (EXPORT_DIR / f"{model_name}_benchmark.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    export()
