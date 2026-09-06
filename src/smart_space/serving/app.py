"""Streamlit demo for the occupancy models.

    streamlit run src/smart_space/serving/app.py
"""

from __future__ import annotations

import numpy as np
import streamlit as st
from PIL import Image

from smart_space.config import EVAL_DIR
from smart_space.inference.predictor import MODELS, OccupancyPredictor

st.set_page_config(page_title="Smart Space Occupancy", page_icon="👥", layout="wide")
st.title("👥 Smart Space Occupancy Monitoring")
st.caption("Estimate how many people are in a space — compare a detector, two regressors and a density model.")


@st.cache_resource
def get_predictor(name: str):
    return OccupancyPredictor.load(name)


tab_image, tab_results = st.tabs(["Single image", "Model comparison"])

with tab_image:
    model_name = st.selectbox("Model", MODELS, index=MODELS.index("xgboost"))
    upload = st.file_uploader("Upload a room image", type=["jpg", "jpeg", "png"])
    if upload:
        image = Image.open(upload).convert("RGB")
        col1, col2 = st.columns(2)
        col1.image(image, caption="input", use_column_width=True)
        try:
            result = get_predictor(model_name).predict(np.array(image)[..., ::-1])
        except FileNotFoundError:
            st.error(f"No trained checkpoint for '{model_name}'. Run `smartspace train {model_name}`.")
        else:
            col2.metric("Estimated occupancy", result.rounded(), help=f"raw = {result.count:.2f}")
            if result.density_map is not None:
                col2.image(
                    (result.density_map / (result.density_map.max() + 1e-8)),
                    caption="predicted density map",
                    use_column_width=True,
                    clamp=True,
                )
            if result.boxes:
                col2.write(f"{len(result.boxes)} person detections")

with tab_results:
    metrics_file = EVAL_DIR / "metrics.json"
    if metrics_file.exists():
        import json

        data = json.loads(metrics_file.read_text())
        rows = []
        for model, res in data.items():
            o = res["overall"]
            rows.append(
                {
                    "model": model,
                    "MAE": round(o["mae"]["value"], 3),
                    "MAE 95% CI": f"[{o['mae']['ci_low']:.2f}, {o['mae']['ci_high']:.2f}]",
                    "RMSE": round(o["rmse"]["value"], 3),
                    "MAPE %": round(o["mape"]["value"], 1),
                    "exact %": round(o["exact_match_pct"]["value"], 1),
                    "bias": round(res["mean_bias"], 2),
                }
            )
        st.dataframe(rows, use_container_width=True)
        for fig in ("pred_vs_true", "error_by_bucket", "calibration", "roc_occupied"):
            p = EVAL_DIR / "figures" / f"{fig}.png"
            if p.exists():
                st.image(str(p))
    else:
        st.info("Run `smartspace evaluate` to populate this tab.")
