"""Streamlit operator dashboard.

Layout
------
Sidebar : API health check, engine selector, controls.
Main    : sensor trajectories | anomaly score timeline | LLM alert card.

Data flow
---------
- Sensor data and anomaly scores are computed locally (no API round-trip per
  window — would be too slow for 100+ windows).
- The LLM alert is fetched from the FastAPI /predict endpoint for the most
  anomalous window only.
"""

from __future__ import annotations

import os

import httpx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
RAG_ENABLED = os.getenv("RAG_ENABLED", "true").lower() != "false"

SENSORS_TO_PLOT = ["sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_11", "sensor_15"]

st.set_page_config(
    page_title="Predictive Maintenance Copilot",
    page_icon="⚙",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading model and data...")
def _load_resources():
    from src.config import settings
    from src.data.loaders import load_cmapss
    from src.data.preprocessing import fit_scaler
    from src.models.autoencoder import LSTMAutoencoder

    ds = load_cmapss("FD001")
    scaler = fit_scaler(ds.train, ds.feature_columns)

    ckpt = torch.load(str(settings.checkpoint_path), map_location="cpu", weights_only=True)
    model = LSTMAutoencoder(
        n_features=len(ckpt["feature_columns"]),
        window_size=ckpt["window_size"],
        hidden_dim=ckpt["hidden_dim"],
        latent_dim=ckpt["latent_dim"],
        num_layers=ckpt.get("num_layers", 1),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    threshold = float(ckpt.get("anomaly_threshold_f1") or ckpt.get("anomaly_threshold") or 0.33)
    return ds, scaler, model, ckpt["feature_columns"], ckpt["window_size"], threshold


@st.cache_data(show_spinner="Computing anomaly scores...")
def _compute_scores(unit_id: int) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    from src.data.preprocessing import apply_scaler
    from src.models.autoencoder import reconstruction_error

    ds, scaler, model, feature_columns, window_size, _ = _load_resources()

    unit_df = ds.test[ds.test["unit"] == unit_id].copy()
    unit_scaled = apply_scaler(unit_df, feature_columns, scaler)

    scaled_values = unit_scaled[feature_columns].values.astype(np.float32)
    x = np.stack(
        [scaled_values[i : i + window_size] for i in range(len(scaled_values) - window_size + 1)]
    )

    if x.shape[0] == 0:
        return unit_df, np.array([]), np.array([]), scaled_values

    with torch.no_grad():
        errors = reconstruction_error(model, torch.from_numpy(x)).numpy()

    cycles = unit_df["cycle"].values[window_size - 1 :]
    return unit_df, cycles, errors, scaled_values


def _fetch_alert(window: np.ndarray, feature_names: list[str]) -> dict | None:
    try:
        resp = httpx.post(
            f"{API_BASE_URL}/predict",
            json={"sensor_window": window.tolist(), "feature_names": feature_names},
            timeout=300.0,
        )
        resp.raise_for_status()
        return dict(resp.json())
    except Exception as e:
        st.warning(f"API call failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("Predictive Maintenance Copilot")
    st.caption("NASA C-MAPSS turbofan — FD001")
    st.divider()

    st.subheader("API status")
    try:
        r = httpx.get(f"{API_BASE_URL}/healthz", timeout=2.0)
        r.raise_for_status()
        info = r.json()
        st.success("Online")
        st.caption(f"LLM: `{info['llm_provider']}`")
        st.caption(f"Threshold: `{info['anomaly_threshold']:.4f}`")
    except Exception as e:
        st.error(f"Offline — {e}")
        st.caption("Start the API with `make serve`.")

    st.divider()
    st.subheader("Engine")
    try:
        ds, _, _, feature_columns, window_size, default_threshold = _load_resources()
        n_units = ds.test["unit"].nunique()
        unit_id = st.selectbox("Select test engine", list(range(1, n_units + 1)), index=0)
        rul_at_end = int(ds.rul_test.loc[unit_id])
        st.metric("True RUL at last cycle", f"{rul_at_end} cycles")
    except Exception as e:
        st.error(f"Could not load data: {e}")
        st.stop()

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

st.header(f"Engine unit {unit_id}  —  RUL = {rul_at_end} cycles")

unit_df, cycles, errors, scaled_values = _compute_scores(unit_id)

if errors.shape[0] == 0:
    st.warning("Not enough cycles for a sliding window on this unit.")
    st.stop()

# ---------------------------------------------------------------------------
# Row 1: sensor trajectories
# ---------------------------------------------------------------------------

st.subheader("Sensor trajectories")

available_sensors = [s for s in SENSORS_TO_PLOT if s in unit_df.columns]
fig, axes = plt.subplots(
    len(available_sensors),
    1,
    figsize=(12, 1.8 * len(available_sensors)),
    sharex=True,
)
if len(available_sensors) == 1:
    axes = [axes]

for ax, sensor in zip(axes, available_sensors, strict=False):
    ax.plot(unit_df["cycle"], unit_df[sensor], color="steelblue", linewidth=0.9)
    ax.set_ylabel(sensor, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel("Cycle")
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

# ---------------------------------------------------------------------------
# Row 2: anomaly score timeline
# ---------------------------------------------------------------------------

st.subheader("Anomaly score per window")

fig2, ax2 = plt.subplots(figsize=(12, 3))
ax2.plot(cycles, errors, color="steelblue", linewidth=0.9, label="Reconstruction error")
ax2.axhline(
    default_threshold,
    color="crimson",
    linestyle="--",
    linewidth=1.2,
    label=f"Threshold = {default_threshold:.4f}",
)

anomaly_mask = errors > default_threshold
if anomaly_mask.any():
    ax2.fill_between(
        cycles, 0, errors, where=anomaly_mask, alpha=0.25, color="crimson", label="Anomaly region"
    )

ax2.set_xlabel("Cycle")
ax2.set_ylabel("MSE")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)
plt.tight_layout()
st.pyplot(fig2)
plt.close(fig2)

n_anomalous = int(anomaly_mask.sum())
col_a, col_b, col_c = st.columns(3)
col_a.metric("Max score", f"{errors.max():.4f}")
col_b.metric("Anomalous windows", f"{n_anomalous} / {len(errors)}")
col_c.metric("Peak cycle", int(cycles[errors.argmax()]))

# ---------------------------------------------------------------------------
# Row 3: LLM alert card
# ---------------------------------------------------------------------------

st.subheader("Operator alert")

if not anomaly_mask.any():
    st.success("No anomaly detected for this engine unit.")
else:
    worst_idx = int(errors.argmax())
    _, _, _, _, window_size, _ = _load_resources()
    worst_window = scaled_values[worst_idx : worst_idx + window_size]

    if not RAG_ENABLED:
        st.info(
            "LLM explanation with RAG is disabled in this deployment. "
            "Clone the repo and run locally for the full pipeline — see README for setup."
        )
    elif st.button("Generate LLM alert for worst window", type="primary"):
        with st.spinner("Calling LLM via RAG pipeline..."):
            result = _fetch_alert(worst_window, feature_columns)

        if result and result.get("alert"):
            alert = result["alert"]
            severity = alert.get("severity", "info")
            color = {"critical": "red", "warning": "orange", "info": "blue"}.get(severity, "blue")

            st.markdown(
                f"**Severity:** :{color}[{severity.upper()}]  "
                f"&nbsp;&nbsp; **Score:** `{result['anomaly_score']:.4f}`  "
                f"&nbsp;&nbsp; **Threshold:** `{result['threshold']:.4f}`"
            )
            st.divider()

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Probable cause**")
                st.write(alert.get("probable_cause", "-"))

                st.markdown("**Affected sensors**")
                sensors = alert.get("affected_sensors", [])
                st.write(", ".join(sensors) if sensors else "-")

            with col2:
                st.markdown("**Recommended action**")
                st.write(alert.get("recommended_action", "-"))

            citations = alert.get("citations", [])
            if citations:
                st.divider()
                st.markdown("**Citations**")
                for c in citations:
                    source = c.get("source", "")
                    page = c.get("page")
                    quote = c.get("quote", "")
                    label = f"{source}" + (f", p.{page}" if page else "")
                    if quote:
                        st.caption(f"_{label}_: {quote}")
                    else:
                        st.caption(f"_{label}_")
        else:
            st.info("No alert returned (healthy or API unavailable).")
