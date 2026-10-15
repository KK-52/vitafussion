"""
VitaFusion AI — SHAP Explainability for Clinical Features
===========================================================
SHAP (SHapley Additive exPlanations) waterfall charts,
summary plots, and force plots for clinical risk factors.

Reference: Lundberg & Lee (2017) "A Unified Approach to Interpreting Model Predictions"
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List, Optional, Tuple
import io


# ── SHAP Value Computation (Simulation) ──────────────────────────────────────

def compute_shap_values_simulated(
    feature_importances: Dict[str, float],
    base_value: float = 0.3,
    output_value: float = 0.65,
) -> Dict:
    """
    Convert clinical feature importances to SHAP-compatible values.

    In simulation mode, we use the clinical encoder's feature_importances
    (which are already effect-direction-aware) and scale them to sum to
    (output_value - base_value).

    Args:
        feature_importances: {feature_name: signed_importance} from clinical encoder
        base_value:   Model baseline risk (population average ~30%)
        output_value: Final predicted risk for this patient

    Returns:
        dict with shap_values, feature_names, base_value, output_value
    """
    names   = list(feature_importances.keys())
    effects = np.array([feature_importances[n] for n in names], dtype=np.float32)

    # Scale effects so they sum to the delta from baseline
    delta = output_value - base_value
    total = abs(effects).sum()
    if total > 0:
        effects = effects * (delta / total)

    return {
        "shap_values":    effects,
        "feature_names":  names,
        "base_value":     base_value,
        "output_value":   output_value,
        "data":           None,  # actual feature values (added separately)
    }


# ── Waterfall Plot ────────────────────────────────────────────────────────────

def plot_shap_waterfall(
    shap_data: Dict,
    feature_values: Optional[Dict] = None,
    max_display: int = 15,
    title: str = "SHAP Waterfall — Clinical Risk Factors",
    figsize: Tuple[int, int] = (10, 8),
) -> bytes:
    """
    Create a SHAP waterfall plot as PNG bytes.

    Shows how each feature pushes the risk prediction up or down
    from the baseline (population average risk).

    Args:
        shap_data:      Output of compute_shap_values_simulated
        feature_values: Dict of actual patient values {feature: value}
        max_display:    Maximum features to show
        title:          Chart title
        figsize:        Figure dimensions

    Returns:
        PNG image as bytes
    """
    shap_vals   = shap_data["shap_values"]
    feat_names  = shap_data["feature_names"]
    base_val    = shap_data["base_value"]
    output_val  = shap_data["output_value"]

    # Sort by absolute value, keep top N
    order     = np.argsort(np.abs(shap_vals))[::-1][:max_display]
    shap_top  = shap_vals[order]
    names_top = [feat_names[i] for i in order]

    # Append feature values to labels if available
    if feature_values:
        labels = []
        for name in names_top:
            val = feature_values.get(name)
            if val is not None:
                if isinstance(val, bool):
                    val_str = "Yes" if val else "No"
                elif isinstance(val, float):
                    val_str = f"{val:.1f}"
                else:
                    val_str = str(val)
                labels.append(f"{name} = {val_str}")
            else:
                labels.append(name)
    else:
        labels = names_top

    # Reverse for bottom-to-top display
    shap_top = shap_top[::-1]
    labels   = labels[::-1]

    # ── Draw waterfall ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize, facecolor="#111827")
    ax.set_facecolor("#111827")

    colors = ["#EF4444" if v > 0 else "#22C55E" for v in shap_top]

    y_positions = np.arange(len(shap_top))
    bars = ax.barh(y_positions, shap_top, color=colors, height=0.65,
                   edgecolor="#374151", linewidth=0.5, alpha=0.9)

    # Cumulative line (waterfall effect)
    cumulative = np.cumsum(shap_top[::-1])[::-1]
    running    = base_val + np.cumsum(shap_top[::-1])[::-1]

    # Value annotations
    for i, (val, bar) in enumerate(zip(shap_top, bars)):
        x_pos = val + (0.005 if val > 0 else -0.005)
        ha    = "left" if val > 0 else "right"
        color = "#EF4444" if val > 0 else "#22C55E"
        sign  = "+" if val > 0 else ""
        ax.text(x_pos, i, f"{sign}{val:.3f}", va="center", ha=ha,
                fontsize=9, color=color, fontweight="bold")

    # Baseline and output lines
    ax.axvline(0, color="#9CA3AF", linewidth=1, linestyle="--", alpha=0.7)

    # Style
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=10, color="#E2E8F0")
    ax.set_xlabel("SHAP Value (impact on risk prediction)", color="#9CA3AF", fontsize=11)
    ax.set_title(title, color="#E2E8F0", fontsize=13, fontweight="bold", pad=15)

    # Tick colors
    ax.tick_params(axis='x', colors="#9CA3AF")
    ax.tick_params(axis='y', colors="#E2E8F0")
    for spine in ax.spines.values():
        spine.set_edgecolor("#374151")

    # Legend
    pos_patch = mpatches.Patch(color="#EF4444", label="↑ Increases risk")
    neg_patch = mpatches.Patch(color="#22C55E", label="↓ Decreases risk")
    ax.legend(handles=[pos_patch, neg_patch], loc="lower right",
              facecolor="#1F2937", edgecolor="#374151", labelcolor="#E2E8F0", fontsize=10)

    # Baseline annotation
    ax.text(0.98, -0.02,
            f"Baseline: {base_val*100:.0f}%  →  Prediction: {output_val*100:.0f}%",
            transform=ax.transAxes, ha="right", va="top",
            color="#9CA3AF", fontsize=10, style="italic")

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor="#111827", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def plot_shap_summary(
    all_shap_values: np.ndarray,
    feature_names: List[str],
    disease: str = "Cardiovascular",
    figsize: Tuple[int, int] = (10, 7),
) -> bytes:
    """
    Dot-summary plot showing feature importance distribution.
    Each dot is one patient, colored by feature value.

    Returns PNG bytes.
    """
    n_features = min(15, len(feature_names))
    mean_abs   = np.abs(all_shap_values).mean(axis=0)
    order      = np.argsort(mean_abs)[::-1][:n_features]

    fig, ax = plt.subplots(figsize=figsize, facecolor="#111827")
    ax.set_facecolor("#111827")

    colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, n_features))

    for i, feat_idx in enumerate(order[::-1]):
        y = i
        vals = all_shap_values[:, feat_idx]
        # Jitter for visibility
        rng = np.random.default_rng(feat_idx)
        jitter = rng.uniform(-0.25, 0.25, len(vals))
        ax.scatter(vals, y + jitter, alpha=0.6, s=15,
                   c=[colors[i]] * len(vals), edgecolors="none")

    ax.set_yticks(range(n_features))
    ax.set_yticklabels([feature_names[i] for i in order[::-1]],
                       fontsize=10, color="#E2E8F0")
    ax.set_xlabel("SHAP Value", color="#9CA3AF", fontsize=11)
    ax.set_title(f"Feature Importance Summary — {disease} Risk",
                 color="#E2E8F0", fontsize=13, fontweight="bold", pad=12)
    ax.axvline(0, color="#9CA3AF", linewidth=1, linestyle="--")
    ax.tick_params(axis='x', colors="#9CA3AF")
    for spine in ax.spines.values():
        spine.set_edgecolor("#374151")

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor="#111827", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def plot_feature_importance_bar(
    feature_importances: Dict[str, float],
    title: str = "Feature Importance",
    max_display: int = 12,
    figsize: Tuple[int, int] = (9, 6),
) -> bytes:
    """
    Horizontal bar chart of feature importances with positive/negative coloring.
    Returns PNG bytes.
    """
    items = sorted(feature_importances.items(), key=lambda x: abs(x[1]), reverse=True)[:max_display]
    names = [k for k, v in items]
    vals  = np.array([v for k, v in items])

    # Sort ascending for display
    sort_idx = np.argsort(vals)
    names = [names[i] for i in sort_idx]
    vals  = vals[sort_idx]

    colors = ["#EF4444" if v > 0 else "#22C55E" for v in vals]

    fig, ax = plt.subplots(figsize=figsize, facecolor="#111827")
    ax.set_facecolor("#111827")

    ax.barh(range(len(vals)), vals, color=colors, edgecolor="#374151",
            linewidth=0.5, height=0.7, alpha=0.9)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10, color="#E2E8F0")
    ax.set_xlabel("Feature Contribution", color="#9CA3AF", fontsize=11)
    ax.set_title(title, color="#E2E8F0", fontsize=13, fontweight="bold", pad=12)
    ax.axvline(0, color="#9CA3AF", linewidth=1, linestyle="--")
    ax.tick_params(axis='x', colors="#9CA3AF")
    for spine in ax.spines.values():
        spine.set_edgecolor("#374151")

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor="#111827", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf.read()
