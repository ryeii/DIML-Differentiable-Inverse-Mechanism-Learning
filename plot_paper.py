# - Creates exactly FOUR figures in a separate folder:
#     (1) E1: two-panel (diff_mse | cfkl_params)
#     (2) E2: two-panel (diff_mse | cfkl_params)
#     (3) E3: two-panel (diff_mse | cfkl_params)
#     (4) E7-combined: all E7_* experiments overlaid in one figure (two-panel: diff_mse | cfkl_params)
#
# Notes:
# - Uses PDF output for quality.
# - Font is tunable via FONT_SIZE (and optionally LABEL_SIZE, LEGEND_SIZE).
# - E7 experiment names in your runner look like: "E7_LargeScale_n{n}_A{A}".
# - For E7 combined plot, we pick the best available model curve per exp:
#     prefer "DIML_Large" > "DIML" > first model found.

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# User settings
# -----------------------------
CSV_PATH = "./diml_results/training_curves.csv"
OUT_DIR = "./diml_results/plots_paper"   # separate folder (per request)
SHOW = False
DPI = 160

# Tunable fonts
FONT_SIZE = 16
LABEL_SIZE = 16
LEGEND_SIZE = 16
TITLE_SIZE = 14  # not used by default (titles off), but available

plt.rcParams.update({
    "font.size": FONT_SIZE,
    "axes.labelsize": LABEL_SIZE,
    "legend.fontsize": LEGEND_SIZE,
    "axes.titlesize": TITLE_SIZE,
})

# Model ordering within each experiment plot
PREFERRED_TAG_ORDER = ["DIML", "StructMLE", "TabularMLE", "DIML_Wrong", "DIML_Large"]

# Only these metrics are needed now
LEFT_METRIC = "diff_mse"
RIGHT_METRIC = "cfkl_params"

# Optional: drop first K epochs
DROP_FIRST_EPOCHS = 0

# Which experiments to plot as single figures
E1_NAME = "E1_UnstructuredNeural_3A"
E2_NAME = "E2_Congestion_4A"
E3_NAME = "E3_PublicGoods_3A"

# Which experiments count as E7 large-scale (prefix match)
E7_PREFIX = "E7_LargeScale"


# -----------------------------
# Helpers
# -----------------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def sanitize_filename(s: str) -> str:
    s = re.sub(r"[^\w\-.]+", "_", s)
    return s[:180]

def parse_exp_and_model(tag: str):
    # tag format in curves: "{exp}/DIML" or "{exp}/DIML_Large" etc.
    if "/" in tag:
        exp = tag.split("/")[0]
        model = tag.split("/")[-1]
    else:
        exp = "UNKNOWN"
        model = tag
    return exp, model

def order_models(models):
    preferred = [m for m in PREFERRED_TAG_ORDER if m in models]
    others = sorted([m for m in models if m not in preferred])
    return preferred + others

def apply_drop_epochs(d: pd.DataFrame):
    if DROP_FIRST_EPOCHS > 0:
        return d[d["epoch"] > DROP_FIRST_EPOCHS].copy()
    return d

def maybe_get_series(d: pd.DataFrame, metric: str):
    if metric not in d.columns:
        return None, None
    x = d["epoch"].astype(int).to_numpy()
    y = pd.to_numeric(d[metric], errors="coerce").to_numpy()
    if np.all(np.isnan(y)):
        return None, None
    return x, y

def save_fig(fig, path, show=False):
    fig.tight_layout()
    fig.savefig(path, format="pdf", dpi=DPI)
    if show:
        plt.show()
    else:
        plt.close(fig)

def plot_two_panel_per_experiment(df_exp: pd.DataFrame, exp_name: str, out_dir: str, show=False):
    """
    For one experiment:
      left panel  = diff_mse over epochs (lines = models)
      right panel = cfkl_params over epochs (lines = models)
    """
    models = order_models(sorted(df_exp["model"].unique().tolist()))

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))  # two panels

    # ----- Left: diff_mse -----
    ax = axes[0]
    any_left = False
    for m in models:
        d = df_exp[df_exp["model"] == m].sort_values("epoch")
        d = apply_drop_epochs(d)
        x, y = maybe_get_series(d, LEFT_METRIC)
        if x is None:
            continue
        ax.plot(x, y, label=m)
        any_left = True
    ax.set_xlabel("Epoch")
    ax.set_ylabel(LEFT_METRIC)
    ax.grid(True, linewidth=0.5, alpha=0.4)
    if not any_left:
        ax.text(0.5, 0.5, f"No {LEFT_METRIC} data", ha="center", va="center", transform=ax.transAxes)

    # ----- Right: cfkl_params -----
    ax = axes[1]
    any_right = False
    for m in models:
        d = df_exp[df_exp["model"] == m].sort_values("epoch")
        d = apply_drop_epochs(d)
        x, y = maybe_get_series(d, RIGHT_METRIC)
        if x is None:
            continue
        ax.plot(x, y, label=m)
        any_right = True
    ax.set_xlabel("Epoch")
    ax.set_ylabel(RIGHT_METRIC)
    ax.grid(True, linewidth=0.5, alpha=0.4)
    if not any_right:
        ax.text(0.5, 0.5, f"No {RIGHT_METRIC} data", ha="center", va="center", transform=ax.transAxes)

    # Shared legend (use handles from right if present else left)
    handles, labels = axes[1].get_legend_handles_labels()
    if len(handles) == 0:
        handles, labels = axes[0].get_legend_handles_labels()
    if len(handles) > 0:
        # Place legend below the plots (centered)
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(4, len(labels)),
            frameon=True,
            bbox_to_anchor=(0.5, -0.08)  # push legend below x-axis labels
        )

        # Leave room at the bottom for the legend
        fig.tight_layout(rect=[0, 0.00, 1, 1])


    ensure_dir(out_dir)
    fn = sanitize_filename(f"{exp_name}__{LEFT_METRIC}__{RIGHT_METRIC}.pdf")
    path = os.path.join(out_dir, fn)
    fig.savefig(path, format="pdf", bbox_inches='tight')
    fn = sanitize_filename(f"{exp_name}__{LEFT_METRIC}__{RIGHT_METRIC}.png")
    path = os.path.join(out_dir, fn)
    fig.savefig(path, format="png", bbox_inches='tight')


def pick_best_model_for_e7(df_one_exp: pd.DataFrame) -> str:
    """
    Pick which model curve to use for E7 combined plot.
    Prefer DIML_Large > DIML > first available.
    """
    models = df_one_exp["model"].unique().tolist()
    if "DIML_Large" in models:
        return "DIML_Large"
    if "DIML" in models:
        return "DIML"
    return sorted(models)[0]

def short_e7_label(exp_name: str) -> str:
    # exp_name like "E7_LargeScale_n120_A25" -> "n120 A25"
    m = re.search(r"n(\d+)_A(\d+)", exp_name)
    if m:
        return f"n{m.group(1)} A{m.group(2)}"
    return exp_name

def plot_e7_combined(df: pd.DataFrame, out_dir: str, show=False):
    """
    One figure total for E7:
      two panels:
        left  = diff_mse over epochs, lines = each E7 experiment (best model per exp)
        right = cfkl_params over epochs, lines = each E7 experiment (best model per exp)
    """
    e7_exps = sorted([e for e in df["exp"].unique().tolist() if e.startswith(E7_PREFIX)])
    if len(e7_exps) == 0:
        print(f"No experiments found with prefix '{E7_PREFIX}'. Skipping E7 plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))

    # Left panel
    axL = axes[0]
    anyL = False

    # Right panel
    axR = axes[1]
    anyR = False

    for exp in e7_exps:
        d_exp = df[df["exp"] == exp].copy()
        best_model = pick_best_model_for_e7(d_exp)
        d_m = d_exp[d_exp["model"] == best_model].sort_values("epoch")
        d_m = apply_drop_epochs(d_m)

        label = short_e7_label(exp)

        x, y = maybe_get_series(d_m, LEFT_METRIC)
        if x is not None:
            axL.plot(x, y, label=label)
            anyL = True

        x, y = maybe_get_series(d_m, RIGHT_METRIC)
        if x is not None:
            axR.plot(x, y, label=label)
            anyR = True

    axL.set_xlabel("Epoch")
    axL.set_ylabel(LEFT_METRIC)
    axL.grid(True, linewidth=0.5, alpha=0.4)
    if not anyL:
        axL.text(0.5, 0.5, f"No {LEFT_METRIC} data", ha="center", va="center", transform=axL.transAxes)

    axR.set_xlabel("Epoch")
    axR.set_ylabel(RIGHT_METRIC)
    axR.grid(True, linewidth=0.5, alpha=0.4)
    if not anyR:
        axR.text(0.5, 0.5, f"No {RIGHT_METRIC} data", ha="center", va="center", transform=axR.transAxes)

    # Shared legend for scales
    handles, labels = axR.get_legend_handles_labels()
    if len(handles) == 0:
        handles, labels = axL.get_legend_handles_labels()
    if len(handles) > 0:
        # Place legend below the plots (centered)
        plt.rcParams.update({
            "legend.fontsize": 13.5,
        })
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(5, len(labels)),
            frameon=True,
            bbox_to_anchor=(0.5, -0.08)  # push legend below x-axis labels
        )

        # Leave room at the bottom for the legend
        fig.tight_layout(rect=[0, 0.00, 1, 1])

    ensure_dir(out_dir)
    fn = sanitize_filename(f"{E7_PREFIX}__combined__{LEFT_METRIC}__{RIGHT_METRIC}.pdf")
    path = os.path.join(out_dir, fn)
    fig.savefig(path, format="pdf", bbox_inches='tight')
    fn = sanitize_filename(f"{E7_PREFIX}__combined__{LEFT_METRIC}__{RIGHT_METRIC}.png")
    path = os.path.join(out_dir, fn)
    fig.savefig(path, format="png", bbox_inches='tight')


# -----------------------------
# Main
# -----------------------------
def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    if "tag" not in df.columns:
        raise ValueError("training_curves.csv must contain a 'tag' column.")

    # Derive exp/model columns
    exps, models = [], []
    for t in df["tag"].astype(str).tolist():
        e, m = parse_exp_and_model(t)
        exps.append(e)
        models.append(m)
    df["exp"] = exps
    df["model"] = models

    # Coerce epoch to int if present
    if "epoch" in df.columns:
        df["epoch"] = df["epoch"].astype(int)
    else:
        raise ValueError("training_curves.csv must contain an 'epoch' column.")

    # Keep only columns we need (+ harmless extras)
    cols = ["exp", "model", "epoch"]
    for c in [LEFT_METRIC, RIGHT_METRIC]:
        if c in df.columns:
            cols.append(c)
    df = df[cols].copy()

    ensure_dir(OUT_DIR)
    print(f"Loaded: {CSV_PATH}")
    print(f"Saving FOUR figures to: {OUT_DIR}")
    print("Font sizes:", {"FONT_SIZE": FONT_SIZE, "LABEL_SIZE": LABEL_SIZE, "LEGEND_SIZE": LEGEND_SIZE})

    # ---- E1, E2, E3 each as its own two-panel figure ----
    for exp_name in [E1_NAME, E2_NAME, E3_NAME]:
        d = df[df["exp"] == exp_name].copy()
        if len(d) == 0:
            print(f"Warning: experiment '{exp_name}' not found in CSV. Skipping.")
            continue
        plot_two_panel_per_experiment(d, exp_name, OUT_DIR, show=SHOW)

    # ---- E7 combined figure ----
    plot_e7_combined(df, OUT_DIR, show=SHOW)

    print("Done.")


if __name__ == "__main__":
    main()
