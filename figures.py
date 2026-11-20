"""Draw every figure of the report. NOT part of the pipeline.

It only reads the tables that pipeline.py saved in outputs/ (and, for the
race figure, data/test_predictions.csv, which the test command writes).
It fits no model and writes nothing to outputs/: each figure is saved in
figures/ as a PDF (for the report) and as a PNG.

Run:  python figures.py          (a few seconds, after the pipeline)

Figures: 1 pipeline, 2 ridge and lasso alpha curves, 3 inputs the lasso
keeps, 4 forest depth, 5 boosting grid, 6 the 6 models, 7 the 64 candidates,
8 learning curves, 9 a typical test race, A1 the shrunk dispersion index.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = "outputs"                              # the tables of the pipeline
PREDICTIONS = "data/test_predictions.csv"    # per athlete, written by the test command
FIG_DIR = "figures"                          # where the figures go

STORY_RACE = "BT2425SWRLCP07SMSP"            # Nove Mesto, 6 March 2025, men (Figure 9)
AUTHOR = "DU PASQUIER"                       # highlighted in Figure 9
AUTHOR_LABEL = "Arnaud Du Pasquier"

# 1 style for all figures. The font is the one of the report (Latin Modern);
# if it is not installed, the next serif font in the list is used.
BLUE, ORANGE, GREEN, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
INK, SECOND, GRID, GREY = "#1a1a1a", "#4a4a48", "#e3e2de", "#b9b8b2"
installed = {f.name for f in font_manager.fontManager.ttflist}
SERIF = [name for name in ["Latin Modern Roman", "LM Roman 10", "CMU Serif"] if name in installed]
plt.rcParams.update({
    "font.family": SERIF + ["DejaVu Serif"], "mathtext.fontset": "cm", "font.size": 10.5,
    "axes.edgecolor": "#8f8e89", "axes.linewidth": 0.7, "axes.labelcolor": INK,
    "xtick.color": SECOND, "ytick.color": SECOND, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "savefig.bbox": "tight", "legend.frameon": False,
    "axes.unicode_minus": False})
LINE = dict(lw=1.6, ms=4.5, mec="white", mew=0.8)
BLUES = ["#f3f3f1", "#cfe0f5", "#8fb8ea", "#5a94dc", "#2f6fc4", "#184f95"]   # light to dark

FAMILIES = [("ridge", "Ridge", BLUE, "o"), ("lasso", "Lasso", ORANGE, "s"),
            ("forest", "Random forest", GREEN, "^"), ("boosting", "Gradient boosting", VIOLET, "D")]

# The 13 inputs: number in the report and short name.
INPUT_NAMES = {
    "last3_finish_pct": "1  Finish percentile, last 3 sprints",
    "last3_ski_pct": "2  Ski percentile, last 3 sprints",
    "last3_range_pct": "3  Range percentile, last 3 sprints",
    "last3_miss_pct": "4  Miss percentile, last 3 sprints",
    "last_finish_pct": "5  Finish percentile, last sprint",
    "n_sprints_window": "6  Sprints in the window",
    "last_place_any": "7  Place in the last race, any format",
    "window_mean_misses": "8  Mean misses in the window",
    "window_miss_dispersion": "9  Dispersion of the misses",
    "race_days_21d": "10  Race days in 21 days (load)",
    "missed_prev_event": "11  Missed the previous event (load)",
    "first_start_major": "12  First start at a championship (load)",
    "no_history": "Flag  No history",
}


def table(name):
    return pd.read_csv(os.path.join(OUT, name + ".csv"))


def save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, name + ".pdf"))
    fig.savefig(os.path.join(FIG_DIR, name + ".png"), dpi=220)
    plt.close(fig)
    print("saved", name)


# ---------------------------------------------------------------- Figure 1
def fig_pipeline():
    """The pipeline of every model, as boxes and arrows."""
    W, H = 128, 52
    fig, ax = plt.subplots(figsize=(9.6, 3.6))
    ax.set_xlim(0, W); ax.set_ylim(3, H); ax.axis("off")

    def box(x, y, w, h, text, ec=INK, ls="-", tc=INK, lw=1.1, fs=8.5, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                                    ec=ec, fc="white", ls=ls, lw=lw))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=tc, fontsize=fs,
                fontweight="bold" if bold else "normal", linespacing=1.25)

    def arrow(x1, y1, x2, y2, c=INK, ls="-"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9,
                                     color=c, lw=1.0, ls=ls, shrinkA=0, shrinkB=0))

    dashed = (0, (3, 3))
    box(1, 12, 15, 26, "Features of\nan athlete\nin a race\n\n10 inputs\n(13 in the\nload test)")
    ax.add_patch(FancyBboxPatch((18.5, 6.5), 53, 42.5, boxstyle="round,pad=0,rounding_size=1.5",
                                ec="#d0d0d0", fc="#f7f7f7", lw=0.8))
    ax.text(45, 46.2, "Preprocessing and model, fitted on training rows only",
            ha="center", va="center", fontsize=8.5, color=SECOND)
    lanes = [33, 21, 9]; bh = 9; A, B, C = 20, 37, 54; bw = 14
    y = lanes[0]
    box(A, y, bw, bh, "Median fill", ec=BLUE); box(B, y, bw, bh, "Scaling", ec=BLUE)
    box(C, y, bw, bh, "Ridge or\nlasso", ec=BLUE, bold=True)
    arrow(A + bw, y + bh / 2, B, y + bh / 2, BLUE); arrow(B + bw, y + bh / 2, C, y + bh / 2, BLUE)
    y = lanes[1]
    box(A, y, bw, bh, "Median fill", ec=GREEN)
    box(B, y, bw, bh, "no scaling\n(trees do not\nneed it)", ec=GREY, ls=dashed, tc=SECOND, lw=0.9, fs=7.5)
    box(C, y, bw, bh, "Random\nforest", ec=GREEN, bold=True)
    arrow(A + bw, y + bh / 2, B, y + bh / 2, GREEN, dashed); arrow(B + bw, y + bh / 2, C, y + bh / 2, GREEN, dashed)
    y = lanes[2]
    box(A, y, bw, bh, "no fill\n(handled inside)", ec=GREY, ls=dashed, tc=SECOND, lw=0.9, fs=7.5)
    box(B, y, bw, bh, "no scaling", ec=GREY, ls=dashed, tc=SECOND, lw=0.9, fs=7.5)
    box(C, y, bw, bh, "Gradient\nboosting", ec=VIOLET, bold=True)
    arrow(A + bw, y + bh / 2, B, y + bh / 2, VIOLET, dashed); arrow(B + bw, y + bh / 2, C, y + bh / 2, VIOLET, dashed)
    for yy, c in zip(lanes, (BLUE, GREEN, VIOLET)):
        arrow(16, 25, A, yy + bh / 2, c)
        ax.plot([C + bw, 74], [yy + bh / 2, yy + bh / 2], color=c, lw=1.0)
    ax.plot([74, 74], [lanes[2] + bh / 2, lanes[0] + bh / 2], color=SECOND, lw=1.0)
    X1, X2, X3 = 77, 94, 111; w = 14.5
    arrow(74, 25, X1, 25, SECOND)
    box(X1, 12, w, 26, "Predicted\npercentile\nfor each\nathlete")
    box(X2, 12, w, 26, "Sorted within\neach race into\nplaces 1 to n\n(ties at random)")
    box(X3 - 0.5, 12, w + 1, 26, "List MAE\nagainst the\nreal percentile", ec=ORANGE, bold=True, lw=1.6)
    arrow(X1 + w, 25, X2, 25); arrow(X2 + w, 25, X3 - 0.5, 25)
    save(fig, "fig1_pipeline")


# ---------------------------------------------------------------- Figure 2
def fig_alpha():
    """Ridge and lasso: list MAE against the penalty strength alpha."""
    ladder = table("baselines_ladder")
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8), sharey=True)
    for ax, model, title in zip(axes, ["ridge", "lasso"], ["Ridge", "Lasso"]):
        rows = ladder[ladder["model"] == model]
        for part, color in [("validation", BLUE), ("train", ORANGE)]:
            q = rows[rows["part"] == part].sort_values("alpha")
            ax.plot(range(len(q)), q["list_mae_mean"], "-o", color=color, **LINE)
        q = rows[rows["part"] == "validation"].sort_values("alpha")
        ax.set_xticks(range(len(q))); ax.set_xticklabels([f"{a:g}" for a in q["alpha"]])
        ax.set_xlabel(r"Penalty strength $\alpha$"); ax.set_title(title, color=INK, fontsize=10.5, loc="left")
    axes[0].set_ylabel("List MAE (mean of 5 folds)"); axes[0].set_ylim(0.157, 0.170)
    axes[1].plot([], [], "-o", color=BLUE, label="validation", **LINE)
    axes[1].plot([], [], "-o", color=ORANGE, label="training", **LINE)
    axes[1].legend(loc="upper left", fontsize=9.5)
    fig.tight_layout()
    save(fig, "fig2_alpha")


# ---------------------------------------------------------------- Figure 3
def fig_lasso_inputs():
    """In how many of the 5 folds the lasso keeps each input, for each alpha."""
    weights = table("baselines_weights")
    lasso = weights[weights["model"] == "lasso"]
    kept = (lasso.assign(kept=lasso["weight"] != 0)
            .groupby(["feature", "alpha"])["kept"].sum().unstack("alpha"))
    alphas = sorted(kept.columns)
    kept = kept[alphas]
    # Order: the inputs that survive longest first, then by their number in the report.
    order = list(INPUT_NAMES)
    kept = kept.loc[sorted(kept.index, key=lambda f: (-kept.loc[f].sum(), order.index(f)))]
    per_fold = (lasso.assign(kept=lasso["weight"] != 0).groupby(["alpha", "fold"])["kept"].sum())
    M = kept.to_numpy(); nr, nc = M.shape
    cmap = LinearSegmentedColormap.from_list("blues", BLUES)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for i in range(nr):
        for j in range(nc):
            v = int(M[i, j])
            ax.add_patch(plt.Rectangle((j + 0.04, i + 0.06), 0.92, 0.88, fc=cmap(v / 5), ec="none"))
            if v > 0:
                ax.text(j + 0.5, i + 0.5, str(v), ha="center", va="center", fontsize=9.5,
                        color="white" if v >= 3 else INK)
    ax.set_xlim(0, nc); ax.set_ylim(nr, -2.0)
    ax.set_yticks(np.arange(nr) + 0.5); ax.set_yticklabels([INPUT_NAMES[f] for f in kept.index], color=INK, fontsize=9.5)
    ax.set_xticks(np.arange(nc) + 0.5); ax.set_xticklabels([f"{a:g}" for a in alphas], color=INK)
    ax.set_xlabel(r"Penalty strength $\alpha$ (stronger to the right)")
    ax.tick_params(length=0); ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for j, alpha in enumerate(alphas):
        low, high = int(per_fold[alpha].min()), int(per_fold[alpha].max())
        text = str(low) if low == high else f"{low} to {high}"
        ax.text(j + 0.5, -0.42, text, ha="center", va="center", fontsize=10, color=INK, fontweight="bold")
    ax.text(-0.12, -0.42, "Inputs kept", ha="right", va="center", fontsize=10, color=INK, fontweight="bold")
    final = alphas.index(0.01)
    ax.add_patch(plt.Rectangle((final, -0.05), 1, nr + 0.05, fc="none", ec=INK, lw=1.1))
    ax.text(final + 0.5, -1.4, r"$\alpha$ of the" + "\nfinal model", ha="center", va="center", fontsize=9, color=INK)
    save(fig, "fig3_lasso_inputs")


# ---------------------------------------------------------------- Figure 4
def fig_forest():
    """Random forest: training and validation list MAE against the depth."""
    curve = table("models_depth_curve")
    q = curve[curve["inputs"] == "NO_LOAD"].reset_index(drop=True)
    x = np.arange(len(q))
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    ax.plot(x, q["val_list_mae_mean"], "-o", color=BLUE, **LINE)
    ax.plot(x, q["train_list_mae_mean"], "-o", color=ORANGE, **LINE)
    ax.set_xticks(x); ax.set_xticklabels(["no limit" if np.isinf(v) else str(int(v)) for v in q["value"]])
    ax.set_xlabel("Largest depth of the trees"); ax.set_ylabel("List MAE (mean of 5 folds)"); ax.set_ylim(0, 0.19)
    ax.text(x[-1] + 0.15, q["val_list_mae_mean"].iloc[-1], "validation", color=BLUE, va="center", fontsize=10)
    ax.text(x[-1] + 0.15, q["train_list_mae_mean"].iloc[-1], "training", color=ORANGE, va="center", fontsize=10)
    best = int(q["val_list_mae_mean"].idxmin())
    ax.annotate(f"best: depth {int(q['value'][best])}", (best, q["val_list_mae_mean"][best]),
                xytext=(0, 9), textcoords="offset points", ha="center", color=INK, fontsize=9.5)
    save(fig, "fig4_forest")


# ---------------------------------------------------------------- Figure 5
def fig_boosting():
    """Gradient boosting: validation list MAE for each learning rate and depth."""
    cand = table("models_candidates")
    b = cand[(cand["family"] == "boosting") & (cand["inputs"] == "NO_LOAD")]
    grid = b.pivot(index="learning_rate", columns="value", values="val_list_mae_mean").sort_index(ascending=False)
    M = grid.to_numpy(); lo, hi = M.min(), M.max()
    cmap = LinearSegmentedColormap.from_list("blues_r", BLUES[::-1])
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            t = (M[i, j] - lo) / (hi - lo)
            ax.add_patch(plt.Rectangle((j + 0.03, i + 0.04), 0.94, 0.92, fc=cmap(t), ec="none"))
            ax.text(j + 0.5, i + 0.5, f"{M[i, j]:.4f}", ha="center", va="center", fontsize=10,
                    color="white" if t < 0.45 else INK)
    bi, bj = np.unravel_index(M.argmin(), M.shape)
    ax.add_patch(plt.Rectangle((bj + 0.03, bi + 0.04), 0.94, 0.92, fc="none", ec=ORANGE, lw=2.2))
    ax.text(bj + 0.5, bi + 0.82, "best", ha="center", va="center", fontsize=8.5, color="white")
    ax.set_xlim(0, M.shape[1]); ax.set_ylim(M.shape[0], 0)
    ax.set_xticks(np.arange(M.shape[1]) + 0.5); ax.set_xticklabels([str(int(d)) for d in grid.columns])
    ax.set_yticks(np.arange(M.shape[0]) + 0.5); ax.set_yticklabels([f"{r:g}" for r in grid.index])
    ax.set_xlabel("Largest tree depth"); ax.set_ylabel("Learning rate")
    ax.tick_params(length=0); ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(lo, hi)), ax=ax, fraction=0.05, pad=0.04)
    bar.set_label("Validation list MAE (lower is better)", fontsize=9.5)
    bar.outline.set_visible(False); bar.ax.tick_params(length=0, labelsize=8.5); bar.ax.invert_yaxis()
    save(fig, "fig5_boosting")


# ---------------------------------------------------------------- Figure 6
def fig_ladder():
    """The 6 models side by side on validation, bars from 0."""
    ladder = table("baselines_ladder"); cand = table("models_candidates")
    val = ladder[ladder["part"] == "validation"]
    rows = [("Random order", val[val["model"] == "constant"].iloc[0]),
            ("Naive rule", val[val["model"] == "naive"].iloc[0])]
    rows = [(label, r["list_mae_mean"], r["list_mae_std"]) for label, r in rows]
    for key, label, _, _ in FAMILIES:
        best = cand[cand["family"] == key].sort_values("val_list_mae_mean").iloc[0]
        rows.append((label + " (best)", best["val_list_mae_mean"], best["val_list_mae_std"]))
    final = cand[cand["final_model"].astype(str).str.lower() == "true"].iloc[0]
    rows.append((r"Final model (lasso, $\alpha$ = 0.01)", final["val_list_mae_mean"], final["val_list_mae_std"]))
    y = np.arange(len(rows))[::-1]
    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    colors = [GREY, GREY] + [BLUE] * 4 + [ORANGE]
    ax.barh(y, [r[1] for r in rows], xerr=[r[2] for r in rows], height=0.55, color=colors,
            error_kw=dict(ecolor=INK, lw=0.9, capsize=2))
    for yy, r in zip(y, rows):
        ax.text(r[1] + r[2] + 0.004, yy, f"{r[1]:.3f}", va="center", color=INK, fontsize=9.5)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], color=INK)
    ax.set_xlabel(r"Validation list MAE (mean of 5 folds; line: $\pm$ 1 standard deviation)")
    ax.set_xlim(0, 0.38); ax.grid(False); ax.tick_params(axis="y", length=0)
    save(fig, "fig6_models")


# ---------------------------------------------------------------- Figure 7
def fig_candidates():
    """The 64 candidates, best first, with the threshold of the selection rule."""
    cand = table("models_candidates"); sel = table("models_selection").iloc[0]
    c = cand.sort_values("val_list_mae_mean").reset_index(drop=True); c["rank"] = c.index + 1
    fig, ax = plt.subplots(figsize=(6.6, 3.3))
    ax.axhspan(sel["best_mean"], sel["threshold"], color="#eef3fb", zorder=0)
    ax.axhline(sel["threshold"], color=SECOND, lw=0.8, ls="--")
    ax.text(17, sel["threshold"] + 0.00015, "threshold: best + 1 standard error", va="bottom", ha="left",
            color=SECOND, fontsize=8.5)
    for key, label, color, marker in FAMILIES:
        q = c[c["family"] == key]
        ax.scatter(q["rank"], q["val_list_mae_mean"], color=color, marker=marker, s=26, edgecolor="white",
                   linewidth=0.7, label=label, zorder=3)
    final = c[c["final_model"].astype(str).str.lower() == "true"].iloc[0]; best = c.iloc[0]
    for r in (final, best):
        ax.scatter([r["rank"]], [r["val_list_mae_mean"]], s=120, facecolor="none", edgecolor=INK, linewidth=1.2, zorder=4)
    ax.annotate("final model: lasso, " + r"$\alpha$ = 0.01," + "\nwithout load (rank %d)" % final["rank"],
                (final["rank"], final["val_list_mae_mean"]), xytext=(-62, 40), textcoords="offset points",
                ha="center", color=INK, fontsize=9, arrowprops=dict(arrowstyle="-", color=INK, lw=0.7))
    ax.annotate("best model: boosting,\nlearning rate %g, depth %d,\nwithout load (rank 1)" % (best["learning_rate"], best["value"]),
                (best["rank"], best["val_list_mae_mean"]), xytext=(3, 0.1619), textcoords="data",
                ha="left", color=INK, fontsize=9, arrowprops=dict(arrowstyle="-", color=INK, lw=0.7))
    ax.set_ylim(0.158, 0.1697); ax.set_xlim(0, 65.5)
    ax.set_xlabel("The 64 candidates, best first"); ax.set_ylabel("Validation list MAE (mean of 5 folds)")
    ax.legend(loc="upper center", ncol=4, bbox_to_anchor=(0.47, 1.02), fontsize=9, handletextpad=0.2, columnspacing=1.0)
    save(fig, "fig7_candidates")


# ---------------------------------------------------------------- Figure 8
def fig_learning():
    """Learning curves: validation list MAE against the share of training races."""
    curve = table("models_learning_curve")
    fig, ax = plt.subplots(figsize=(5.8, 3.1))
    for key, label, color, marker in FAMILIES:
        q = curve[curve["name"].str.startswith(key)].sort_values("share")
        name = label + " (final model)" if key == "lasso" else label
        ax.plot(q["share"] * 100, q["val_list_mae"], "-", marker=marker, color=color, label=name, **LINE)
    ax.set_xticks([20, 40, 60, 80, 100]); ax.set_ylim(0.158, 0.1626)
    ax.set_xlabel("Share of the training races used (%)"); ax.set_ylabel("Validation list MAE (mean of 5 folds)")
    ax.legend(loc="upper right", ncol=2, fontsize=9)
    save(fig, "fig8_learning")


# ---------------------------------------------------------------- Figure 9
def fig_race():
    """A typical test race: predicted against real place, the author in red."""
    pred = pd.read_csv(PREDICTIONS, dtype={"season": str})
    race = pred[pred["race_id"] == STORY_RACE]
    me = race[race["Name"].str.upper().str.contains(AUTHOR)]
    others = race.drop(me.index); n = len(race)
    fig, ax = plt.subplots(figsize=(4.9, 4.9))
    ax.plot([1, n], [1, n], color=SECOND, lw=0.8, ls="--", label="perfect prediction")
    ax.scatter(others["real_place"], others["pred_place"], s=20, color=BLUE, edgecolor="white", linewidth=0.5,
               label="1 athlete")
    if len(me) == 1:
        me = me.iloc[0]
        ax.scatter([me["real_place"]], [me["pred_place"]], s=62, color=ORANGE, edgecolor="white", linewidth=0.9,
                   zorder=5, label="the author")
        ax.annotate(AUTHOR_LABEL + "\nreal place %d, predicted %d" % (me["real_place"], me["pred_place"]),
                    xy=(me["real_place"], me["pred_place"]), xytext=(3, n - 9), fontsize=9, color=INK,
                    arrowprops=dict(arrowstyle="-", color=SECOND, lw=0.7, shrinkB=5))
    ax.set_xlabel("Real place"); ax.set_ylabel("Place in the predicted list")
    ax.set_xlim(0, n + 2); ax.set_ylim(0, n + 2); ax.set_aspect("equal"); ax.grid(True, axis="both")
    ax.legend(loc="lower right", fontsize=9)
    save(fig, "fig9_race")


# ---------------------------------------------------------------- Figure A1
def fig_shrink():
    """The shrunk dispersion index D* = (n D + 5) / (n + 5) for 3 values of D."""
    n = np.linspace(0, 30, 301); k = 5
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    ax.axhline(1, color=GREY, lw=0.9, zorder=0)
    ax.text(30.4, 1, "pure luck\n(D = 1)", va="center", ha="left", color=SECOND, fontsize=9)
    for d, word, color in [(0.5, "very steady", GREEN), (1.5, "streaky", ORANGE), (2.5, "very streaky", VIOLET)]:
        ax.plot(n, (n * d + k) / (n + k), color=color, lw=1.8)
        ax.hlines(d, 0, 30, colors=color, linestyles=(0, (4, 4)), lw=0.8)
        for point in (2, 20):
            ax.plot([point], [(point * d + k) / (point + k)], "o", ms=5, color=color, mec="white", mew=1.0, zorder=3)
        ax.text(30.4, d, f"raw D = {d}\n{word}", va="center", ha="left", color=INK, fontsize=9.5, linespacing=1.15)
    for point, label in [(2, "2 sprints"), (20, "20 sprints")]:
        ax.axvline(point, color=GRID, lw=0.8, zorder=0)
        ax.text(point, 2.78, label, ha="center", fontsize=9, color=SECOND)
    ax.set_xlim(0, 30); ax.set_ylim(0.3, 2.7); ax.grid(False)
    ax.set_xlabel("Number of sprints n in the window"); ax.set_ylabel("Shrunk index D*")
    save(fig, "figA1_shrink")


if __name__ == "__main__":
    fig_pipeline()
    fig_alpha()
    fig_lasso_inputs()
    fig_forest()
    fig_boosting()
    fig_ladder()
    fig_candidates()
    fig_learning()
    if os.path.exists(PREDICTIONS):
        fig_race()
    else:
        print("skipped fig9_race:", PREDICTIONS, "is missing (run 'python pipeline.py test' first)")
    fig_shrink()
