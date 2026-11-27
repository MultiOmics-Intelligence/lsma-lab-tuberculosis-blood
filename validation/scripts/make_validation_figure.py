#!/usr/bin/env python3
"""Figure for the validation chapter. Usage: python make_validation_figure.py <validation_dir>"""
import json, pathlib, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NAVY, CORAL, PLUM, GREY, TEAL = "#0F1B33", "#F06A42", "#8A2D5B", "#56607A", "#1E6F6B"


def main():
    d = pathlib.Path(sys.argv[1])
    rep = json.loads((d / "tables" / "validation_summary.json").read_text())
    scores = pd.read_csv(d / "tables" / "validation_signature_scores.tsv", sep="\t")
    paired = pd.read_csv(d / "tables" / "validation_paired_scores.tsv", sep="\t", index_col=0)

    fig = plt.figure(figsize=(15, 5.6))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.15, 1], wspace=0.3)
    ax1, ax2, ax3 = (fig.add_subplot(gs[0, i]) for i in range(3))

    # 1. signature score at diagnosis vs after treatment
    sub = scores[scores["timepoint_tidy"].isin(["Diagnosis", "Month_6"])]
    groups = [sub.loc[sub["timepoint_tidy"] == g, "score"].dropna().to_numpy()
              for g in ("Diagnosis", "Month_6")]
    parts = ax1.violinplot(groups, positions=[0, 1], widths=0.75, showextrema=False)
    for b, c in zip(parts["bodies"], (CORAL, TEAL)):
        b.set_facecolor(c); b.set_alpha(0.35); b.set_edgecolor(c)
    for i, (g, c) in enumerate(zip(groups, (CORAL, TEAL))):
        x = np.random.default_rng(1).normal(i, 0.055, len(g))
        ax1.scatter(x, g, s=11, color=c, alpha=0.75, zorder=3, linewidths=0)
        ax1.hlines(np.median(g), i - 0.3, i + 0.3, color=NAVY, lw=2.2, zorder=4)
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels([f"at diagnosis\n(n={len(groups[0])})", f"after 6 months\n(n={len(groups[1])})"],
                        fontsize=9.6, color=NAVY)
    ax1.set_ylabel("Tuberculosis signature score\nin the independent cohort", fontsize=9.8, color=NAVY)
    es = rep["external_signature"]
    ax1.set_title(f"The signature carries over\nAUC {es['auc_diagnosis_vs_month6']:.3f}",
                  fontsize=12, color=NAVY, loc="left", pad=10)
    ax1.grid(axis="y", color="#DDE1E8"); ax1.set_axisbelow(True)
    for s in ("top", "right"): ax1.spines[s].set_visible(False)

    # 2. the same patients, before and after
    fell = 0
    for _, r in paired.iterrows():
        down = r["Diagnosis"] > r["Month_6"]
        fell += bool(down)
        ax2.plot([0, 1], [r["Diagnosis"], r["Month_6"]], color=(TEAL if down else CORAL),
                 alpha=0.45, lw=1.1, zorder=2)
        ax2.scatter([0, 1], [r["Diagnosis"], r["Month_6"]], s=9,
                    color=(TEAL if down else CORAL), zorder=3, linewidths=0)
    ax2.set_xlim(-0.25, 1.25); ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["diagnosis", "6 months"], fontsize=10, color=NAVY)
    ax2.set_ylabel("Signature score", fontsize=9.8, color=NAVY)
    pr = rep.get("external_signature_paired", {})
    ax2.set_title(f"The same {len(paired)} patients, treated\n"
                  f"{fell} of {len(paired)} fall", fontsize=12, color=NAVY, loc="left", pad=10)
    ax2.grid(axis="y", color="#DDE1E8"); ax2.set_axisbelow(True)
    for s in ("top", "right"): ax2.spines[s].set_visible(False)

    # 3. what the model does, and the sex control
    sx = rep["sex_check"]
    bars = [("model, held-out\nsame cohort", 0.8152, GREY),
            ("model, independent\ncohort", rep["external_model"]["auc"], NAVY),
            ("model with sex genes\nremoved", sx["tb_model_without_sex_linked_genes"]["external_auc"], NAVY),
            ("predicting sex\ninstead", sx["predicting_sex_instead"]["cv_auc_mean"], PLUM)]
    ys = np.arange(len(bars))[::-1]
    ax3.barh(ys, [b[1] for b in bars], color=[b[2] for b in bars], height=0.58, zorder=3)
    for y, b in zip(ys, bars):
        ax3.text(b[1] + 0.015, y, f"{b[1]:.3f}", va="center", fontsize=10.5, fontweight="bold", color=NAVY)
    ax3.set_yticks(ys); ax3.set_yticklabels([b[0] for b in bars], fontsize=9, color=NAVY)
    ax3.axvline(0.5, color=GREY, ls="--", lw=1.1, zorder=2)
    ax3.text(0.5, len(bars) - 0.35, " chance", fontsize=8.5, color=GREY)
    ax3.set_xlim(0, 1.17); ax3.set_xlabel("Area under the ROC curve", fontsize=9.8, color=NAVY)
    ax3.set_title("Is it tuberculosis, or is it sex?\nNone of the 200 features is sex-linked",
                  fontsize=12, color=NAVY, loc="left", pad=10)
    ax3.grid(axis="x", color="#DDE1E8"); ax3.set_axisbelow(True)
    for s in ("top", "right", "left"): ax3.spines[s].set_visible(False)

    fig.suptitle("Tested in 76 other patients, in three other countries",
                 fontsize=14.5, color=NAVY, x=0.009, ha="left", y=0.985)
    fig.text(0.009, 0.015, "LSMA Analysis Lab  ·  discovery GSE193777, validation GSE193978  ·  "
                           "educational and research material, not medical advice", fontsize=8.4, color=GREY)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.80, bottom=0.17)
    for ext in ("png", "pdf"):
        fig.savefig(d / f"validation-independent-cohort.{ext}", dpi=190 if ext == "png" else None,
                    facecolor="white", bbox_inches="tight")
    print("wrote validation-independent-cohort.png / .pdf")


if __name__ == "__main__":
    main()
