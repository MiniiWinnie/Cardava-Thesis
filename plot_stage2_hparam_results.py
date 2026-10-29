from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\saiar\Pictures\cardava_thesis"
)

SEARCH_DIR = (
    PROJECT_ROOT
    / "runs"
    / "stage2_hparam_search_final"
)

FIGURE_DIR = (
    SEARCH_DIR
    / "figures"
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SEARCH RESULTS
# ============================================================

SUMMARY_CSV = (
    SEARCH_DIR
    / "stage2_hparam_results.csv"
)

if not SUMMARY_CSV.exists():
    raise FileNotFoundError(
        f"Stage 2 hyperparameter results not found:\n{SUMMARY_CSV}"
    )


summary = pd.read_csv(
    SUMMARY_CSV
)

summary = (
    summary
    .sort_values("trial")
    .reset_index(drop=True)
)


# ============================================================
# VERIFY REQUIRED COLUMNS
# ============================================================

required_summary_columns = {
    "trial",
    "best_val_map50_95",
    "best_val_map50",
    "best_val_precision",
    "best_val_recall",
    "final_val_map50_95",
    "best_val_epoch",
}

missing = (
    required_summary_columns
    - set(summary.columns)
)

if missing:
    raise ValueError(
        "Missing required columns in Stage 2 summary:\n"
        + "\n".join(sorted(missing))
    )


# ============================================================
# FIGURE 1
# BEST VALIDATION mAP@0.50:0.95
# ============================================================

plt.figure(
    figsize=(9, 6)
)

plt.bar(
    [f"Trial {t}" for t in summary["trial"]],
    summary["best_val_map50_95"] * 100
)

plt.xlabel(
    "Hyperparameter Trial"
)

plt.ylabel(
    "mAP@0.50:0.95 (%)"
)

plt.title(
    "Best Validation mAP@0.50:0.95 Across Stage 2 Hyperparameter Trials"
)

plt.ylim(
    0,
    100
)

plt.tight_layout()

FIGURE_1 = (
    FIGURE_DIR
    / "stage2_hparam_best_map50_95.png"
)

plt.savefig(
    FIGURE_1,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# EPOCH HISTORY FILES
# ============================================================

TRIAL_FILES = {
    1: SEARCH_DIR / "trial_01" / "results.csv",
    2: SEARCH_DIR / "trial_02" / "results.csv",
    3: SEARCH_DIR / "trial_03" / "results.csv",
    4: SEARCH_DIR / "trial_04" / "results.csv",
}


# ============================================================
# FIGURE 2
# VALIDATION mAP@0.50:0.95 ACROSS EPOCHS
# ============================================================

plt.figure(
    figsize=(10, 6)
)

for trial, csv_path in TRIAL_FILES.items():

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Trial {trial} results.csv not found:\n{csv_path}"
        )

    df = pd.read_csv(
        csv_path
    )

    required_columns = {
        "epoch",
        "metrics/mAP50-95(B)",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"Trial {trial} is missing columns:\n"
            + "\n".join(sorted(missing))
        )

    plt.plot(
        df["epoch"],
        df["metrics/mAP50-95(B)"] * 100,
        label=f"Trial {trial}"
    )


plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Validation mAP@0.50:0.95 (%)"
)

plt.title(
    "Validation mAP@0.50:0.95 Across Epochs for Stage 2 Hyperparameter Trials"
)

plt.ylim(
    0,
    100
)

plt.legend()

plt.tight_layout()

FIGURE_2 = (
    FIGURE_DIR
    / "stage2_hparam_map50_95_training_curves.png"
)

plt.savefig(
    FIGURE_2,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# PRINT RESULTS
# ============================================================

print("=" * 70)
print("STAGE 2 HYPERPARAMETER FIGURES GENERATED")
print("=" * 70)

print()
print("Summary source:")
print(SUMMARY_CSV)

print()
print("Figure 1:")
print(FIGURE_1)

print()
print("Figure 2:")
print(FIGURE_2)

print()
print("Trial summary:")
print(
    summary[
        [
            "trial",
            "learning_rate",
            "batch_size",
            "weight_decay",
            "best_val_map50_95",
            "best_val_epoch",
            "best_val_map50",
            "best_val_precision",
            "best_val_recall",
            "final_val_map50_95",
        ]
    ].to_string(index=False)
)

print()
print("[DONE]")