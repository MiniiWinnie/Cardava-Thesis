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
    / "stage1_hparam_search_final"
)

# Corrected validation results
INPUT_CSV = (
    SEARCH_DIR
    / "stage1_hparam_corrected_validation.csv"
)

FIGURE_DIR = (
    SEARCH_DIR
    / "figures"
)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

if not INPUT_CSV.exists():
    raise FileNotFoundError(
        f"Corrected validation CSV not found:\n{INPUT_CSV}"
    )

df = pd.read_csv(INPUT_CSV)

required_columns = {
    "trial",
    "corrected_val_accuracy",
    "macro_f1",
    "lowest_val_loss",
}

missing_columns = (
    required_columns
    - set(df.columns)
)

if missing_columns:
    raise ValueError(
        "Missing required columns:\n"
        + "\n".join(sorted(missing_columns))
    )


# Sort by actual trial number
df = (
    df
    .sort_values("trial")
    .reset_index(drop=True)
)


# ============================================================
# CONVERT VALUES TO PERCENTAGES
# ============================================================

df["accuracy_percent"] = (
    df["corrected_val_accuracy"] * 100
)

df["macro_f1_percent"] = (
    df["macro_f1"] * 100
)


# ============================================================
# FIGURE 1
# VALIDATION ACCURACY + MACRO F1
# ============================================================

plt.figure(
    figsize=(9, 6)
)

x = range(len(df))
width = 0.35

plt.bar(
    [i - width / 2 for i in x],
    df["accuracy_percent"],
    width=width,
    label="Validation Accuracy"
)

plt.bar(
    [i + width / 2 for i in x],
    df["macro_f1_percent"],
    width=width,
    label="Macro F1"
)

plt.xticks(
    list(x),
    [f"Trial {t}" for t in df["trial"]]
)

plt.ylabel("Percentage (%)")

plt.xlabel("Hyperparameter Trial")

plt.title(
    "Stage 1 Validation Accuracy and Macro F1 Across Hyperparameter Trials"
)

plt.ylim(
    0,
    100
)

plt.legend()

plt.tight_layout()

FIGURE_1 = (
    FIGURE_DIR
    / "stage1_hparam_accuracy_macro_f1.png"
)

plt.savefig(
    FIGURE_1,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FIGURE 2
# MINIMUM VALIDATION LOSS
# ============================================================

plt.figure(
    figsize=(9, 6)
)

plt.bar(
    [f"Trial {t}" for t in df["trial"]],
    df["lowest_val_loss"]
)

plt.xlabel(
    "Hyperparameter Trial"
)

plt.ylabel(
    "Minimum Validation Loss"
)

plt.title(
    "Minimum Validation Loss Across Stage 1 Hyperparameter Trials"
)

plt.tight_layout()

FIGURE_2 = (
    FIGURE_DIR
    / "stage1_hparam_min_validation_loss.png"
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
print("STAGE 1 HYPERPARAMETER FIGURES GENERATED")
print("=" * 70)

print()
print("Source CSV:")
print(INPUT_CSV)

print()
print("Figure 1:")
print(FIGURE_1)

print()
print("Figure 2:")
print(FIGURE_2)

print()
print("Values used:")
print()

print(
    df[
        [
            "trial",
            "corrected_val_accuracy",
            "macro_f1",
            "lowest_val_loss",
            "lowest_val_loss_epoch",
        ]
    ].to_string(index=False)
)

print()
print("[DONE]")