from pathlib import Path
import random
import pandas as pd
from ultralytics import YOLO


# --------------------------------------------------
# Search settings
# --------------------------------------------------

SEARCH_SEED = 2026
TRAIN_SEED = 42
N_TRIALS = 4

DATASET = "dataset/stage2_det/data.yaml"
MODEL_PATH = "yolo11n.pt"

RESULTS_FILE = Path("stage2_hparam_results.csv")
CONFIG_FILE = Path("stage2_hparam_configs.csv")


# --------------------------------------------------
# Generate random configurations
# --------------------------------------------------

def generate_configs():
    rng = random.Random(SEARCH_SEED)

    configs = []
    seen = set()

    while len(configs) < N_TRIALS:
        config = {
            "trial": len(configs) + 1,
            "lr0": round(rng.uniform(0.0005, 0.002), 6),
            "batch": rng.choice([4, 8]),
            "weight_decay": round(rng.uniform(0.0001, 0.001), 6),
        }

        key = (
            config["lr0"],
            config["batch"],
            config["weight_decay"],
        )

        if key not in seen:
            seen.add(key)
            configs.append(config)

    pd.DataFrame(configs).to_csv(
        CONFIG_FILE,
        index=False
    )

    return configs


# --------------------------------------------------
# Run one training trial
# --------------------------------------------------

def run_trial(config):
    trial_number = config["trial"]

    print("\n" + "=" * 70)
    print(f"STAGE 2 HYPERPARAMETER TRIAL {trial_number}")
    print("=" * 70)

    print(f"Learning rate : {config['lr0']}")
    print(f"Batch size    : {config['batch']}")
    print(f"Weight decay  : {config['weight_decay']}")

    model = YOLO(MODEL_PATH)

    model.train(
        data=DATASET,
        epochs=50,
        imgsz=640,
        batch=config["batch"],
        device=0,
        optimizer="AdamW",
        lr0=config["lr0"],
        weight_decay=config["weight_decay"],
        patience=20,
        workers=0,
        amp=False,
        seed=TRAIN_SEED,
        cache=False,
        plots=True,
        save=True,
        verbose=True,
        project="runs/stage2_hparam",
        name=f"trial_{trial_number:02d}",
    )

    # Get the actual Ultralytics run directory
    save_dir = Path(model.trainer.save_dir)

    results_path = save_dir / "results.csv"

    if not results_path.exists():
        raise FileNotFoundError(
            f"Could not find results.csv at {results_path}"
        )

    df = pd.read_csv(results_path)

    # Select the best epoch using mAP@0.50-0.95
    metric = "metrics/mAP50-95(B)"
    best_index = df[metric].idxmax()
    best_row = df.loc[best_index]

    result = {
        "trial": trial_number,
        "lr0": config["lr0"],
        "batch": config["batch"],
        "weight_decay": config["weight_decay"],
        "best_epoch": int(best_row["epoch"]),

        "best_precision": float(
            best_row["metrics/precision(B)"]
        ),

        "best_recall": float(
            best_row["metrics/recall(B)"]
        ),

        "best_mAP50": float(
            best_row["metrics/mAP50(B)"]
        ),

        "best_mAP50_95": float(
            best_row["metrics/mAP50-95(B)"]
        ),

        "best_val_box_loss": float(
            best_row["val/box_loss"]
        ),

        "best_val_cls_loss": float(
            best_row["val/cls_loss"]
        ),

        "best_val_dfl_loss": float(
            best_row["val/dfl_loss"]
        ),

        "final_mAP50": float(
            df.iloc[-1]["metrics/mAP50(B)"]
        ),

        "final_mAP50_95": float(
            df.iloc[-1]["metrics/mAP50-95(B)"]
        ),

        "epochs_recorded": int(len(df)),

        "run_directory": str(save_dir),
    }

    return result


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    configs = generate_configs()

    print("\nGenerated Stage 2 configurations:")
    print(pd.DataFrame(configs).to_string(index=False))

    all_results = []

    for config in configs:

        try:
            result = run_trial(config)
            all_results.append(result)

            # Save results after every successful trial
            pd.DataFrame(all_results).to_csv(
                RESULTS_FILE,
                index=False
            )

            print("\nTrial completed:")
            print(result)

        except Exception as e:
            print("\nTRIAL FAILED")
            print(f"Trial: {config['trial']}")
            print(f"Error: {e}")

    if all_results:

        results_df = pd.DataFrame(all_results)

        results_df = results_df.sort_values(
            by="best_mAP50_95",
            ascending=False
        )

        results_df.to_csv(
            RESULTS_FILE,
            index=False
        )

        print("\n" + "=" * 70)
        print("STAGE 2 HYPERPARAMETER SEARCH SUMMARY")
        print("=" * 70)

        print(results_df.to_string(index=False))

        print("\nBest Stage 2 trial:")
        print(results_df.iloc[0].to_string())

    else:
        print("\nNo trials completed successfully.")


if __name__ == "__main__":
    main()