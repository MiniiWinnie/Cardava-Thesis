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

DATASET = "dataset/stage1_cls"
MODEL_PATH = "yolo11n-cls.pt"

RESULTS_FILE = Path("stage1_hparam_results.csv")
CONFIG_FILE = Path("stage1_hparam_configs.csv")


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
            "batch": rng.choice([8, 16]),
            "weight_decay": round(rng.uniform(0.0001, 0.001), 6),
            "dropout": round(rng.uniform(0.0, 0.2), 3),
        }

        key = (
            config["lr0"],
            config["batch"],
            config["weight_decay"],
            config["dropout"],
        )

        if key not in seen:
            seen.add(key)
            configs.append(config)

    pd.DataFrame(configs).to_csv(CONFIG_FILE, index=False)

    return configs


# --------------------------------------------------
# Run one training trial
# --------------------------------------------------

def run_trial(config):
    trial_number = config["trial"]

    print("\n" + "=" * 70)
    print(f"STAGE 1 HYPERPARAMETER TRIAL {trial_number}")
    print("=" * 70)

    print(f"Learning rate : {config['lr0']}")
    print(f"Batch size    : {config['batch']}")
    print(f"Weight decay  : {config['weight_decay']}")
    print(f"Dropout       : {config['dropout']}")

    model = YOLO(MODEL_PATH)

    model.train(
        data=DATASET,
        epochs=50,
        imgsz=224,
        batch=config["batch"],
        device=0,
        optimizer="AdamW",
        lr0=config["lr0"],
        weight_decay=config["weight_decay"],
        dropout=config["dropout"],
        patience=20,
        workers=0,
        amp=False,
        seed=TRAIN_SEED,
        cache=False,
        plots=True,
        save=True,
        verbose=True,
        project="runs/stage1_hparam",
        name=f"trial_{trial_number:02d}",
    )

    # Locate the actual Ultralytics run directory
    save_dir = Path(model.trainer.save_dir)

    results_path = save_dir / "results.csv"

    if not results_path.exists():
        raise FileNotFoundError(
            f"Could not find results.csv at {results_path}"
        )

    df = pd.read_csv(results_path)

    # Best validation accuracy
    best_index = df["metrics/accuracy_top1"].idxmax()
    best_row = df.loc[best_index]

    result = {
        "trial": trial_number,
        "lr0": config["lr0"],
        "batch": config["batch"],
        "weight_decay": config["weight_decay"],
        "dropout": config["dropout"],
        "best_epoch": int(best_row["epoch"]),
        "best_val_accuracy": float(
            best_row["metrics/accuracy_top1"]
        ),
        "best_val_loss": float(
            best_row["val/loss"]
        ),
        "final_val_accuracy": float(
            df.iloc[-1]["metrics/accuracy_top1"]
        ),
        "final_val_loss": float(
            df.iloc[-1]["val/loss"]
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

    print("\nGenerated Stage 1 configurations:")
    print(pd.DataFrame(configs).to_string(index=False))

    all_results = []

    for config in configs:
        try:
            result = run_trial(config)
            all_results.append(result)

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
            by="best_val_accuracy",
            ascending=False
        )

        results_df.to_csv(
            RESULTS_FILE,
            index=False
        )

        print("\n" + "=" * 70)
        print("STAGE 1 HYPERPARAMETER SEARCH SUMMARY")
        print("=" * 70)
        print(results_df.to_string(index=False))

        print("\nBest Stage 1 trial:")
        print(results_df.iloc[0].to_string())

    else:
        print("\nNo trials completed successfully.")


if __name__ == "__main__":
    main()