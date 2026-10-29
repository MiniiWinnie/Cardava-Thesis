import os
import shutil
import pandas as pd

from sklearn.model_selection import train_test_split


RAW_DIR = r"dataset/raw"
OUTPUT_DIR = r"dataset/stage1_cls"

RANDOM_STATE = 42


records = []

for class_name in ["Class_A", "Class_B", "Rejected"]:

    class_dir = os.path.join(
        RAW_DIR,
        class_name,
        class_name
    )

    if not os.path.exists(class_dir):
        raise FileNotFoundError(
            f"Missing folder: {class_dir}"
        )

    for filename in os.listdir(class_dir):

        if filename.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff")
        ):

            filepath = os.path.join(
                class_dir,
                filename
            )

            if class_name in ["Class_A", "Class_B"]:
                stage1_class = "Healthy"
            else:
                stage1_class = "Unhealthy"

            records.append({
                "original_class": class_name,
                "stage1_class": stage1_class,
                "filename": filename,
                "filepath": filepath
            })

    files_in_dir = os.listdir(class_dir)
    print(f"{class_name}: {len(files_in_dir)} items found, e.g. {files_in_dir[:5]}")
df = pd.DataFrame(records)

print("Total images:", len(df))

print("\nOriginal classes:")
print(df["original_class"].value_counts())

print("\nStage 1 classes:")
print(df["stage1_class"].value_counts())


train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["stage1_class"],
    random_state=RANDOM_STATE
)


val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["stage1_class"],
    random_state=RANDOM_STATE
)

print("\nTRAIN:")
print(train_df["stage1_class"].value_counts())

print("\nVALIDATION:")
print(val_df["stage1_class"].value_counts())

print("\nTEST:")
print(test_df["stage1_class"].value_counts())


for split_name in ["train", "val", "test"]:

    for class_name in ["Healthy", "Unhealthy"]:

        os.makedirs(
            os.path.join(
                OUTPUT_DIR,
                split_name,
                class_name
            ),
            exist_ok=True
        )

def copy_images(split_df, split_name):

    for _, row in split_df.iterrows():

        destination = os.path.join(
            OUTPUT_DIR,
            split_name,
            row["stage1_class"],
            row["filename"]
        )

        shutil.copy2(
            row["filepath"],
            destination
        )


copy_images(train_df, "train")
copy_images(val_df, "val")
copy_images(test_df, "test")


split_records = []

for _, row in train_df.iterrows():

    split_records.append({
        "filename": row["filename"],
        "original_class": row["original_class"],
        "stage1_class": row["stage1_class"],
        "split": "train"
    })


for _, row in val_df.iterrows():

    split_records.append({
        "filename": row["filename"],
        "original_class": row["original_class"],
        "stage1_class": row["stage1_class"],
        "split": "val"
    })


for _, row in test_df.iterrows():

    split_records.append({
        "filename": row["filename"],
        "original_class": row["original_class"],
        "stage1_class": row["stage1_class"],
        "split": "test"
    })


split_df = pd.DataFrame(split_records)

split_df.to_csv(
    "stage1_split_record.csv",
    index=False
)


print("\n========================================")
print("STAGE 1 DATASET PREPARATION COMPLETE")
print("========================================")

print(
    f"Training images:   {len(train_df)}"
)

print(
    f"Validation images: {len(val_df)}"
)

print(
    f"Testing images:    {len(test_df)}"
)

print(
    "\nOutput folder:"
)

print(
    os.path.abspath(OUTPUT_DIR)
)

print(
    "\nSplit record:"
)

print(
    os.path.abspath("stage1_split_record.csv")
)