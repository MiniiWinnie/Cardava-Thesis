PRESENCE MODEL IMPLEMENTATION KIT

Contains:
  training/propose_rejected_boxes.py
  training/merge_rejected_labels.py
  training/build_presence_dataset.py
  training/train_presence.py
  training/evaluate_presence.py
  docs/PRESENCE_MODEL.md

Copy the training/ and docs/ folders into the existing thesis repository root (the directory containing train_stage2_final.py, yolo11n.pt, stage1_split_record.csv, and stage2_annotation_preview/stage2_annotation_record.csv).

Read docs/PRESENCE_MODEL.md before running any script. The kit does not contain the original dataset images or the 150 new background photographs. It cannot train a checkpoint until those images and the reviewed Rejected labels are supplied.

The code was Python syntax-checked and the dataset builder was tested against synthetic miniature data. It has NOT been run on the actual project dataset, has NOT trained any weights, and has NOT been tested on the Raspberry Pi application. If you do not have background photos yet, use `python training/build_presence_dataset.py --allow-no-background` for preliminary banana-only training. The evaluator will explicitly mark background false-trigger performance as NOT EVALUATED and will not mark all targets as passed. Targets in the guide are acceptance criteria, not achieved results.


LOCAL DATA FOLDER CHECKLIST

Project root: C:\Users\saiar\Pictures\cardava_thesis
Confirm these dataset folders exist:
  C:\Users\saiar\Pictures\cardava_thesis\dataset\raw\Class_A\
  C:\Users\saiar\Pictures\cardava_thesis\dataset\raw\Class_B\
  C:\Users\saiar\Pictures\cardava_thesis\dataset\raw\Rejected\Rejected\  (original Rejected photos)

Also confirm these project files exist under the same project root:
  stage1_split_record.csv
  stage2_annotation_preview/stage2_annotation_record.csv
  yolo11n.pt
  runs/stage2_final/weights/best.pt

The dataset builder searches recursively inside dataset/raw, so the nested Rejected folder is supported. The rejected-box proposal script defaults directly to dataset/raw/Rejected/Rejected.

The Stage 2 checkpoint and pretrained checkpoint are already in the uploaded thesis ZIP. The original 1,500 image files were not in that ZIP, but the scripts can read the files already stored on your computer. You do not need to upload all image files to ChatGPT to run the scripts locally.

For the current no-background situation, continue box review first, then run:
  python training/build_presence_dataset.py --allow-no-background
  python training/train_presence.py
  python training/evaluate_presence.py

This is a preliminary banana-only run. It cannot establish false-trigger performance on empty scenes, and the evaluator will not mark the complete acceptance targets as passed.
