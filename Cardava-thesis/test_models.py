from ultralytics import YOLO


print("Loading Stage 1 model...")

stage1_model = YOLO("yolo11n-cls.pt")

print("Stage 1 model loaded successfully.")


print("\nLoading Stage 2 model...")

stage2_model = YOLO("yolo11n.pt")

print("Stage 2 model loaded successfully.")


print("\nBoth models initialized successfully.")