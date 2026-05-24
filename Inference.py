# =========================================================
# FACE MASK DETECTION - TESTING SCRIPT
# FOR VS CODE
# =========================================================
# OUTPUT:
# 1. Detect objects from test images
# 2. Draw bounding boxes
# 3. Show confidence score
# 4. Show instance count on image
# 5. Save output images automatically
# =========================================================

import os
import cv2
import torch
import torchvision
import numpy as np

from PIL import Image
from torchvision.transforms import functional as F
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# =========================================================
# CONFIGURATION
# =========================================================

TEST_FOLDER = r"C:\Users\adity\Desktop\Mask Detection CV Project\Face Mask Wearing Conditon Detection\Test Set_2"

OUTPUT_FOLDER = r"C:\Users\adity\Desktop\Mask Detection CV Project\Output_2"

MODEL_PATH = r"C:\Users\adity\Desktop\Mask Detection CV Project\Model\best_val_loss.pth"

# Your classes
# IMPORTANT:
# Index 0 = background

CLASS_NAMES = [
    "background",
    "with_mask",
    "without_mask",
    "mask_weared_incorrect"
]

# Detection threshold
CONFIDENCE_THRESHOLD = 0.5

# =========================================================
# DEVICE
# =========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using Device:", device)

# =========================================================
# CREATE OUTPUT FOLDER
# =========================================================

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# =========================================================
# LOAD MODEL
# =========================================================

num_classes = 4

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
    weights=None
)

in_features = model.roi_heads.box_predictor.cls_score.in_features

model.roi_heads.box_predictor = FastRCNNPredictor(
    in_features,
    num_classes
)

# Load checkpoint
checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(device)

model.eval()

print("Model loaded successfully!")

# =========================================================
# IMAGE FILES
# =========================================================

image_files = [
    f for f in os.listdir(TEST_FOLDER)
    if f.lower().endswith((
        ".jpg",
        ".jpeg",
        ".png"
    ))
]

print(f"\nTotal Test Images: {len(image_files)}")

# =========================================================
# TEST LOOP
# =========================================================

for image_name in image_files:

    image_path = os.path.join(
        TEST_FOLDER,
        image_name
    )

    # -----------------------------------------
    # READ IMAGE
    # -----------------------------------------

    image = Image.open(image_path).convert("RGB")

    image_np = np.array(image)

    image_tensor = F.to_tensor(image).to(device)

    # -----------------------------------------
    # PREDICTION
    # -----------------------------------------

    with torch.no_grad():

        prediction = model([image_tensor])[0]

    boxes = prediction["boxes"].cpu().numpy()

    scores = prediction["scores"].cpu().numpy()

    labels = prediction["labels"].cpu().numpy()

    # -----------------------------------------
    # INSTANCE COUNTS
    # -----------------------------------------

    counts = {}

    # -----------------------------------------
    # DRAW BOXES
    # -----------------------------------------

    for box, score, label in zip(
        boxes,
        scores,
        labels
    ):

        if score < CONFIDENCE_THRESHOLD:
            continue

        x1, y1, x2, y2 = map(int, box)

        class_name = CLASS_NAMES[label]

        counts[class_name] = counts.get(
            class_name,
            0
        ) + 1

        # -------------------------------------
        # BOX COLOR
        # -------------------------------------

        if class_name == "with_mask":
            color = (0, 255, 0)

        elif class_name == "without_mask":
            color = (0, 0, 255)

        else:
            color = (255, 165, 0)

        # -------------------------------------
        # DRAW RECTANGLE
        # -------------------------------------

        cv2.rectangle(
            image_np,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        # -------------------------------------
        # LABEL TEXT
        # -------------------------------------

        text = f"{class_name}: {score:.2f}"

        cv2.putText(
            image_np,
            text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    # =====================================================
    # PRINT INSTANCE COUNT ON IMAGE
    # =====================================================

    y_position = 30

    for class_name, count in counts.items():

        count_text = f"{class_name}: {count}"

        cv2.putText(
            image_np,
            count_text,
            (20, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        y_position += 35

    # =====================================================
    # SAVE OUTPUT IMAGE
    # =====================================================

    output_path = os.path.join(
        OUTPUT_FOLDER,
        image_name
    )

    cv2.imwrite(
        output_path,
        cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    )

    print(f"Processed: {image_name}")

# =========================================================
# FINISHED
# =========================================================

print("\n===================================")
print("ALL TEST IMAGES PROCESSED")
print("OUTPUT SAVED SUCCESSFULLY")
print("===================================")

# !pip install torch torchvision torchaudio opencv-python pillow matplotlib numpy tqdm pycocotools torchmetrics -q