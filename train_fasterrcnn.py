"""
Faster R-CNN + ResNet50 FPN — Face Mask Detection
Dataset  : COCO-format annotations.json
GPU      : Kaggle T4
Split    : 80% train / 20% val
Saves    : best_train_loss.pth | best_val_loss.pth | final_epoch.pth
Augment  : Heavy (flip, color jitter, blur, grayscale, perspective, cutout)
"""

# ── 0. Imports ────────────────────────────────────────────────────────
import os, json, random
import numpy as np
from PIL import Image
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader, Subset

from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import albumentations as A
from albumentations.pytorch import ToTensorV2

from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ── 1. CONFIG ─────────────────────────────────────────────────────────
ANNOTATIONS = "/kaggle/input/datasets/arbmonty/face-mask-wearing-condition-detection/annotations.json"
IMAGES_DIR  = "/kaggle/input/datasets/arbmonty/face-mask-wearing-condition-detection/Images/Images"
OUTPUT_DIR  = "/kaggle/working"

NUM_CLASSES  = 4        # background(0) + with_mask(1) + without_mask(2) + mask_weared_incorrect(3)
NUM_EPOCHS   = 30
BATCH_SIZE   = 4
LR           = 0.005
MOMENTUM     = 0.9
WEIGHT_DECAY = 0.0005
VAL_SPLIT    = 0.2
SEED         = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ── 2. LOSSES USED ────────────────────────────────────────────────────
"""
Faster R-CNN computes 4 losses automatically and returns them as a dict:

┌─────────────────────────┬──────────────────┬──────────────────────────────────────┐
│ Loss Name               │ Type             │ Purpose                              │
├─────────────────────────┼──────────────────┼──────────────────────────────────────┤
│ loss_objectness         │ Binary CrossEnt  │ RPN: object vs background per anchor │
│ loss_rpn_box_reg        │ Smooth L1        │ RPN: refine anchor box coordinates   │
│ loss_classifier         │ CrossEntropy     │ ROI: classify region into class      │
│ loss_box_reg            │ Smooth L1        │ ROI: refine final box per class      │
└─────────────────────────┴──────────────────┴──────────────────────────────────────┘

Total Loss = loss_objectness + loss_rpn_box_reg + loss_classifier + loss_box_reg
This total is what gets backpropagated.
"""


# ── 3. HEAVY AUGMENTATIONS ────────────────────────────────────────────
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.1),
    A.RandomRotate90(p=0.2),
    A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2, p=0.7),
    A.GaussianBlur(blur_limit=(3, 7), p=0.3),
    A.GaussNoise(var_limit=(10, 50), p=0.3),
    A.RandomGrayscale(p=0.1),
    A.RandomBrightnessContrast(p=0.4),
    A.HueSaturationValue(p=0.3),
    A.Perspective(scale=(0.05, 0.1), p=0.3),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.4),
    A.CoarseDropout(max_holes=8, max_height=32, max_width=32,
                    min_holes=1, fill_value=0, p=0.3),  # Cutout
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
], bbox_params=A.BboxParams(format="coco", label_fields=["class_labels"], min_visibility=0.3))

val_transform = A.Compose([
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
], bbox_params=A.BboxParams(format="coco", label_fields=["class_labels"], min_visibility=0.3))


# ── 4. DATASET ────────────────────────────────────────────────────────
class MaskDataset(Dataset):
    def __init__(self, annotations_file, images_dir, transform=None):
        with open(annotations_file) as f:
            self.coco = json.load(f)
        self.images_dir = images_dir
        self.transform  = transform
        self.img_map    = {img["id"]: img for img in self.coco["images"]}
        self.ann_map    = defaultdict(list)
        for ann in self.coco["annotations"]:
            self.ann_map[ann["image_id"]].append(ann)
        self.img_ids = list(self.img_map.keys())

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id   = self.img_ids[idx]
        img_info = self.img_map[img_id]
        anns     = self.ann_map[img_id]

        img_path = os.path.join(self.images_dir, img_info["file_name"])
        image    = np.array(Image.open(img_path).convert("RGB"))

        boxes, labels = [], []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w > 1 and h > 1:
                boxes.append([x, y, w, h])
                labels.append(ann["category_id"] + 1)  # 0 = background

        if self.transform and len(boxes) > 0:
            out    = self.transform(image=image, bboxes=boxes, class_labels=labels)
            image  = out["image"]
            boxes  = list(out["bboxes"])
            labels = list(out["class_labels"])
        else:
            plain = A.Compose([
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
            image = plain(image=image)["image"]

        # COCO [x,y,w,h] → Pascal VOC [x1,y1,x2,y2]
        converted = [[x, y, x+w, y+h] for x, y, w, h in boxes]

        if len(converted) == 0:
            target = {"boxes":  torch.zeros((0,4), dtype=torch.float32),
                      "labels": torch.zeros((0,),  dtype=torch.int64),
                      "image_id": torch.tensor([img_id])}
        else:
            target = {"boxes":  torch.tensor(converted, dtype=torch.float32),
                      "labels": torch.tensor(labels,    dtype=torch.int64),
                      "image_id": torch.tensor([img_id])}
        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))


# ── 5. LOAD & SPLIT ───────────────────────────────────────────────────
print("Loading dataset...")
total     = len(MaskDataset(ANNOTATIONS, IMAGES_DIR))
val_size  = int(total * VAL_SPLIT)
train_size= total - val_size
indices   = list(range(total))
random.shuffle(indices)
train_idx, val_idx = indices[:train_size], indices[train_size:]
print(f"Total: {total} | Train: {train_size} | Val: {val_size}")

train_dataset = Subset(MaskDataset(ANNOTATIONS, IMAGES_DIR, transform=train_transform), train_idx)
val_dataset   = Subset(MaskDataset(ANNOTATIONS, IMAGES_DIR, transform=val_transform),   val_idx)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          collate_fn=collate_fn, num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                          collate_fn=collate_fn, num_workers=2, pin_memory=True)


# ── 6. MODEL ──────────────────────────────────────────────────────────
print("Building Faster R-CNN + ResNet50 FPN...")
model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
model.to(DEVICE)
print(f"Model ready — {NUM_CLASSES} classes (0=bg, 1=with_mask, 2=without_mask, 3=mask_weared_incorrect)")


# ── 7. OPTIMIZER & SCHEDULER ──────────────────────────────────────────
params    = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


# ── 8. TRAINING LOOP ──────────────────────────────────────────────────
train_losses, val_losses = [], []
best_train_loss = float("inf")
best_val_loss   = float("inf")

print("\n" + "="*60)
print("Starting Training")
print("="*60)

for epoch in range(1, NUM_EPOCHS + 1):

    # ── TRAIN ──
    model.train()
    epoch_train_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{NUM_EPOCHS} [Train]", leave=True)

    for images, targets in pbar:
        images  = [img.to(DEVICE) for img in images]
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses    = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        epoch_train_loss += losses.item()
        pbar.set_postfix({
            "total" : f"{losses.item():.3f}",
            "cls"   : f"{loss_dict.get('loss_classifier',  torch.tensor(0)).item():.3f}",
            "box"   : f"{loss_dict.get('loss_box_reg',     torch.tensor(0)).item():.3f}",
            "rpn"   : f"{loss_dict.get('loss_rpn_box_reg', torch.tensor(0)).item():.3f}",
            "obj"   : f"{loss_dict.get('loss_objectness',  torch.tensor(0)).item():.3f}",
        })

    avg_train = epoch_train_loss / len(train_loader)
    train_losses.append(avg_train)

    # ── VALIDATION ──
    model.train()  # keep train mode to get loss dict
    epoch_val_loss = 0.0
    pbar_v = tqdm(val_loader, desc=f"Epoch {epoch:02d}/{NUM_EPOCHS} [Val]  ", leave=True)

    with torch.no_grad():
        for images, targets in pbar_v:
            images  = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            loss_dict      = model(images, targets)
            losses         = sum(loss for loss in loss_dict.values())
            epoch_val_loss += losses.item()
            pbar_v.set_postfix({"val_loss": f"{losses.item():.3f}"})

    avg_val = epoch_val_loss / len(val_loader)
    val_losses.append(avg_val)
    scheduler.step()

    print(f"\nEpoch {epoch:02d} | Train: {avg_train:.4f} | Val: {avg_val:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

    # ── SAVE BEST TRAIN LOSS ──
    if avg_train < best_train_loss:
        best_train_loss = avg_train
        torch.save({"epoch": epoch, "train_loss": avg_train, "val_loss": avg_val,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict()},
                   f"{OUTPUT_DIR}/best_train_loss.pth")
        print(f"  ✅ best_train_loss.pth saved  (train={best_train_loss:.4f})")

    # ── SAVE BEST VAL LOSS ──
    if avg_val < best_val_loss:
        best_val_loss = avg_val
        torch.save({"epoch": epoch, "train_loss": avg_train, "val_loss": avg_val,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict()},
                   f"{OUTPUT_DIR}/best_val_loss.pth")
        print(f"  ✅ best_val_loss.pth saved    (val={best_val_loss:.4f})")

    print("-" * 60)

# ── SAVE FINAL EPOCH ──
torch.save({"epoch": NUM_EPOCHS, "train_loss": train_losses[-1], "val_loss": val_losses[-1],
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()},
           f"{OUTPUT_DIR}/final_epoch.pth")
print("✅ final_epoch.pth saved")


# ── 9. LOSS CURVE ─────────────────────────────────────────────────────
plt.figure(figsize=(10, 5))
plt.plot(range(1, NUM_EPOCHS+1), train_losses, label="Train Loss", marker="o", color="steelblue")
plt.plot(range(1, NUM_EPOCHS+1), val_losses,   label="Val Loss",   marker="s", color="darkorange")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Faster R-CNN ResNet50 — Train vs Val Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/loss_curve.png", dpi=150)
plt.show()

print("\n" + "="*60)
print("Training Complete!")
print(f"  Best Train Loss : {best_train_loss:.4f}")
print(f"  Best Val Loss   : {best_val_loss:.4f}")
print("  Saved → best_train_loss.pth | best_val_loss.pth | final_epoch.pth")
print("="*60)
