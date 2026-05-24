"""
coco.py — COCO Dataset Utility
Provides tools to load, parse, query, and convert COCO-format annotation files.
Also includes a YOLO <-> COCO converter.
"""

import json
import os
from pathlib import Path
from collections import defaultdict


class COCO:
    """
    COCO dataset helper class.
    Loads and parses a COCO-format annotations JSON file.
    """

    def __init__(self, annotation_file=None):
        self.dataset = {}
        self.anns = {}       # ann_id  -> annotation dict
        self.imgs = {}       # img_id  -> image dict
        self.cats = {}       # cat_id  -> category dict
        self.imgToAnns = defaultdict(list)  # img_id  -> [ann, ...]
        self.catToImgs = defaultdict(list)  # cat_id  -> [img_id, ...]

        if annotation_file:
            self.load(annotation_file)

    # ------------------------------------------------------------------ #
    #  Loading
    # ------------------------------------------------------------------ #

    def load(self, annotation_file):
        """Load a COCO JSON annotation file."""
        print(f"[COCO] Loading annotations from: {annotation_file}")
        with open(annotation_file, "r") as f:
            self.dataset = json.load(f)
        self._build_index()

    def _build_index(self):
        """Build lookup dictionaries for fast querying."""
        print("[COCO] Building index...")

        for img in self.dataset.get("images", []):
            self.imgs[img["id"]] = img

        for cat in self.dataset.get("categories", []):
            self.cats[cat["id"]] = cat

        for ann in self.dataset.get("annotations", []):
            self.anns[ann["id"]] = ann
            self.imgToAnns[ann["image_id"]].append(ann)
            self.catToImgs[ann["category_id"]].append(ann["image_id"])

        print(
            f"[COCO] Index built — "
            f"{len(self.imgs)} images, "
            f"{len(self.cats)} categories, "
            f"{len(self.anns)} annotations"
        )

    # ------------------------------------------------------------------ #
    #  Getters
    # ------------------------------------------------------------------ #

    def getImgIds(self, imgIds=None, catIds=None):
        """Return image IDs, optionally filtered by image/category IDs."""
        ids = set(self.imgs.keys())
        if imgIds:
            ids &= set(imgIds)
        if catIds:
            filtered = set()
            for cat_id in catIds:
                filtered |= set(self.catToImgs[cat_id])
            ids &= filtered
        return sorted(ids)

    def getCatIds(self, catNms=None, supNms=None, catIds=None):
        """Return category IDs filtered by name, supercategory, or ID."""
        cats = list(self.cats.values())
        if catNms:
            cats = [c for c in cats if c["name"] in catNms]
        if supNms:
            cats = [c for c in cats if c.get("supercategory") in supNms]
        if catIds:
            cats = [c for c in cats if c["id"] in catIds]
        return [c["id"] for c in cats]

    def getAnnIds(self, imgIds=None, catIds=None, areaRng=None):
        """Return annotation IDs filtered by image, category, or area range."""
        anns = list(self.anns.values())
        if imgIds:
            anns = [a for a in anns if a["image_id"] in imgIds]
        if catIds:
            anns = [a for a in anns if a["category_id"] in catIds]
        if areaRng:
            lo, hi = areaRng
            anns = [a for a in anns if lo <= a.get("area", 0) <= hi]
        return [a["id"] for a in anns]

    def loadImgs(self, ids):
        """Return image dicts for given IDs."""
        return [self.imgs[i] for i in ids if i in self.imgs]

    def loadAnns(self, ids):
        """Return annotation dicts for given IDs."""
        return [self.anns[i] for i in ids if i in self.anns]

    def loadCats(self, ids):
        """Return category dicts for given IDs."""
        return [self.cats[i] for i in ids if i in self.cats]

    # ------------------------------------------------------------------ #
    #  Summary
    # ------------------------------------------------------------------ #

    def info(self):
        """Print dataset info."""
        meta = self.dataset.get("info", {})
        print("=== COCO Dataset Info ===")
        for k, v in meta.items():
            print(f"  {k}: {v}")
        print(f"  Images     : {len(self.imgs)}")
        print(f"  Categories : {len(self.cats)}")
        print(f"  Annotations: {len(self.anns)}")

    def category_names(self):
        """Return a list of all category names."""
        return [c["name"] for c in self.cats.values()]

    # ------------------------------------------------------------------ #
    #  Export
    # ------------------------------------------------------------------ #

    def save(self, output_path):
        """Save the current dataset back to a COCO JSON file."""
        with open(output_path, "w") as f:
            json.dump(self.dataset, f, indent=2)
        print(f"[COCO] Saved to {output_path}")


# ------------------------------------------------------------------ #
#  YOLO <-> COCO Converters
# ------------------------------------------------------------------ #

def yolo_to_coco(images_dir, labels_dir, category_names, output_json):
    """
    Convert a YOLO-format dataset to COCO JSON.

    Args:
        images_dir    : folder containing images (.jpg/.png)
        labels_dir    : folder containing YOLO .txt label files
        category_names: list of class names, index = class id
        output_json   : path to write the output annotations.json
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)

    dataset = {
        "info": {"description": "Converted from YOLO format"},
        "images": [],
        "annotations": [],
        "categories": []
    }

    # Build categories
    for idx, name in enumerate(category_names):
        dataset["categories"].append({
            "id": idx,
            "name": name,
            "supercategory": "none"
        })

    ann_id = 0
    img_id = 0

    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}

    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() not in image_exts:
            continue

        # Try to get image dimensions
        try:
            from PIL import Image as PILImage
            with PILImage.open(img_path) as im:
                w, h = im.size
        except ImportError:
            print("[WARN] Pillow not installed; using placeholder size 640x640")
            w, h = 640, 640

        dataset["images"].append({
            "id": img_id,
            "file_name": img_path.name,
            "width": w,
            "height": h
        })

        label_path = labels_dir / (img_path.stem + ".txt")
        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls, xc, yc, bw, bh = map(float, parts)
                    # Convert YOLO normalized -> COCO absolute pixel
                    x1 = (xc - bw / 2) * w
                    y1 = (yc - bh / 2) * h
                    bw_abs = bw * w
                    bh_abs = bh * h
                    dataset["annotations"].append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": int(cls),
                        "bbox": [round(x1, 2), round(y1, 2),
                                 round(bw_abs, 2), round(bh_abs, 2)],
                        "area": round(bw_abs * bh_abs, 2),
                        "iscrowd": 0
                    })
                    ann_id += 1

        img_id += 1

    with open(output_json, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"[YOLO→COCO] Done. {img_id} images, {ann_id} annotations → {output_json}")
    return dataset


def coco_to_yolo(annotation_json, output_labels_dir):
    """
    Convert a COCO JSON to YOLO-format .txt label files.

    Args:
        annotation_json   : path to COCO annotations.json
        output_labels_dir : folder where per-image .txt files will be written
    """
    output_labels_dir = Path(output_labels_dir)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    with open(annotation_json) as f:
        dataset = json.load(f)

    # Build lookup: img_id -> image info
    img_map = {img["id"]: img for img in dataset["images"]}

    # Group annotations by image
    ann_by_img = defaultdict(list)
    for ann in dataset["annotations"]:
        ann_by_img[ann["image_id"]].append(ann)

    for img_id, anns in ann_by_img.items():
        img = img_map[img_id]
        w, h = img["width"], img["height"]
        stem = Path(img["file_name"]).stem
        out_path = output_labels_dir / (stem + ".txt")

        with open(out_path, "w") as f:
            for ann in anns:
                x1, y1, bw, bh = ann["bbox"]
                # Convert COCO absolute -> YOLO normalized
                xc = (x1 + bw / 2) / w
                yc = (y1 + bh / 2) / h
                bw_n = bw / w
                bh_n = bh / h
                f.write(f"{ann['category_id']} {xc:.6f} {yc:.6f} {bw_n:.6f} {bh_n:.6f}\n")

    print(f"[COCO→YOLO] Labels written to: {output_labels_dir}")


# ------------------------------------------------------------------ #
#  Quick demo (run: python coco.py)
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import tempfile

    print("=== COCO.py Demo ===\n")

    # Create a tiny sample COCO JSON in memory
    sample = {
        "info": {"description": "Sample Dataset", "version": "1.0"},
        "images": [
            {"id": 1, "file_name": "dog.jpg",  "width": 640, "height": 480},
            {"id": 2, "file_name": "cat.jpg",  "width": 320, "height": 240},
        ],
        "categories": [
            {"id": 0, "name": "dog", "supercategory": "animal"},
            {"id": 1, "name": "cat", "supercategory": "animal"},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 0, "bbox": [50, 30, 200, 150], "area": 30000, "iscrowd": 0},
            {"id": 2, "image_id": 2, "category_id": 1, "bbox": [10, 10, 100, 80],  "area": 8000,  "iscrowd": 0},
        ]
    }

    # Write to a temp file
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
        json.dump(sample, tmp)
        tmp_path = tmp.name

    # Load with COCO class
    coco = COCO(tmp_path)
    coco.info()

    print("\n--- Category names ---")
    print(coco.category_names())

    print("\n--- All image IDs ---")
    print(coco.getImgIds())

    print("\n--- Image IDs with 'dog' ---")
    cat_ids = coco.getCatIds(catNms=["dog"])
    print(coco.getImgIds(catIds=cat_ids))

    print("\n--- Annotations for image 1 ---")
    ann_ids = coco.getAnnIds(imgIds=[1])
    print(coco.loadAnns(ann_ids))

    # Demo COCO -> YOLO
    print("\n--- COCO → YOLO conversion ---")
    with tempfile.TemporaryDirectory() as tmpdir:
        coco_to_yolo(tmp_path, tmpdir)
        for f in Path(tmpdir).iterdir():
            print(f"  {f.name}:", f.read_text().strip())

    os.unlink(tmp_path)
    print("\nDone!")
