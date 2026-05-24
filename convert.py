from coco import yolo_to_coco

yolo_to_coco(
    images_dir     = "Images",
    labels_dir     = "Labels",
    category_names = ["with_mask", "without_mask", "mask_weared_incorrect"],
    output_json    = "annotations.json"
)