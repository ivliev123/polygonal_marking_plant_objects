import os
import cv2
import csv
import shutil

# ========== НАСТРОЙКИ ==========
DATASET_ROOT = "yolo_segmentation_dataset_2"
VIS_DIR = os.path.join(DATASET_ROOT, "visualizations")
IMAGES_DIR = os.path.join(DATASET_ROOT, "images")
LABELS_DIR = os.path.join(DATASET_ROOT, "labels")
CSV_PATH = os.path.join(DATASET_ROOT, "dataset_info.csv")

WINDOW_NAME = "Segmentation Review"
# ===============================


def remove_sample(image_stem):
    """Удаляет sample из images, labels, visualizations и CSV"""
    print(f"🗑 Удаление sample: {image_stem}")

    # 1. visualization
    vis_path = os.path.join(VIS_DIR, f"{image_stem}_all_segmentation.jpg")
    if os.path.exists(vis_path):
        os.remove(vis_path)

    # 2. images / labels (train, val, test)
    for split in ["train", "val", "test"]:
        img_path = os.path.join(IMAGES_DIR, split, f"{image_stem}.jpg")
        lbl_path = os.path.join(LABELS_DIR, split, f"{image_stem}.txt")

        if os.path.exists(img_path):
            os.remove(img_path)

        if os.path.exists(lbl_path):
            os.remove(lbl_path)

    # 3. CSV — фильтрация
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, newline='', encoding="utf-8") as f:
            rows = list(csv.reader(f))

        header = rows[0]
        filtered = [header] + [
            r for r in rows[1:] if r[0] != f"{image_stem}.jpg"
        ]

        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(filtered)


def main():
    files = sorted([
        f for f in os.listdir(VIS_DIR)
        if f.endswith("_all_segmentation.jpg")
    ])

    print(f"Найдено {len(files)} visualization файлов")

    for fname in files:
        image_stem = fname.replace("_all_segmentation.jpg", "")
        path = os.path.join(VIS_DIR, fname)

        img = cv2.imread(path)
        if img is None:
            continue

        cv2.imshow(WINDOW_NAME, img)

        print(f"\n▶ {fname}")
        print("  [k / SPACE] — оставить")
        print("  [d]         — удалить")
        print("  [q]         — выход")

        key = cv2.waitKey(0) & 0xFF

        if key in [ord('k'), 32]:
            continue

        elif key == ord('d'):
            remove_sample(image_stem)

        elif key == ord('q'):
            break

    cv2.destroyAllWindows()
    print("Готово ✅")


if __name__ == "__main__":
    main()
