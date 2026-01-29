import os

# ===== НАСТРОЙКИ =====
VIS_DIR = "yolo_segmentation_dataset_2/visualizations"
IMAGES_DIR = "train_images_edit_2"
IMAGE_EXT = ".jpg"
# ====================


def main():
    # 1. Собираем stem'ы из visualizations
    vis_stems = set()

    for fname in os.listdir(VIS_DIR):
        if fname.endswith("_all_segmentation.jpg"):
            stem = fname.replace("_all_segmentation.jpg", "")
            vis_stems.add(stem)

    print(f"Найдено визуализаций: {len(vis_stems)}")

    # 2. Проверяем изображения
    removed = 0
    total = 0

    for fname in os.listdir(IMAGES_DIR):
        if not fname.endswith(IMAGE_EXT):
            continue

        total += 1
        stem = os.path.splitext(fname)[0]

        if stem not in vis_stems:
            img_path = os.path.join(IMAGES_DIR, fname)
            os.remove(img_path)
            removed += 1
            print(f"🗑 Удалено: {fname}")

    print("\n====== РЕЗУЛЬТАТ ======")
    print(f"Всего изображений: {total}")
    print(f"Удалено без visualization: {removed}")
    print(f"Осталось: {total - removed}")
    print("Готово ✅")


if __name__ == "__main__":
    main()
