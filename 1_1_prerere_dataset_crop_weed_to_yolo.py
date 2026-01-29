# Libraries
from PIL import Image
import numpy as np
import cv2
import xml.etree.ElementTree as ET
import os
import csv
import glob
import random


def open_image_local(path_to_image):
    """Open an image from a local path."""
    image = Image.open(path_to_image).convert("RGB")
    image_array = np.array(image)
    return image_array


def load_xml_data(xml_path):
    """Load XML annotations."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    true_boxes = []
    true_classes = []

    for obj in root.findall("object"):
        class_name = obj.find("name").text
        bndbox = obj.find("bndbox")
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)
        
        true_boxes.append((xmin, ymin, xmax, ymax))
        true_classes.append(class_name)

    return true_boxes, true_classes


def adaptive_refinement_v2(image_bgr):
    """Адаптивный метод выделения контура на основе HSV-фильтрации."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    h_ranges_to_try = [(22 + i, 85) for i in range(14)]
    refinement_history = []

    best_score = -1
    best_mask = None
    best_contour = None

    flag_area = True

    for h_i, (h_lower, h_upper) in enumerate(h_ranges_to_try):
        lower = np.array([h_lower, 0, 0])
        upper = np.array([h_upper, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)

        # Морфологическая очистка
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        main_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(main_contour)
        if area < 50:
            continue

        # === Форма (компактность, соотношение) ===
        h, w = mask.shape[:2]
        x_min, y_min, w_contour, h_contour = cv2.boundingRect(main_contour)

        dist_left = x_min
        dist_top = y_min
        dist_right = w - (x_min + w_contour)
        dist_bottom = h - (y_min + h_contour)

        dist_summ = dist_left + dist_top + dist_right + dist_bottom
        shape_score = dist_summ / (h + w)

        if flag_area:
            k_erea = 1
            flag_area = False
        else:
            k_erea = area / old_area

        score = 0.70 * k_erea + 0.30 * shape_score
        old_area = area

        refinement_history.append({
            'h_lower': h_lower,
            'h_upper': h_upper,
            'area': area,
            'score': float(score),
            'mask': mask.copy(),
            'contour': main_contour.copy(),
            'best_edge': False
        })

        if score > best_score:
            best_score = score
            best_mask = mask.copy()
            best_contour = main_contour.copy()

    return refinement_history


def find_best_index_by_score_delta(scores):
    """Выбирает лучший контур по принципу 'первая значительная потеря качества'."""
    scores = np.array(scores, dtype=float)
    if len(scores) < 2:
        return 0

    score_min = np.min(scores)
    score_max = np.max(scores)
    score_range = score_max - score_min
    threshold_drop = score_range * 0.2

    best_idx = 0
    for i in range(1, len(scores)):
        delta = scores[i] - scores[i - 1]
        if delta < -threshold_drop:
            print(f"  Обнаружено падение на шаге {i}: Δ={delta:.3f}, выбор индекса {i-1}")
            best_idx = i - 1
            break
        best_idx = i

    return best_idx


def convert_mask_to_yolo_format(mask, image_width, image_height):
    """Конвертирует бинарную маску в формат YOLO для сегментации."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return []
    
    main_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.002 * cv2.arcLength(main_contour, True)
    approx_contour = cv2.approxPolyDP(main_contour, epsilon, True)
    
    yolo_polygon = []
    for point in approx_contour:
        x = point[0][0] / image_width
        y = point[0][1] / image_height
        yolo_polygon.extend([x, y])
    
    return yolo_polygon


def create_yolo_dataset_structure(base_path):
    """Создает структуру папок для датасета YOLO с тремя разделами."""
    directories = [
        "images/train",
        "labels/train",
        "images/val",
        "labels/val",
        "images/test",
        "labels/test",
    ]
    
    for directory in directories:
        os.makedirs(os.path.join(base_path, directory), exist_ok=True)
    
    return os.path.join(base_path, "data.yaml")


def create_yolo_config_file(yaml_path, class_names):
    """Создает файл конфигурации data.yaml для YOLO."""
    config_content = f"""# YOLO segmentation dataset configuration
path: {os.path.dirname(yaml_path)}
train: images/train
val: images/val
test: images/test

nc: {len(class_names)}
names: {class_names}
"""
    
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    return yaml_path


def save_yolo_annotation(label_path, class_id, polygon):
    """Сохраняет аннотацию в формате YOLO."""
    with open(label_path, 'w', encoding='utf-8') as f:
        if polygon:
            line = f"{class_id} " + " ".join([f"{coord:.6f}" for coord in polygon])
            f.write(line + "\n")


def visualize_all_segmentation(image, masks_data, save_path):
    """Визуализирует все сегментации на одном изображении."""
    vis_image = image.copy()
    class_colors = {
        "crop": (0, 255, 0),
        "weed": (255, 0, 0),
    }
    
    for i, mask_data in enumerate(masks_data):
        class_name = mask_data['class_name']
        color = class_colors.get(class_name, (0, 255, 255))
        
        if mask_data['polygon']:
            h, w = image.shape[:2]
            points = []
            for j in range(0, len(mask_data['polygon']), 2):
                x = int(mask_data['polygon'][j] * w)
                y = int(mask_data['polygon'][j + 1] * h)
                points.append([x, y])
            
            if points:
                points = np.array(points, dtype=np.int32)
                cv2.polylines(vis_image, [points], True, color, 2)
                
                if len(points) > 0:
                    centroid = np.mean(points, axis=0).astype(int)
                    cv2.putText(vis_image, f'{class_name}_{mask_data["crop_id"]}', 
                              (centroid[0], centroid[1]), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    legend_y = 30
    for class_name, color in class_colors.items():
        cv2.putText(vis_image, f'{class_name}', (10, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        legend_y += 20
    
    cv2.imwrite(save_path, vis_image)


def process_crop_class(crop_image, class_name):
    """Обрабатывает изображение для создания сегментации."""
    print(f"  Обработка класса {class_name}...")
    
    history = adaptive_refinement_v2(crop_image)
    
    if not history:
        print(f"  Не найдено контуров для {class_name}")
        return None
    
    scores = [item['score'] for item in history]
    best_index = find_best_index_by_score_delta(scores)
    
    return history[best_index]['mask']


def split_dataset(image_paths, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """Разделяет датасет на train, val и test."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001, "Сумма долей должна быть равна 1"
    
    random.seed(seed)
    shuffled_paths = image_paths.copy()
    random.shuffle(shuffled_paths)
    
    total = len(shuffled_paths)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    train_set = shuffled_paths[:train_end]
    val_set = shuffled_paths[train_end:val_end]
    test_set = shuffled_paths[val_end:]
    
    return train_set, val_set, test_set


def main():
    # path_images = "train_images/"
    path_images = "train_images_edit/"
    output_root = "yolo_segmentation_dataset/"
    
    class_names = ["crop", "weed"]
    class_to_id = {name: i for i, name in enumerate(class_names)}
    
    yaml_path = create_yolo_dataset_structure(output_root)
    create_yolo_config_file(yaml_path, class_names)
    
    csv_path = os.path.join(output_root, "dataset_info.csv")
    csv_header = ["image_name", "object_id", "class_name", "best_contour_id", "polygon_points", "split"]
    
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)
    
    vis_dir = os.path.join(output_root, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    image_paths = sorted(glob.glob(os.path.join(path_images, "*.jpg")))
    
    # Разделение на train, val, test (70%, 15%, 15%)
    train_paths, val_paths, test_paths = split_dataset(
        image_paths, 
        train_ratio=0.7, 
        val_ratio=0.15, 
        test_ratio=0.15
    )
    
    print(f"Всего изображений: {len(image_paths)}")
    print(f"Train: {len(train_paths)} ({len(train_paths)/len(image_paths)*100:.1f}%)")
    print(f"Val: {len(val_paths)} ({len(val_paths)/len(image_paths)*100:.1f}%)")
    print(f"Test: {len(test_paths)} ({len(test_paths)/len(image_paths)*100:.1f}%)")
    
    # Создаем словарь для быстрого определения раздела
    split_mapping = {}
    for path in train_paths:
        split_mapping[os.path.basename(path)] = "train"
    for path in val_paths:
        split_mapping[os.path.basename(path)] = "val"
    for path in test_paths:
        split_mapping[os.path.basename(path)] = "test"
    
    processed_count = 0
    
    for idx, image_path in enumerate(image_paths):
        image_name = os.path.basename(image_path)
        xml_name = os.path.splitext(image_name)[0] + ".xml"
        xml_path = os.path.join(path_images, xml_name)
        
        if not os.path.exists(xml_path):
            print(f"Пропущено {image_name} (отсутствует XML-аннотация)")
            continue
        
        # Определяем раздел из словаря
        split_folder = split_mapping.get(image_name, "train")
        
        image = open_image_local(image_path)
        true_boxes, true_classes = load_xml_data(xml_path)
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_height, image_width = image_bgr.shape[:2]
        
        yolo_image_path = os.path.join(output_root, "images", split_folder, image_name)
        cv2.imwrite(yolo_image_path, image_bgr)
        
        yolo_label_path = os.path.join(output_root, "labels", split_folder, 
                                     os.path.splitext(image_name)[0] + ".txt")
        
        all_masks_data = []
        
        with open(yolo_label_path, 'w', encoding='utf-8') as label_file:
            for obj_id, (box, class_name) in enumerate(zip(true_boxes, true_classes)):
                if class_name not in class_to_id:
                    print(f"Пропущен неизвестный класс: {class_name}")
                    continue
                    
                xmin, ymin, xmax, ymax = box
                class_id = class_to_id[class_name]
                
                if class_name in ['crop', 'weed']:
                    obj_image = image_bgr[ymin:ymax, xmin:xmax]
                    
                    best_mask = process_crop_class(obj_image, class_name)
                    
                    if best_mask is None:
                        continue
                    
                    full_mask = np.zeros((image_height, image_width), dtype=np.uint8)
                    full_mask[ymin:ymax, xmin:xmax] = best_mask
                    
                    yolo_polygon = convert_mask_to_yolo_format(full_mask, image_width, image_height)
                    
                    if yolo_polygon:
                        line = f"{class_id} " + " ".join([f"{coord:.6f}" for coord in yolo_polygon])
                        label_file.write(line + "\n")
                        
                        mask_data = {
                            'mask': full_mask,
                            'polygon': yolo_polygon,
                            'crop_id': obj_id,
                            'class_name': class_name
                        }
                        all_masks_data.append(mask_data)
                        
                        with open(csv_path, "a", newline="", encoding="utf-8") as f:
                            writer = csv.writer(f)
                            writer.writerow([
                                image_name, 
                                obj_id, 
                                class_name,
                                0,
                                len(yolo_polygon) // 2,
                                split_folder
                            ])
        
        if all_masks_data:
            vis_filename = f"{os.path.splitext(image_name)[0]}_all_segmentation.jpg"
            vis_path = os.path.join(vis_dir, vis_filename)
            visualize_all_segmentation(image_bgr, all_masks_data, vis_path)
        
        processed_count += 1
        if processed_count % 10 == 0:
            print(f"Прогресс: {processed_count}/{len(image_paths)} изображений обработано")
    
    print("\nПодготовка датасета для YOLO сегментации завершена")
    print(f"Датасет: {output_root}")
    print(f"Конфигурация YOLO: {yaml_path}")
    print(f"Распределение: {len(train_paths)} train, {len(val_paths)} val, {len(test_paths)} test")
    print(f"Обработано изображений: {processed_count}")
    print(f"Классы: {class_names}")


if __name__ == "__main__":
    main()