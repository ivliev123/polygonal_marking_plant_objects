# Libraries
from PIL import Image
import numpy as np
import cv2
import xml.etree.ElementTree as ET
import os
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import json
import math
import csv
import glob


# Open an image from a local path
def open_image_local(path_to_image):
    image = Image.open(path_to_image).convert("RGB")
    image_array = np.array(image)
    return image_array

# Load XML annotations
def load_xml_data(xml_path):
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
    """
    Гибридный метод выделения контура.
    Комбинирует:
    - Цветовую однородность (HSV)
    - Градиентную четкость (Canny)
    - Текстурную гладкость (локальные контрасты)
    - Форму (компактность и площадь)
    """

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    h_ranges_to_try = [(22 + i, 85) for i in range(14)]
    refinement_history = []

    # Canny — карта градиента
    edges = cv2.Canny(gray, 50, 150)
    edges_norm = edges / 255.0  # для подсчётов

    best_score = -1
    best_mask = None
    best_contour = None

    flag_area = True

    for h_i, (h_lower, h_upper) in enumerate(h_ranges_to_try):
        lower = np.array([h_lower, 5, 5])
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
        if area < 50:  # фильтр шумных контуров
            continue

        # === 1. Цветовая однородность ===
        mask_filled = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(mask_filled, [main_contour], -1, 255, -1)
        mean_val, std_val = cv2.meanStdDev(hsv, mask=mask_filled)
        color_std = np.mean(std_val)
        color_score = 1.0 / (color_std + 1e-3)

        # === 2. Градиентная насыщенность по границе ===
        contour_mask = np.zeros_like(edges_norm)
        cv2.drawContours(contour_mask, [main_contour], -1, 1, 1)
        edge_strength = np.mean(edges_norm[contour_mask > 0])
        edge_score = edge_strength ** 2  # усиливаем влияние сильных границ

        # === 3. Текстурная гладкость ===
        std_texture = np.std(gray[mask_filled > 0])
        texture_score = 1.0 / (std_texture + 1e-3)

        # === 4. Форма (компактность, соотношение) ===
        h, w = mask.shape[:2]
        x_min, y_min, w_contour, h_contour = cv2.boundingRect(main_contour)

        dist_left   = x_min
        dist_top    = y_min
        dist_right  = w - (x_min + w_contour)
        dist_bottom = h - (y_min + h_contour)

        dist_summ = dist_left + dist_top + dist_right + dist_bottom
        shape_score = dist_summ / (h + w)

        if flag_area:
            k_erea = 1
            flag_area = False
        else:
            k_erea = area / old_area

        score = (
            0.70 * k_erea +
            0.00 * color_score +
            0.00 * edge_score +
            0.00 * texture_score +
            0.30 * shape_score
        )

        old_area = area

        refinement_history.append({
            'h_lower': h_lower,
            'h_upper': h_upper,
            'area': area,
            'color_std': float(color_std),
            'edge_strength': float(edge_strength),
            'texture_std': float(std_texture),
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
    """
    Анализирует последовательность оценок score и выбирает лучший контур
    по принципу 'первая значительная потеря качества'.
    """
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
            print(f"⚠️ Резкое падение на шаге {i}: Δ={delta:.3f} → выбираем индекс {i-1}")
            best_idx = i - 1
            break

        best_idx = i

    return best_idx

def convert_mask_to_yolo_format(mask, image_width, image_height):
    """
    Конвертирует бинарную маску в формат YOLO для сегментации
    Возвращает список полигонов в нормализованных координатах
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return []
    
    # Берем самый большой контур
    main_contour = max(contours, key=cv2.contourArea)
    
    # Упрощаем контур (уменьшаем количество точек)
    epsilon = 0.002 * cv2.arcLength(main_contour, True)
    approx_contour = cv2.approxPolyDP(main_contour, epsilon, True)
    
    # Нормализуем координаты для YOLO
    yolo_polygon = []
    for point in approx_contour:
        x = point[0][0] / image_width
        y = point[0][1] / image_height
        yolo_polygon.extend([x, y])
    
    return yolo_polygon

def create_yolo_dataset_structure(base_path):
    """
    Создает структуру папок для датасета YOLO
    """
    directories = [
        "images/train",
        "labels/train",
        "images/val", 
        "labels/val",
        # "masks"
    ]
    
    for directory in directories:
        os.makedirs(os.path.join(base_path, directory), exist_ok=True)
    
    return os.path.join(base_path, "data.yaml")

def create_yolo_config_file(yaml_path, class_names):
    """
    Создает файл конфигурации data.yaml для YOLO
    """
    config_content = f"""# YOLO segmentation dataset configuration
path: {os.path.dirname(yaml_path)}
train: images/train
val: images/val

nc: {len(class_names)}
names: {class_names}
"""
    
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    return yaml_path

def save_yolo_annotation(label_path, class_id, polygon):
    """
    Сохраняет аннотацию в формате YOLO
    """
    with open(label_path, 'w', encoding='utf-8') as f:
        if polygon:  # Если есть полигон для сегментации
            line = f"{class_id} " + " ".join([f"{coord:.6f}" for coord in polygon])
            f.write(line + "\n")

def visualize_all_segmentation(image, masks_data, save_path):
    """
    Визуализирует ВСЕ сегментации на одном изображении
    masks_data - список словарей с информацией о масках и полигонах
    """
    # Создаем копию изображения для визуализации
    vis_image = image.copy()
    
    # Цвета для разных классов
    class_colors = {
        "crop": (0, 255, 0),    # Зеленый для crop
        "weed": (255, 0, 0),    # Красный для weed
    }
    
    # Рисуем все маски и контуры
    for i, mask_data in enumerate(masks_data):
        class_name = mask_data['class_name']
        color = class_colors.get(class_name, (0, 255, 255))  # Желтый по умолчанию
        
        # Рисуем контур полигона
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
                
                # Добавляем метку класса
                if len(points) > 0:
                    centroid = np.mean(points, axis=0).astype(int)
                    cv2.putText(vis_image, f'{class_name}_{mask_data["crop_id"]}', 
                              (centroid[0], centroid[1]), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Добавляем легенду
    legend_y = 30
    for class_name, color in class_colors.items():
        cv2.putText(vis_image, f'{class_name}', (10, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        legend_y += 20
    
    # Сохраняем визуализацию
    cv2.imwrite(save_path, vis_image)

def process_crop_class(crop_image, class_name):
    """
    Обрабатывает изображение crop для создания сегментации
    """
    print(f"🔍 Обработка класса {class_name}...")
    
    # Для crop используем адаптивный HSV-анализ
    if class_name == 'crop':
        history = adaptive_refinement_v2(crop_image)
        
        if not history:
            print(f"❌ Нет контуров для {class_name}")
            return None
        
        scores = [item['score'] for item in history]
        best_index = find_best_index_by_score_delta(scores)
        history[best_index]['best_edge'] = True
        
        return history[best_index]['mask']
    
    # Для weed можно использовать другую логику или ту же самую
    elif class_name == 'weed':
        history = adaptive_refinement_v2(crop_image)
        
        if not history:
            print(f"❌ Нет контуров для {class_name}")
            return None
        
        scores = [item['score'] for item in history]
        best_index = find_best_index_by_score_delta(scores)
        history[best_index]['best_edge'] = True
        
        return history[best_index]['mask']
    
    return None

def main():
    # === ПУТИ ===
    path_images = "train_images/"
    output_root = "yolo_segmentation_dataset_2/"
    
    # Классы для сегментации - теперь включаем и crop и weed
    class_names = ["crop", "weed"]
    class_to_id = {name: i for i, name in enumerate(class_names)}
    
    # Создаем структуру папок YOLO
    yaml_path = create_yolo_dataset_structure(output_root)
    create_yolo_config_file(yaml_path, class_names)
    
    # CSV для метаданных (опционально)
    csv_path = os.path.join(output_root, "dataset_info.csv")
    csv_header = ["image_name", "object_id", "class_name", "best_contour_id", "polygon_points"]
    
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)
    
    # Папка для визуализаций
    vis_dir = os.path.join(output_root, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Список изображений для обработки
    image_paths = sorted(glob.glob(os.path.join(path_images, "*.jpg")))
    # image_paths = image_paths[:500]  # Ограничиваем количество для тестирования
    
    train_ratio = 0.8  # 80% train, 20% validation
    train_count = int(len(image_paths) * train_ratio)
    
    for idx, image_path in enumerate(image_paths):
        image_name = os.path.basename(image_path)
        xml_name = os.path.splitext(image_name)[0] + ".xml"
        xml_path = os.path.join(path_images, xml_name)
        
        if not os.path.exists(xml_path):
            print(f"⚠️ Пропущен {image_name} (нет XML-аннотации)")
            continue
        
        # Определяем train/val split
        split_folder = "train" if idx < train_count else "val"
        
        # === 1. Открываем изображение и XML ===
        image = open_image_local(image_path)
        true_boxes, true_classes = load_xml_data(xml_path)
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_height, image_width = image_bgr.shape[:2]
        
        # Сохраняем оригинальное изображение в папку YOLO
        yolo_image_path = os.path.join(output_root, "images", split_folder, image_name)
        cv2.imwrite(yolo_image_path, image_bgr)
        
        # Файл для аннотаций YOLO
        yolo_label_path = os.path.join(output_root, "labels", split_folder, 
                                     os.path.splitext(image_name)[0] + ".txt")
        
        # Собираем все маски для визуализации
        all_masks_data = []
        
        # Открываем файл для записи всех аннотаций этого изображения
        with open(yolo_label_path, 'w', encoding='utf-8') as label_file:
            for obj_id, (box, class_name) in enumerate(zip(true_boxes, true_classes)):
                # Пропускаем классы, которых нет в нашем списке
                if class_name not in class_to_id:
                    print(f"⚠️ Пропущен неизвестный класс: {class_name}")
                    continue
                    
                xmin, ymin, xmax, ymax = box
                class_id = class_to_id[class_name]
                
                # Обрабатываем как crop, так и weed
                if class_name in ['crop', 'weed']:
                    obj_image = image_bgr[ymin:ymax, xmin:xmax]
                    obj_height, obj_width = obj_image.shape[:2]
                    
                    print(f"🔍 [{image_name}] Обработка {class_name} {obj_id}...")
                    
                    # === 2. Запускаем обработку в зависимости от класса ===
                    best_mask = process_crop_class(obj_image, class_name)
                    
                    if best_mask is None:
                        print(f"❌ Не удалось создать маску для {image_name} {class_name} {obj_id}")
                        continue
                    
                    # === 3. Конвертируем маску в полигон YOLO ===
                    # Создаем полную маску для всего изображения
                    full_mask = np.zeros((image_height, image_width), dtype=np.uint8)
                    full_mask[ymin:ymax, xmin:xmax] = best_mask
                    
                    # Конвертируем в формат YOLO
                    yolo_polygon = convert_mask_to_yolo_format(full_mask, image_width, image_height)
                    
                    if yolo_polygon:
                        # Записываем аннотацию в файл
                        line = f"{class_id} " + " ".join([f"{coord:.6f}" for coord in yolo_polygon])
                        label_file.write(line + "\n")
                        
                        # # === 4. Сохраняем маску для отладки ===
                        # mask_filename = f"{os.path.splitext(image_name)[0]}_{class_name}{obj_id}_mask.png"
                        # mask_path = os.path.join(output_root, "masks", mask_filename)
                        # cv2.imwrite(mask_path, full_mask * 255)
                        
                        # Сохраняем данные для визуализации
                        mask_data = {
                            'mask': full_mask,
                            'polygon': yolo_polygon,
                            'crop_id': obj_id,
                            'class_name': class_name
                        }
                        all_masks_data.append(mask_data)
                        
                        # === 5. Записываем метаданные в CSV ===
                        with open(csv_path, "a", newline="", encoding="utf-8") as f:
                            writer = csv.writer(f)
                            writer.writerow([
                                image_name, 
                                obj_id, 
                                class_name,
                                0,  # best_contour_id (можно адаптировать при необходимости)
                                len(yolo_polygon) // 2  # количество точек в полигоне
                            ])
                        
                        print(f"✅ {image_name} {class_name} {obj_id} — сегментация сохранена")
        
        # === 6. Визуализация ВСЕХ сегментаций на одном изображении ===
        if all_masks_data:
            vis_filename = f"{os.path.splitext(image_name)[0]}_all_segmentation.jpg"
            vis_path = os.path.join(vis_dir, vis_filename)
            visualize_all_segmentation(image_bgr, all_masks_data, vis_path)
            print(f"🖼️ Визуализация всех контуров сохранена: {vis_filename}")
        
        print(f"✅ {image_name} обработано и сохранено в {split_folder}")
        print(f"   Найдено объектов: {len(all_masks_data)}")
    
    print("\n🎯 Подготовка обучающей выборки для YOLO сегментации завершена!")
    print(f"📁 Датасет: {output_root}")
    print(f"📄 Конфигурация YOLO: {yaml_path}")
    print(f"📊 Распределение: {train_count} train, {len(image_paths) - train_count} val")
    print(f"👁️ Визуализации: {vis_dir}")
    print(f"📋 Метаданные: {csv_path}")
    print(f"🎯 Классы: {class_names}")

if __name__ == "__main__":
    main()