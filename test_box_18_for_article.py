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

def visualize_hsv_distribution(crop_image, output_dir, image_name, class_name, obj_id):
    """
    Визуализация распределения оттенков в HSV-пространстве (Рисунок 3.4.2)
    """
    hsv = cv2.cvtColor(crop_image, cv2.COLOR_BGR2HSV)
    h_channel = hsv[:, :, 0]
    
    plt.figure(figsize=(10, 6))
    plt.hist(h_channel.flatten(), bins=50, color='green', alpha=0.7, edgecolor='black')
    plt.title('Распределение оттенков зелёной массы растений в пространстве HSV')
    plt.xlabel('Значение H (оттенок)')
    plt.ylabel('Частота')
    plt.grid(True, alpha=0.3)
    
    output_path = os.path.join(output_dir, f"1_hsv_distribution.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path

def visualize_h_masks(crop_image, history, best_index, output_dir, image_name, class_name, obj_id):
    """
    Визуализация примеров масок при различных диапазонах H (Рисунок 3.4.3, 3.4.4, 3.4.6)
    """
    # Выбираем ключевые этапы для демонстрации
    key_indices = [0, len(history)//3, 2*len(history)//3, best_index, -1]
    key_indices = [i for i in key_indices if i < len(history)]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, idx in enumerate(key_indices):
        if i >= len(axes):
            break
            
        result = history[idx]
        
        # Создаем визуализацию маски с контуром
        mask_viz = cv2.cvtColor(result['mask'], cv2.COLOR_GRAY2BGR)
        contour_color = (0, 255, 0) if idx == best_index else (255, 0, 0)
        cv2.drawContours(mask_viz, [result['contour']], -1, contour_color, 2)
        
        axes[i].imshow(cv2.cvtColor(mask_viz, cv2.COLOR_BGR2RGB))
        title = f'H={result["h_lower"]}-{result["h_upper"]}'
        if idx == best_index:
            title += ' (ЛУЧШИЙ)'
        axes[i].set_title(title, fontsize=10)
        axes[i].axis('off')
    
    # Скрываем неиспользуемые subplots
    for i in range(len(key_indices), len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Примеры выделения маски при различных диапазонах H', fontsize=14)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f"2_h_masks_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path

def visualize_binary_mask(crop_image, best_mask_data, output_dir, image_name, class_name, obj_id):
    """
    Визуализация бинарной маски и морфологических операций (Рисунок 3.4.4, 3.4.5)
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Исходное изображение
    axes[0].imshow(cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Исходный фрагмент')
    axes[0].axis('off')
    
    # Бинарная маска
    axes[1].imshow(best_mask_data['mask'], cmap='gray')
    axes[1].set_title('Бинарная маска после цветовой сегментации')
    axes[1].axis('off')
    
    # Маска с контуром
    mask_with_contour = cv2.cvtColor(best_mask_data['mask'], cv2.COLOR_GRAY2BGR)
    cv2.drawContours(mask_with_contour, [best_mask_data['contour']], -1, (0, 255, 0), 2)
    axes[2].imshow(cv2.cvtColor(mask_with_contour, cv2.COLOR_BGR2RGB))
    axes[2].set_title('Маска с выделенным контуром')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f"3_binary_mask_processing.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path

def visualize_score_progression(history, best_index, output_dir, image_name, class_name, obj_id):
    """
    Визуализация изменения оценки качества сегментации (Рисунок 3.4.7)
    """
    scores = [item['score'] for item in history]
    h_ranges = [f"{item['h_lower']}-{item['h_upper']}" for item in history]
    areas = [item['area'] for item in history]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # График оценки
    ax1.plot(range(len(scores)), scores, 'b-', marker='o', linewidth=2, markersize=6)
    ax1.axvline(x=best_index, color='r', linestyle='--', linewidth=2, label=f'Лучший (индекс {best_index})')
    ax1.set_title('Изменение интегральной оценки качества сегментации')
    ax1.set_xlabel('Индекс диапазона H')
    ax1.set_ylabel('Score')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # График площади
    ax2.plot(range(len(areas)), areas, 'g-', marker='s', linewidth=2, markersize=6)
    ax2.axvline(x=best_index, color='r', linestyle='--', linewidth=2, label=f'Лучший (индекс {best_index})')
    ax2.set_title('Изменение площади контура')
    ax2.set_xlabel('Индекс диапазона H')
    ax2.set_ylabel('Площадь (пиксели)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f"4_score_progression.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path

def visualize_final_contour(image_bgr, crop_image, best_mask_data, box, output_dir, image_name, class_name, obj_id):
    """
    Визуализация финального контура на изображении (Рисунок 3.4.8, 3.4.9, 3.4.11)
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Контур на кропе
    contour_on_crop = crop_image.copy()
    cv2.drawContours(contour_on_crop, [best_mask_data['contour']], -1, (0, 255, 0), 3)
    axes[0].imshow(cv2.cvtColor(contour_on_crop, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Итоговый контур маски на фрагменте')
    axes[0].axis('off')
    
    # Контур на полном изображении
    xmin, ymin, xmax, ymax = box
    contour_on_full = image_bgr.copy()
    
    # Преобразуем координаты контура в глобальные
    global_contour = best_mask_data['contour'] + np.array([xmin, ymin])
    cv2.drawContours(contour_on_full, [global_contour], -1, (0, 255, 0), 3)
    
    # Рисуем bounding box
    cv2.rectangle(contour_on_full, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
    
    axes[1].imshow(cv2.cvtColor(contour_on_full, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Контур на исходном изображении')
    axes[1].axis('off')
    
    # Полигональная аппроксимация
    mask = best_mask_data['mask']
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        main_contour = max(contours, key=cv2.contourArea)
        
        # Упрощаем контур
        epsilon = 0.01 * cv2.arcLength(main_contour, True)
        approx_contour = cv2.approxPolyDP(main_contour, epsilon, True)
        
        # Визуализация
        poly_vis = crop_image.copy()
        cv2.drawContours(poly_vis, [main_contour], -1, (255, 0, 0), 2)  # Исходный
        cv2.drawContours(poly_vis, [approx_contour], -1, (0, 255, 0), 2)  # Упрощенный
        
        axes[2].imshow(cv2.cvtColor(poly_vis, cv2.COLOR_BGR2RGB))
        axes[2].set_title(f'Полигональная аппроксимация\n({len(approx_contour)} точек)')
        axes[2].axis('off')
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f"5_final_contour_result.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path

def process_single_image_by_index(image_index=0, path_images="test_images/", output_root="single_image_output/"):
    """
    Обрабатывает одно изображение по указанному индексу с раздельной визуализацией
    """
    # Создаем папку для выходных данных
    os.makedirs(output_root, exist_ok=True)
    
    # Получаем список изображений
    image_paths = sorted(glob.glob(os.path.join(path_images, "*.jpg")))
    
    if image_index >= len(image_paths):
        print(f"❌ Ошибка: индекс {image_index} превышает количество изображений ({len(image_paths)})")
        return
    
    image_path = image_paths[image_index]
    image_name = os.path.basename(image_path)
    xml_name = os.path.splitext(image_name)[0] + ".xml"
    xml_path = os.path.join(path_images, xml_name)
    
    print(f"🎯 Обработка изображения {image_index}: {image_name}")
    
    if not os.path.exists(xml_path):
        print(f"❌ Пропущен {image_name} (нет XML-аннотации)")
        return
    
    # Открываем изображение и XML
    image = open_image_local(image_path)
    true_boxes, true_classes = load_xml_data(xml_path)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image_height, image_width = image_bgr.shape[:2]
    
    # Основная папка для изображения
    image_main_dir = os.path.join(output_root, f"image_{image_index}_{os.path.splitext(image_name)[0]}")
    os.makedirs(image_main_dir, exist_ok=True)
    
    all_masks_data = []
    all_visualization_paths = []
    
    for obj_id, (box, class_name) in enumerate(zip(true_boxes, true_classes)):
        # Создаем отдельную папку для каждого объекта
        if class_name in ['crop', 'weed']:
            obj_dir = os.path.join(image_main_dir, f"{class_name}_{obj_id:02d}")
            os.makedirs(obj_dir, exist_ok=True)
            
            xmin, ymin, xmax, ymax = box
            crop_image = image_bgr[ymin:ymax, xmin:xmax]
            
            print(f"🔍 Обработка {class_name} {obj_id}...")
            
            # Сохраняем исходный кроп
            crop_path = os.path.join(obj_dir, "0_original_crop.jpg")
            cv2.imwrite(crop_path, crop_image)
            
            # Запускаем адаптивный HSV-анализ
            history = adaptive_refinement_v2(crop_image)
            
            if not history:
                print(f"❌ Нет контуров для {class_name} {obj_id}")
                continue
            
            # Находим лучший контур
            scores = [item['score'] for item in history]
            best_index = find_best_index_by_score_delta(scores)
            best_mask_data = history[best_index]
            
            # === РАЗДЕЛЬНАЯ ВИЗУАЛИЗАЦИЯ КАЖДОГО ЭТАПА ===
            visualization_paths = []
            
            # 1. Распределение HSV (Рисунок 3.4.2)
            hsv_path = visualize_hsv_distribution(crop_image, obj_dir, 
                                                os.path.splitext(image_name)[0], class_name, obj_id)
            visualization_paths.append(("1. HSV распределение", hsv_path))
            
            # 2. Маски при разных H (Рисунок 3.4.3, 3.4.4, 3.4.6)
            h_masks_path = visualize_h_masks(crop_image, history, best_index, obj_dir,
                                           os.path.splitext(image_name)[0], class_name, obj_id)
            visualization_paths.append(("2. Маски при разных H", h_masks_path))
            
            # 3. Бинарная маска (Рисунок 3.4.4, 3.4.5)
            binary_mask_path = visualize_binary_mask(crop_image, best_mask_data, obj_dir,
                                                   os.path.splitext(image_name)[0], class_name, obj_id)
            visualization_paths.append(("3. Бинарная маска", binary_mask_path))
            
            # 4. График оценки (Рисунок 3.4.7)
            score_path = visualize_score_progression(history, best_index, obj_dir,
                                                  os.path.splitext(image_name)[0], class_name, obj_id)
            visualization_paths.append(("4. График оценки", score_path))
            
            # 5. Финальный контур (Рисунок 3.4.8, 3.4.9, 3.4.11)
            final_contour_path = visualize_final_contour(image_bgr, crop_image, best_mask_data, box,
                                                       obj_dir, os.path.splitext(image_name)[0], class_name, obj_id)
            visualization_paths.append(("5. Финальный контур", final_contour_path))
            
            print(f"📊 Создано визуализаций для {class_name} {obj_id}: {len(visualization_paths)}")
            
            # Сохраняем лучшую маску
            best_mask = best_mask_data['mask']
            mask_path = os.path.join(obj_dir, "6_final_mask.png")
            cv2.imwrite(mask_path, best_mask * 255)
            
            # Создаем полную маску для всего изображения
            full_mask = np.zeros((image_height, image_width), dtype=np.uint8)
            full_mask[ymin:ymax, xmin:xmax] = best_mask
            
            # Конвертируем в формат YOLO
            yolo_polygon = convert_mask_to_yolo_format(full_mask, image_width, image_height)
            
            if yolo_polygon:
                mask_data = {
                    'mask': full_mask,
                    'polygon': yolo_polygon,
                    'class_name': class_name,
                    'obj_id': obj_id,
                    'box': box,
                    'visualizations': visualization_paths
                }
                all_masks_data.append(mask_data)
                
                print(f"✅ {class_name} {obj_id} — сегментация сохранена")
            
            all_visualization_paths.extend(visualization_paths)
    
    # Создаем summary файл с информацией о визуализациях
    if all_visualization_paths:
        summary_path = os.path.join(image_main_dir, "visualization_summary.txt")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"ВИЗУАЛИЗАЦИИ ДЛЯ ИЗОБРАЖЕНИЯ: {image_name}\n")
            f.write("=" * 50 + "\n\n")
            
            for class_name in ['crop', 'weed']:
                class_objects = [m for m in all_masks_data if m['class_name'] == class_name]
                if class_objects:
                    f.write(f"{class_name.upper()} ОБЪЕКТЫ:\n")
                    for obj in class_objects:
                        f.write(f"  {class_name}_{obj['obj_id']:02d}:\n")
                        for viz_name, viz_path in obj.get('visualizations', []):
                            f.write(f"    {viz_name}: {os.path.basename(viz_path)}\n")
                    f.write("\n")
        
        print(f"📋 Summary файл создан: {summary_path}")
    
    # Выводим общую информацию
    print(f"\n📁 ОБРАБОТКА ЗАВЕРШЕНА:")
    print(f"   Изображение: {image_name}")
    print(f"   Найдено объектов: crop={len([m for m in all_masks_data if m['class_name'] == 'crop'])}, "
          f"weed={len([m for m in all_masks_data if m['class_name'] == 'weed'])}")
    print(f"   Создано визуализаций: {len(all_visualization_paths)}")
    print(f"   Результаты в: {image_main_dir}")
    
    return all_masks_data, all_visualization_paths

def main():
    """
    Основная функция с двумя режимами:
    1. Обработка одного изображения по индексу
    2. Обработка всего датасета
    """
    # === НАСТРОЙКИ ===
    path_images = "test_images/"
    
    # РЕЖИМ РАБОТЫ: 
    SINGLE_IMAGE_MODE = True  # True для обработки одного изображения, False для всего датасета
    IMAGE_INDEX = 0  # Индекс изображения для обработки (начинается с 0)
    
    if SINGLE_IMAGE_MODE:
        # Обработка одного изображения
        output_root = f"single_image_output/"
        results, viz_paths = process_single_image_by_index(IMAGE_INDEX, path_images, output_root)
        
        if results:
            print(f"\n✅ Обработка изображения {IMAGE_INDEX} завершена!")
            print(f"📁 Все результаты сохранены в: {output_root}")
        else:
            print(f"\n❌ Не удалось обработать изображение {IMAGE_INDEX}")
    
    else:
        # Обработка всего датасета (можно добавить позже)
        print("Режим обработки всего датасета временно отключен")

if __name__ == "__main__":
    main()