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

def visualize_single_crop_preprocessing(image_bgr, crop_image, box, class_name, obj_id, output_dir, img_name_article):
    """
    Визуализация предобработки для одного вырезанного растения
    """
    # Создаем копию исходного изображения с выделенным bounding box
    original_with_box = image_bgr.copy()
    xmin, ymin, xmax, ymax = box
    
    # Рисуем bounding box
    color = (0, 255, 0) if class_name == 'crop' else (0, 0, 255)
    cv2.rectangle(original_with_box, (xmin, ymin), (xmax, ymax), color, 3)
    
    # Создаем черный фон такого же размера как кроп
    black_background = np.zeros_like(crop_image)
    
    # Копируем растение на черный фон
    black_background = crop_image.copy()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Левое изображение: исходное с выделенным bounding box
    ax1.imshow(cv2.cvtColor(original_with_box, cv2.COLOR_BGR2RGB))
    ax1.set_title(f'Исходное изображение\nВыделен {class_name} {obj_id}', fontsize=12)
    ax1.axis('off')
    
    # Правое изображение: вырезанное растение
    ax2.imshow(cv2.cvtColor(black_background, cv2.COLOR_BGR2RGB))
    ax2.set_title(f'Вырезанный фрагмент\n{class_name} {obj_id}', fontsize=12)
    ax2.axis('off')
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, img_name_article)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return output_path



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

    h_ranges_to_try = [(22 + i, 85) for i in range(15)]
    # print(h_ranges_to_try)
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
            refinement_history.append({
                'h_lower': h_lower,
                'h_upper': h_upper,
                'area': False,
                'color_std': False,
                'edge_strength': False,
                'texture_std': False,
                'score': False,
                'mask': mask.copy(),
                'contour': False,
                'best_edge': False
                })
            continue

        main_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(main_contour)
        # if area < 50:  # фильтр шумных контуров
        #     continue

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
    Отбрасывает последние точки где score = 0 (нет контура).
    """
    scores = np.array(scores, dtype=float)
    
    # Находим индекс последнего ненулевого score
    non_zero_indices = np.where(scores > 0)[0]
    
    if len(non_zero_indices) == 0:
        return 0  # Все scores равны 0
    
    # Обрезаем scores до последнего ненулевого значения
    last_valid_index = non_zero_indices[-1]
    valid_scores = scores[:last_valid_index + 1]
    
    # print(f"📊 Анализ scores: всего {len(scores)}, валидных {len(valid_scores)}")
    # print(f"📊 Валидные scores: {valid_scores}")
    
    if len(valid_scores) < 2:
        return 0

    score_min = np.min(valid_scores)
    score_max = np.max(valid_scores)
    score_range = score_max - score_min

    threshold_drop = score_range * 0.2

    best_idx = 0
    for i in range(1, len(valid_scores)):
        delta = valid_scores[i] - valid_scores[i - 1]

        if delta < -threshold_drop:
            print(f"⚠️ Резкое падение на шаге {i}: Δ={delta:.3f} → выбираем индекс {i-1}")
            best_idx = i - 1
            break

        best_idx = i

    print(f"✅ Выбран лучший индекс: {best_idx} (score={valid_scores[best_idx]:.3f})")
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

# VISUAL
def visualize_all_crops_preprocessing(image_bgr, boxes, classes, output_dir, img_name_article):
    """
    Визуализация предобработки данных: исходное изображение и ВСЕ вырезанные растения на черном фоне
    Рисунок 3.4.1 — Схема предобработки данных
    """
    # Создаем копию исходного изображения
    original_with_boxes = image_bgr.copy()
    
    # Создаем черный фон такого же размера
    black_background = np.zeros_like(image_bgr)
    
    # Счетчики для разных классов
    crop_count = 0
    weed_count = 0
    
    # Рисуем bounding boxes на исходном изображении и вырезаем растения
    for i, (box, class_name) in enumerate(zip(boxes, classes)):
        if class_name not in ['crop', 'weed']:
            continue
            
        xmin, ymin, xmax, ymax = box
        
        # Определяем цвет
        if class_name == 'crop':
            color = (0, 255, 0)  # зеленый для crop
            crop_count += 1
        else:
            color = (0, 0, 255)  # красный для weed
            weed_count += 1
        
        # Рисуем bounding box на исходном изображении (БЕЗ подписей)
        cv2.rectangle(original_with_boxes, (xmin, ymin), (xmax, ymax), color, 2)
        
        # Копируем вырезанное растение на черный фон
        crop = image_bgr[ymin:ymax, xmin:xmax]
        black_background[ymin:ymax, xmin:xmax] = crop
    
    # === СОХРАНЕНИЕ ОТДЕЛЬНЫХ КАРТИНОК ===
    
    # Сохраняем левую картинку (исходное с bounding boxes)
    left_image_path = os.path.join(output_dir, "0_left_original_with_boxes.png")
    cv2.imwrite(left_image_path, original_with_boxes)  # Сохраняем в BGR для OpenCV
    
    # Сохраняем правую картинку (растения на черном фоне)
    right_image_path = os.path.join(output_dir, "0_right_crops_on_black.png")
    cv2.imwrite(right_image_path, black_background)  # Сохраняем в BGR для OpenCV
    
    print(f"✅ Визуализировано объектов: {crop_count} crop, {weed_count} weed")
    print(f"✅ Левая картинка: {left_image_path}")
    print(f"✅ Правая картинка: {right_image_path}")
    
    return {
        'left': left_image_path,
        'right': right_image_path
    }

# === ВИЗУАЛИЗАЦИЯ ЛУЧШЕЙ МАСКИ ===
def visualize_mask(crop_image, output_dir, img_name_article):

    h_ranges_to_try = [(22 + i, 85) for i in range(15)]

    hsv = cv2.cvtColor(crop_image, cv2.COLOR_BGR2HSV)

    for h_i in range(1):
        h_lower, h_upper = h_ranges_to_try[h_i]

        lower = np.array([h_lower, 0, 0])
        upper = np.array([h_upper, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
    
    image_path = os.path.join(output_dir, "3_1_" + img_name_article)
    cv2.imwrite(image_path, crop_image)
    
    image_path = os.path.join(output_dir, "3_2_" + img_name_article)
    cv2.imwrite(image_path, hsv)
    
    image_path = os.path.join(output_dir, "3_3_" + img_name_article)
    cv2.imwrite(image_path, mask)




# === ВИЗУАЛИЗАЦИЯ ВОССТАНОВЛЕННОЙ МАСКИ ===
def visualize_restore_mask_pretty(crop_image, output_dir, img_name_article):
    h_ranges_to_try = [(22 + i, 85) for i in range(15)]

    hsv = cv2.cvtColor(crop_image, cv2.COLOR_BGR2HSV)

    for h_i in range(1):
        h_lower, h_upper = h_ranges_to_try[h_i]

        lower = np.array([h_lower, 0, 0])
        upper = np.array([h_upper, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)

        # Морфологическая очистка
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_1 = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask_2 = cv2.morphologyEx(mask_1, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(mask_2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # cv2.drawContours(vis, [result['contour']], -1, color, 2)

    # mask_filled = np.zeros(mask.shape, dtype=np.uint8)
    # cv2.drawContours(mask_filled, [contours], -1, 255, -1)

    mask_filled = np.zeros_like(crop_image, dtype=np.uint8)
    if contours and len(contours) > 0:
        # правильно передаём список contours и рисуем все контуры (-1)
        cv2.drawContours(mask_filled, contours, -1, (0, 255, 0), thickness=1)
    else:
        # оставляем mask_filled пустой (черной) если контуров нет
        pass
    
    image_path = os.path.join(output_dir, "4_1_" + img_name_article)
    cv2.imwrite(image_path, mask)
    
    image_path = os.path.join(output_dir, "4_2_" + img_name_article)
    cv2.imwrite(image_path, mask_1)
    
    image_path = os.path.join(output_dir, "4_3_" + img_name_article)
    cv2.imwrite(image_path, mask_2)

    image_path = os.path.join(output_dir, "4_4_" + img_name_article)
    cv2.imwrite(image_path, mask_filled)



def visualize_restore_mask(refinement_history, output_dir, img_name_article):

    if not refinement_history:
        print("❌ Нет данных для визуализации")
        return

    n = len(refinement_history)
    print(n)
    # n_show = min(15, n)
    n_show = 15

    # === 1. Визуализация контуров ===
    fig, axes = plt.subplots(3, 5, figsize=(18, 9))
    axes = axes.ravel()

    for i in range(15):  # Всегда 15 ячеек
        if i < n_show:  # Есть данные для этого индекса
            result = refinement_history[i]
            vis = result['mask']
            
            # Проверяем есть ли контур для отрисовки
            if result['contour'] is not False and result['contour'] is not None:
                vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
                
                # Формируем заголовок с метриками
                title = (f"H=[{result['h_lower']},{result['h_upper']}]\n"
                        f"Score={result['score']:.2f}\n")
                        # f"σ_color={result['color_std']:.1f}, Edge={result['edge_strength']:.2f}")
            else:
                # Нет контура - показываем исходное изображение без контура
                vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
                title = (f"H=[{result['h_lower']},{result['h_upper']}]\n"
                        # f"NO CONTOUR\n"
                        f"Score=0.00\n")
            
            axes[i].imshow(vis_rgb)
            axes[i].set_title(title, fontsize=12)
            
        else:
            # Нет данных для этого индекса - пустая ячейка
            axes[i].imshow(np.ones_like(image_bgr) * 255)  # Белый фон
            axes[i].set_title(f"Нет данных\nИндекс {i}", fontsize=12)
        
        axes[i].axis('off')


    plt.tight_layout()

    output_path = os.path.join(output_dir, img_name_article)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def visualize_score_progression(history, best_index, output_dir, image_name, class_name, obj_id, img_name_article):
    """
    Визуализация изменения оценки качества сегментации - график Score
    """
    # Фильтруем только валидные точки (score > 0)
    valid_indices = [i for i, item in enumerate(history) if item['score'] > 0]
    
    if not valid_indices:
        print("❌ Нет валидных данных для визуализации прогрессии Score")
        return None
    
    scores = [history[i]['score'] for i in valid_indices]
    
    # Корректируем best_index
    if best_index in valid_indices:
        valid_best_index = valid_indices.index(best_index)
    else:
        valid_best_index = len(valid_indices) - 1  # последний валидный
    
    print(f"📊 Визуализация Score: {len(valid_indices)}/{len(history)} валидных точек")
    
    # График SCORE
    fig, ax = plt.subplots(figsize=(10, 5))
    x_positions = range(len(scores))
    
    # График оценки
    ax.plot(x_positions, scores, 'b-', marker='o', linewidth=2, markersize=6)
    ax.axvline(x=valid_best_index, color='r', linestyle='--', linewidth=2)
    
    # Настройки
    ax.set_xticks(x_positions)  # Каждый индекс
    ax.set_xlabel('Индекс диапазона H')
    ax.set_ylabel('Score')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Сохраняем график
    output_path = os.path.join(output_dir, img_name_article)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    
    print(f"✅ График Score сохранен: {output_path}")
    return output_path

def visualize_area_progression(history, best_index, output_dir, image_name, class_name, obj_id, img_name_article):
    """
    Визуализация изменения площади контура - график Area
    """
    # Фильтруем только валидные точки (score > 0)
    valid_indices = [i for i, item in enumerate(history) if item['score'] > 0]
    
    if not valid_indices:
        print("❌ Нет валидных данных для визуализации прогрессии Area")
        return None
    
    areas = [history[i]['area'] for i in valid_indices]
    
    # Корректируем best_index
    if best_index in valid_indices:
        valid_best_index = valid_indices.index(best_index)
    else:
        valid_best_index = len(valid_indices) - 1  # последний валидный
    
    print(f"📊 Визуализация Area: {len(valid_indices)}/{len(history)} валидных точек")
    
    # График AREA
    fig, ax = plt.subplots(figsize=(10, 5))
    x_positions = range(len(areas))
    
    # График площади
    ax.plot(x_positions, areas, 'g-', marker='s', linewidth=2, markersize=6)
    ax.axvline(x=valid_best_index, color='r', linestyle='--', linewidth=2)
    
    # Настройки
    ax.set_xticks(x_positions)  # Каждый индекс
    ax.set_xlabel('Индекс диапазона H')
    ax.set_ylabel('Площадь (пиксели)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Сохраняем график
    output_path = os.path.join(output_dir, img_name_article)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    
    print(f"✅ График Area сохранен: {output_path}")
    return output_path

def visualize_contour_history(image_bgr, refinement_history, output_dir, img_name_article):
    """
    Визуализирует гибридный процесс адаптации:
    - отображает контуры
    - подписывает метрики (score, color_std, edge_strength, и т.д.)
    - строит график изменения score по диапазонам H
    """

    if not refinement_history:
        print("❌ Нет данных для визуализации")
        return

    n = len(refinement_history)
    n_show = min(15, n)

    # === 1. Визуализация контуров ===
    fig, axes = plt.subplots(3, 5, figsize=(18, 9))
    axes = axes.ravel()

    for i in range(15):  # Всегда 15 ячеек
        if i < n_show:  # Есть данные для этого индекса
            result = refinement_history[i]
            vis = image_bgr.copy()
            
            # Проверяем есть ли контур для отрисовки
            if result['contour'] is not False and result['contour'] is not None:
                color = (0, 255, 0) if result['best_edge'] else (255, 0, 255)
                cv2.drawContours(vis, [result['contour']], -1, color, 2)
                vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
                
                # Формируем заголовок с метриками
                title = (f"H=[{result['h_lower']},{result['h_upper']}]\n"
                        f"Score={result['score']:.2f}\n")
                        # f"σ_color={result['color_std']:.1f}, Edge={result['edge_strength']:.2f}")
            else:
                # Нет контура - показываем исходное изображение без контура
                vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
                title = (f"H=[{result['h_lower']},{result['h_upper']}]\n"
                        # f"NO CONTOUR\n"
                        f"Score=0.00\n")
            
            axes[i].imshow(vis_rgb)
            axes[i].set_title(title, fontsize=12)
            
        else:
            # Нет данных для этого индекса - пустая ячейка
            axes[i].imshow(np.ones_like(image_bgr) * 255)  # Белый фон
            axes[i].set_title(f"Нет данных\nИндекс {i}", fontsize=12)
        
        axes[i].axis('off')

    plt.tight_layout()

    output_path = os.path.join(output_dir, img_name_article)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
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

    img_name_article_array = [
                                "1_visualize_preprocessing.png",
                                "2_visualize_mask.png",
                                "3_visualize_restore_mask.png",
                                "4_visualize_binary_mask_history.png",
                                "5_visualize_score_progression.png",
                                "5_visualize_area_progression.png",
                                "6_visualize_contour_history.png",

    ]

    preprocessing_path = visualize_all_crops_preprocessing(image_bgr, true_boxes, true_classes, image_main_dir, "1_visualize_preprocessing.png")
    
    all_masks_data = []

    
    for obj_id, (box, class_name) in enumerate(zip(true_boxes, true_classes)):
        # Создаем отдельную папку для каждого объекта
        if class_name in ['crop', 'weed']:
            obj_dir = os.path.join(image_main_dir, f"{class_name}_{obj_id:02d}")
            os.makedirs(obj_dir, exist_ok=True)
            
            xmin, ymin, xmax, ymax = box
            crop_image = image_bgr[ymin:ymax, xmin:xmax]
            
            print(f"🔍 Обработка {class_name} {obj_id}...")
            
            # Сохраняем исходный 

            # Запускаем адаптивный HSV-анализ
            history = adaptive_refinement_v2(crop_image)
            
            if not history:
                print(f"❌ Нет контуров для {class_name} {obj_id}")
                continue
            
            # Находим лучший контур
            scores = [item['score'] for item in history]
            best_index = find_best_index_by_score_delta(scores)
            best_mask_data = history[best_index]
            history[best_index]['best_edge'] = True          
            
            # === РАЗДЕЛЬНАЯ ВИЗУАЛИЗАЦИЯ КАЖДОГО ЭТАПА ===      
            print(obj_id)

            visualize_mask(crop_image, obj_dir, "_visualize_mask.png")
            visualize_restore_mask_pretty(crop_image, obj_dir, "_visualize_restore_mask.png")
            visualize_restore_mask(history, obj_dir, "4_visualize_binary_mask_history.png")
            visualize_score_progression(history, best_index, obj_dir, os.path.splitext(image_name)[0], class_name, obj_id, "5_visualize_score_progression.png")
            visualize_area_progression(history, best_index, obj_dir, os.path.splitext(image_name)[0], class_name, obj_id, "5_visualize_area_progression.png")
            visualize_contour_history(crop_image, history, obj_dir, "6_visualize_contour_history.png")



            
            # print(f"📊 Создано визуализаций для {class_name} {obj_id}: {len(visualization_paths)}")
            
            # Сохраняем лучшую маску

            # тут что-то криво-косо
            # best_mask = best_mask_data['mask']
            # mask_path = os.path.join(obj_dir, "6_final_mask.png")
            # cv2.imwrite(mask_path, best_mask * 255)
            
            # # Создаем полную маску для всего изображения
            # full_mask = np.zeros((image_height, image_width), dtype=np.uint8)
            # full_mask[ymin:ymax, xmin:xmax] = best_mask
            
            # # Конвертируем в формат YOLO
            # yolo_polygon = convert_mask_to_yolo_format(full_mask, image_width, image_height)
            
            # if yolo_polygon:
            #     mask_data = {
            #         'mask': full_mask,
            #         'polygon': yolo_polygon,
            #         'class_name': class_name,
            #         'obj_id': obj_id,
            #         'box': box,
            #     }
            #     all_masks_data.append(mask_data)
                
            #     print(f"✅ {class_name} {obj_id} — сегментация сохранена")
            

    

    
    # Выводим общую информацию
    print(f"\n📁 ОБРАБОТКА ЗАВЕРШЕНА:")
    print(f"   Изображение: {image_name}")
    print(f"   Найдено объектов: crop={len([m for m in all_masks_data if m['class_name'] == 'crop'])}, "
          f"weed={len([m for m in all_masks_data if m['class_name'] == 'weed'])}")
    print(f"   Результаты в: {image_main_dir}")
    
    return all_masks_data


def process_dataset_range(start_idx, end_idx, path_images, output_root):
    """
    Обрабатывает участок датасета от start_idx до end_idx включительно.
    Для каждого изображения создаёт визуализации:
        - 2_visualize_mask.png
        - 3_visualize_restore_mask.png
    """

    image_files = sorted([f for f in os.listdir(path_images) if f.lower().endswith((".png", ".jpg"))])

    if not image_files:
        print("❌ В папке нет изображений.")
        return

    end_idx = min(end_idx, len(image_files) - 1)

    print(f"\n=== ▶ Обработка изображений от {start_idx} до {end_idx} ===")

    os.makedirs(output_root, exist_ok=True)

    for idx in range(start_idx, end_idx + 1):
        img_name = image_files[idx]
        print(f"\n--- [{idx}] Обработка {img_name} ---")

        # Вычисление директорий
        img_output_dir = os.path.join(output_root, f"img_{idx:04d}")
        os.makedirs(img_output_dir, exist_ok=True)

        # Загружаем и обрабатываем
        output = process_single_image_by_index(idx, path_images, img_output_dir)

        # НЕТ НИКАКИХ ОБЪЕКТОВ → ПРОПУСК
        if not output or output == 0:
            print(f"  ⚠ Изображение [{idx}] пропущено (нет объектов или ошибка)")
            continue

        results, viz_paths = output

        # Берём лучшую маску
        best_mask_data = results["best_mask"]
        crop_image = results["crop_image"]

        # Визуализации
        visualize_mask(crop_image, best_mask_data, img_output_dir, "2_visualize_mask.png")
        visualize_restore_mask_pretty(crop_image, best_mask_data, img_output_dir, "3_visualize_restore_mask.png")

        print(f"  ✔ Готово → {img_output_dir}")

    print("\n🎉 Готово! Участок датасета обработан.")



def main():
    # === НАСТРОЙКИ ===
    path_images = "test_images/"

    MODE = "RANGE"  
    # варианты:
    # MODE = "SINGLE"
    # MODE = "ALL"
    # MODE = "RANGE"

    IMAGE_INDEX = 0  # для MODE=SINGLE
    RANGE_START = 10  # для MODE=RANGE
    RANGE_END = 250   # включительно

    if MODE == "SINGLE":
        output_root = "single_image_output/"
        results, viz_paths = process_single_image_by_index(IMAGE_INDEX, path_images, output_root)

        if results:
            print(f"\n✅ Обработка изображения {IMAGE_INDEX} завершена!")
            print(f"📁 Результаты: {output_root}")
        else:
            print(f"\n❌ Не удалось обработать изображение {IMAGE_INDEX}")

    elif MODE == "RANGE":
        output_root = "dataset_range_output/"
        process_dataset_range(RANGE_START, RANGE_END, path_images, output_root)

    elif MODE == "ALL":
        print("Функция обработки всех изображений будет добавлена позже.")

    else:
        print("❌ Ошибка: неизвестный MODE")


if __name__ == "__main__":
    main()