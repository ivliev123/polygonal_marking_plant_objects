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

def visualize_binary_mask(refinement_history, output_dir, img_name_article):

    if not refinement_history:
        print("❌ Нет данных для визуализации")
        return

    n = len(refinement_history)
    n_show = min(15, n)

    # === 1. Визуализация контуров ===
    fig, axes = plt.subplots(3, 5, figsize=(18, 9))
    axes = axes.ravel()

    for i, result in enumerate(refinement_history[:n_show]):
        vis = result['mask']
        vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        axes[i].imshow(vis_rgb)
        axes[i].set_title(
            f"H=[{result['h_lower']},{result['h_upper']}]\n"
            f"Score={result['score']:.2f}\n"
            f"σ_color={result['color_std']:.1f}, Edge={result['edge_strength']:.2f}",
            fontsize=8
        )
        axes[i].axis('off')

    # Заполняем оставшиеся ячейки, если меньше 12 контуров
    for j in range(n_show, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()

    output_path = os.path.join(output_dir, img_name_article)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

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
        
        # Определяем цвет и счетчик
        if class_name == 'crop':
            color = (0, 255, 0)  # зеленый для crop
            crop_count += 1
            label = f"crop_{crop_count}"
        else:
            color = (0, 0, 255)  # красный для weed
            weed_count += 1
            label = f"weed_{weed_count}"
        
        # Рисуем bounding box на исходном изображении
        cv2.rectangle(original_with_boxes, (xmin, ymin), (xmax, ymax), color, 2)
        
        # Добавляем текст с классом
        cv2.putText(original_with_boxes, label, (xmin, ymin-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Копируем вырезанное растение на черный фон
        crop = image_bgr[ymin:ymax, xmin:xmax]
        black_background[ymin:ymax, xmin:xmax] = crop
    
    # Создаем фигуру с двумя subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Левое изображение: исходное с bounding boxes
    ax1.imshow(cv2.cvtColor(original_with_boxes, cv2.COLOR_BGR2RGB))
    ax1.set_title(f'Исходное изображение с ограничивающими рамками\n(crop: {crop_count}, weed: {weed_count})', 
                  fontsize=14, pad=15)
    ax1.axis('off')
    
    # Правое изображение: ВСЕ растения на черном фоне
    ax2.imshow(cv2.cvtColor(black_background, cv2.COLOR_BGR2RGB))
    ax2.set_title('Все вырезанные растения на черном фоне', fontsize=14, pad=15)
    ax2.axis('off')
    
    # Добавляем легенду
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.6, label='Crop растения'),
        Patch(facecolor='red', alpha=0.6, label='Weed растения')
    ]
    
    fig.legend(handles=legend_elements, loc='upper center', 
               bbox_to_anchor=(0.5, 0.05), ncol=2, fontsize=12)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)  # место для легенды
    
    output_path = os.path.join(output_dir, img_name_article)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"✅ Визуализировано объектов: {crop_count} crop, {weed_count} weed")
    return output_path



def visualize_score_progression(history, best_index, output_dir, image_name, class_name, obj_id, img_name_article):
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
    
    output_path = os.path.join(output_dir, img_name_article)
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

    
    for obj_id, (box, class_name) in enumerate(zip(true_boxes, true_classes)):
        # Создаем отдельную папку для каждого объекта
        if class_name in ['crop', 'weed']:
            obj_dir = os.path.join(image_main_dir, f"{class_name}_{obj_id:02d}")
            os.makedirs(obj_dir, exist_ok=True)
            
            xmin, ymin, xmax, ymax = box
            crop_image = image_bgr[ymin:ymax, xmin:xmax]
            
            print(f"🔍 Обработка {class_name} {obj_id}...")
            
            # Сохраняем исходный 

            # crop_path = os.path.join(obj_dir, "0_original_crop.jpg")
            # cv2.imwrite(crop_path, crop_image)
            
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
            img_name_article = "4_visualize_binary_mask.png"
            # visualize_binary_mask(crop_image, best_mask_data, obj_dir, os.path.splitext(image_name)[0], class_name, obj_id, img_name_article)
            visualize_binary_mask(history, obj_dir, img_name_article)

            img_name_article = "5_visualize_score_progression.png"
            visualize_score_progression(history, best_index, obj_dir, os.path.splitext(image_name)[0], class_name, obj_id, img_name_article)
            
            img_name_article = "6_visualize_hybrid_refinement.png"
            visualize_hybrid_refinement(crop_image, history, obj_dir, img_name_article)



            
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

def main():
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