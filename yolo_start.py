# Создать скрипт для регенерации результатов
import os
from ultralytics import YOLO

# Путь к данным
data_yaml = "yolo_segmentation_dataset/data.yaml"

# Загружаем лучшие веса
model = YOLO('runs/segment/yolov11n-seg_300e/weights/best.pt')

# Валидация сгенерирует новые графики
results = model.val(
    data=data_yaml,
    imgsz=640,
    batch=32,
    save_json=True,
    plots=True,  # Это создаст confusion matrix, F1, PR, P, R curves
    name='yolov11n-seg_300e_val'  # Новая папка для результатов
)

print("Results saved to:", results.save_dir)