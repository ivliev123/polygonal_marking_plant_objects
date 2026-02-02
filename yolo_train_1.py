from ultralytics import YOLO
import os

# Путь к данным
data_yaml = "yolo_segmentation_dataset/data.yaml"

# Проверяем данные
if not os.path.exists(data_yaml):
    print(f"Ошибка: {data_yaml} не найден")
    exit()

# Загружаем YOLOv11
model = YOLO('yolo11l-seg.pt')

# Обучаем
model.train(
    data=data_yaml,
    epochs=300,
    imgsz=640,
    batch=8,
    # device='cpu',
    device='0',
    # workers=0,
    name = 'yolo11l-seg_300e'
)

print("Обучение завершено")