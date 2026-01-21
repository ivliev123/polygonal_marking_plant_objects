from ultralytics import YOLO
import os

# Путь к данным
data_yaml = "yolo_segmentation_dataset/data.yaml"

# Проверяем данные
if not os.path.exists(data_yaml):
    print(f"Ошибка: {data_yaml} не найден")
    exit()

# Загружаем YOLOv11
model = YOLO('yolo11s-seg.pt')

# Обучаем
model.train(
    data=data_yaml,
    epochs=200,
    imgsz=640,
    batch=16,
    # device='cpu',
    device='0',
    workers=0,
    name = 'yolo11s-seg_200e'
)

# # Обучение
# model.train(
#     data="coco8.yaml",
#     epochs=50,
#     imgsz=640,
#     batch=4,      # если вдруг OOM → 2
#     device=0,
#     workers=0,
#     amp=True
# )

# device='0' # для видеокарты

print("Обучение завершено")