from ultralytics import YOLO
# Load a modei
# rrenet-n, rrenet-s, rrenet-m
model = YOLO('config/rrenet-n.yaml')
device = [0, 1]
# Train the model
# model.train(data='mycoco.yaml', epochs=300, imgsz=640, batch=64, device=device)
model.train(data='ultralytics/cfg/datasets/VOC.yaml', epochs=300, imgsz=640, batch=64, device=[0, 1])

