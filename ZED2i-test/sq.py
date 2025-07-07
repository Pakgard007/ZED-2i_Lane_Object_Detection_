import torch

model = torch.hub.load('ultralytics/yolov5', 'custom', path='/absolute/path/to/best.pt', force_reload=True)
print(model)
