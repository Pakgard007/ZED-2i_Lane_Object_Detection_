import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

# กำหนด device (ใช้งาน GPU ถ้ามี ไม่งั้นใช้ CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class YourModel(nn.Module):
    def __init__(self, num_classes=80):  # ตัวอย่าง, เปลี่ยนตามจำนวนคลาสที่ใช้งาน
        super(YourModel, self).__init__()
        # โหลดโมเดล YOLOv5 (สามารถโหลดโมเดลที่ฝึกแล้วจากไฟล์ได้เช่นกัน)
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)  # ใช้ YOLOv5s ตัวอย่าง

    def forward(self, x):
        return self.model(x)
    
# ฟังก์ชันสำหรับการโหลดโมเดล
def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    model = YourModel()  # แทนที่ด้วยชื่อโมเดลที่คุณใช้
    model.load_state_dict(checkpoint['state_dict'])  # โหลดพารามิเตอร์จาก checkpoint
    model.to(device)
    model.eval()  # เปลี่ยนโมเดลเป็นโหมดทดสอบ
    return model

# ฟังก์ชันสำหรับการทำนาย
def predict(image_path, model, device):
    # โหลดภาพ
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # เปลี่ยนสีจาก BGR เป็น RGB
    img_resized = cv2.resize(img, (640, 640))  # ปรับขนาดภาพให้ตรงกับขนาดที่โมเดลต้องการ

    # แปลงภาพเป็น Tensor
    img_tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1).unsqueeze(0).to(device)

    # ทำนายผล
    with torch.no_grad():
        predictions = model(img_tensor)

    # ถ้าผลลัพธ์เป็นการทำนายในรูปของ Mask หรือ บางอย่างที่สามารถแสดงได้
    pred_image = predictions[0].cpu().numpy()  # แปลงเป็น numpy เพื่อแสดงผล
    return pred_image

# ฟังก์ชันแสดงผล
def show_image(image):
    plt.imshow(image)
    plt.axis('off')  # ปิดแสดงแกน
    plt.show()

# Main function สำหรับการโหลดโมเดลและทำนาย
def main():
    model_path = '/home/satoi/python/runs/BddDataset/checkpoint.pth'  # ใส่ path ไปยัง checkpoint ของคุณ
    image_path = '/home/satoi/python/captured_images/image_20241202_145142.jpg'  # ใส่ path ไปยังภาพที่ต้องการทำนาย

    # โหลดโมเดล
    model = load_model(model_path, device)

    # ทำนายภาพ
    pred_image = predict(image_path, model, device)

    # แสดงผล
    show_image(pred_image)

if __name__ == "__main__":
    main()
