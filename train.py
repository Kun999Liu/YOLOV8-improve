
from ultralytics import YOLO

if __name__=="__main__":

    # 使用自己的YOLOv8.yamy文件搭建模型并加载预训练权重训练模型
    model = YOLO(r"./yolov8.yaml")

    # Display model information (optional)
    model.info()

    results = model.train(data=r"./TransmissionTower.yaml",
                          epochs=10,
                          imgsz=256,
                          device="cpu")



