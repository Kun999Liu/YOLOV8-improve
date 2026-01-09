
from ultralytics import YOLO

if __name__=="__main__":


    # 使用YOLOv8.yamy文件搭建的模型训练
    # model = YOLO(r"D:\bilibili\model\ultralytics-main\ultralytics\cfg\models\v8\yolov8_my.yaml")  # build a new model from YAML
    # results = model.train(data=r'D:\bilibili\model\ultralytics-main\ultralytics\cfg\datasets\VOC_my.yaml',
    #                       epochs=100, imgsz=640, batch=4)
    #
    # # 加载已训练好的模型权重搭建模型训练
    # model = YOLO(r'D:\bilibili\model\ultralytics-main\tests\yolov8n.pt')  # load a pretrained model (recommended for training)
    # results = model.train(data=r'D:\bilibili\model\ultralytics-main\ultralytics\cfg\datasets\VOC_my.yaml',
    #                       epochs=100, imgsz=640, batch=4)

    # 使用自己的YOLOv8.yamy文件搭建模型并加载预训练权重训练模型
    model = YOLO("yolov8n.pt")

    # Display model information (optional)
    model.info()

    results = model.train(data="coco8.yaml", epochs=10, imgsz=640,
                          device="cpu")



