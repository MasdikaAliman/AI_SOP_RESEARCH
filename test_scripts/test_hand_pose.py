from ultralytics import YOLO



YOLO("../models/yolo26s-pose-hands.pt").predict(
    source=0,
    show=True,
    save=True,
    project="webcam",
    name="test",
    device="cpu"
)