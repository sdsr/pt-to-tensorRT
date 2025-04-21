from ultralytics import YOLO

model = YOLO("yolo11x.pt")          # 학습한 v11 가중치
model.export(
    format="engine",               # 바로 TensorRT
    device="cuda",                 # "cpu"면 ONNX → TRT 단계만 생략
    imgsz=640,                     # 입력 해상도
    half=True,                     # FP16
    dynamic=True,                  # 가변 배치 (1–8)
    workspace=4                    # GB, Jetson은 4–6 GB 권장
)
# 결과: yolo11s.engine + yolo11s.onnx  (두 파일 모두 생성)
