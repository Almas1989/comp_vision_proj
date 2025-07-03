from ultralytics import YOLO
import os

def train_yolo(exp_name, batch_size, lr):
    weights_file = "yolo11n.pt"
    if not os.path.exists(weights_file):
        raise FileNotFoundError(f"{weights_file} not found in project directory")
    model = YOLO(weights_file)
    try:
        model.train(
            data='data/dataset.yaml',
            epochs=20,
            batch=batch_size,
            imgsz=416,
            name=exp_name,
            project='results/metrics',
            lr0=lr,
            momentum=0.937 if exp_name == 'yolo11_exp1' else 0.9,
            weight_decay=0.0005
        )
        print(f"✅ Training completed for {exp_name}")
    except Exception as e:
        print(f"⚠️ Training failed for {exp_name}: {e}")
        raise

if __name__ == "__main__":
    train_yolo('yolo11_exp1', 16, 0.01)
    train_yolo('yolo11_exp2', 8, 0.005)