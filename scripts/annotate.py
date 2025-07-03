from ultralytics import YOLO
import os
import cv2
import argparse
from tqdm import tqdm

def generate_initial_annotations(input_base_dir, annotation_base_dir, weights_file):
    model = YOLO(weights_file)
    os.makedirs(annotation_base_dir, exist_ok=True)
    for subdir in os.listdir(input_base_dir):
        subdir_path = os.path.join(input_base_dir, subdir)
        if os.path.isdir(subdir_path):
            output_subdir = os.path.join(annotation_base_dir, subdir)
            os.makedirs(output_subdir, exist_ok=True)
            images = [f for f in os.listdir(subdir_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            for img_name in tqdm(images, desc=f"Processing {subdir}"):
                img_path = os.path.join(subdir_path, img_name)
                results = model.predict(img_path, conf=0.3, verbose=False)
                ann_path = os.path.join(output_subdir, img_name.replace('.jpg', '.txt'))
                with open(ann_path, 'w') as f:
                    for box in results[0].boxes:
                        class_id = box.cls[0].item()  # Use predicted class (may need mapping)
                        x_center, y_center, width, height = box.xywhn[0].tolist()
                        f.write(f"{int(class_id)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            print(f"✅ Initial annotations generated for {subdir} in {output_subdir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate initial annotations using a pre-trained YOLO model.")
    parser.add_argument('--input_base_dir', type=str, default="data/frames", help="Base directory containing frame subdirectories.")
    parser.add_argument('--annotation_base_dir', type=str, default="data/annotations", help="Base directory to save initial annotations.")
    parser.add_argument('--weights_file', type=str, default="yolo11n.pt", help="Path to pre-trained weights.")
    args = parser.parse_args()
    generate_initial_annotations(args.input_base_dir, args.annotation_base_dir, args.weights_file)