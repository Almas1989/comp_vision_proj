from ultralytics import YOLO
import cv2
import os
from tqdm import tqdm
import argparse

def visualize_results(model_path, test_dir, output_video_path, fps=10):
       if not os.path.exists(model_path):
           raise FileNotFoundError(f"Model weights not found at {model_path}")
       if not os.path.exists(test_dir):
           raise FileNotFoundError(f"Test directory not found at {test_dir}")
       model = YOLO(model_path)
       images = sorted([
           f for f in os.listdir(test_dir)
           if f.lower().endswith(('.jpg', '.jpeg', '.png'))
       ])
       if not images:
           raise FileNotFoundError(f"No images found in {test_dir}")
       first_img_path = os.path.join(test_dir, images[0])
       first_img = cv2.imread(first_img_path)
       if first_img is None:
           raise FileNotFoundError(f"Failed to read {first_img_path}")
       height, width = first_img.shape[:2]
       os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
       out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
       for img_name in tqdm(images, desc=f"Generating video for {os.path.basename(model_path)}"):
           img_path = os.path.join(test_dir, img_name)
           results = model.predict(img_path, conf=0.5, verbose=False)
           annotated_img = results[0].plot()
           out.write(annotated_img)
       out.release()
       print(f"✅ Output video saved to {output_video_path}")

if __name__ == "__main__":
       parser = argparse.ArgumentParser(description="Visualize YOLOv11 predictions as a video.")
       parser.add_argument('--model_paths', nargs='+', default=['results/metrics/yolo11_exp2/weights/best.pt', 'results/metrics/yolo11_exp13/weights/best.pt'], help="Paths to model weights.")
       parser.add_argument('--test_dir', type=str, default='data/augmented/test/images', help="Directory with test images.")
       parser.add_argument('--output_dir', type=str, default='results', help="Directory to save output videos.")
       parser.add_argument('--fps', type=float, default=10, help="Frames per second for output video.")
       args = parser.parse_args()
       for i, model_path in enumerate(args.model_paths):
           output_video_path = os.path.join(args.output_dir, f"output_video_exp{i+1}.mp4")
           visualize_results(model_path, args.test_dir, output_video_path, args.fps)