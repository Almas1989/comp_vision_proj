from ultralytics import YOLO
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

def evaluate_model(model_path, output_dir, exp_name):
       if not os.path.exists(model_path):
           raise FileNotFoundError(f"Model weights not found at {model_path}")
       if not os.path.exists('data/dataset.yaml'):
           raise FileNotFoundError("data/dataset.yaml not found")
       model = YOLO(model_path)
       try:
           metrics = model.val(data='data/dataset.yaml', split='test')
       except Exception as e:
           raise RuntimeError(f"Validation failed: {e}")
       precision = metrics.box.p
       recall = metrics.box.r
       map50 = metrics.box.map50
       map5095 = metrics.box.map
       f1 = 2 * (precision * recall) / (precision + recall + 1e-16)
       results = {
           'Experiment': exp_name,
           'mAP@0.5': map50,
           'mAP@0.5:0.95': map5095,
           'Precision': precision,
           'Recall': recall,
           'F1-Score': f1
       }
       os.makedirs(output_dir, exist_ok=True)
       csv_path = os.path.join(output_dir, 'evaluation.csv')
       df = pd.DataFrame([results])
       if os.path.exists(csv_path):
           df_existing = pd.read_csv(csv_path)
           df = pd.concat([df_existing, df], ignore_index=True)
       df.to_csv(csv_path, index=False)
       print(f"✅ Saved metrics for {exp_name} to {csv_path}")
       print(df)
       plt.figure(figsize=(8, 6))
       plt.bar(results.keys(), results.values(), color=['blue', 'cyan', 'green', 'orange', 'red'])
       plt.title(f'Evaluation Metrics - {exp_name}')
       plt.ylabel('Score')
       plt.ylim(0, 1)
       plt.tight_layout()
       plt.savefig(os.path.join(output_dir, f'metrics_plot_{exp_name}.png'))
       plt.close()
       print(f"✅ Saved plot for {exp_name} to {output_dir}/metrics_plot_{exp_name}.png")

if __name__ == "__main__":
       parser = argparse.ArgumentParser(description="Evaluate YOLOv11 model on test dataset.")
       parser.add_argument('--model_paths', nargs='+', default=['results/metrics/yolo11_exp1/weights/best.pt', 'results/metrics/yolo11_exp2/weights/best.pt'], help="Paths to model weights.")
       parser.add_argument('--output_dir', type=str, default='results/metrics', help="Directory to save metrics and plots.")
       args = parser.parse_args()
       for i, model_path in enumerate(args.model_paths):
           exp_name = f"yolo11_exp{i+1}"
           evaluate_model(model_path, args.output_dir, exp_name)