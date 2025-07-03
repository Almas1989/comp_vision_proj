import cv2
import os
import argparse

def extract_frames(video_dir, output_base_dir, fps=0.2):
    os.makedirs(output_base_dir, exist_ok=True)
    for video_file in os.listdir(video_dir):
        if video_file.lower().endswith(('.mp4', '.avi', '.mov')):
            video_path = os.path.join(video_dir, video_file)
            video_name = os.path.splitext(video_file)[0]
            output_dir = os.path.join(output_base_dir, video_name)
            os.makedirs(output_dir, exist_ok=True)
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"⚠️ Cannot open video {video_path}, skipping.")
                continue
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            if video_fps == 0:
                print(f"⚠️ Video FPS is zero for {video_path}, skipping.")
                cap.release()
                continue
            frame_interval = max(int(video_fps / fps), 1)  # e.g., 1 frame every 5 seconds if video_fps=25
            count = 0
            frame_number = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if count % frame_interval == 0:
                    frame_path = os.path.join(output_dir, f"frame_{frame_number:06d}.jpg")
                    cv2.imwrite(frame_path, frame)
                    frame_number += 1
                count += 1
            cap.release()
            print(f"✅ Extracted {frame_number} frames from {video_file} to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from multiple video files.")
    parser.add_argument('--video_dir', type=str, default="data/raw", help="Directory containing video files.")
    parser.add_argument('--output_base_dir', type=str, default="data/frames", help="Base directory to save frames.")
    parser.add_argument('--fps', type=float, default=0.2, help="Frames to extract per second (default=0.2, ~1 frame every 5 seconds).")
    args = parser.parse_args()
    extract_frames(args.video_dir, args.output_base_dir, args.fps)