import albumentations as A
import cv2
import os
import random

def collect_images_and_annotations(input_dir, annotation_dir):
    image_paths = []
    annotation_paths = []

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(root, file)
                relative_path = os.path.relpath(img_path, input_dir)

                # Handle different image extensions for annotation matching
                base_name = os.path.splitext(relative_path)[0]
                ann_path = os.path.join(annotation_dir, base_name + '.txt')

                if os.path.exists(ann_path):
                    image_paths.append(img_path)
                    annotation_paths.append(ann_path)
                else:
                    print(f"⚠️ Annotation not found for {relative_path}")

    print(f"Found {len(image_paths)} image-annotation pairs")
    return list(zip(image_paths, annotation_paths))

def augment_images(input_dir, annotation_dir, output_dir, split_ratios=(0.7, 0.15, 0.15), augment_count=3):
    transform = A.Compose([
        A.Rotate(limit=30, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        A.HorizontalFlip(p=0.5),
        A.RandomScale(scale_limit=0.1, p=0.3),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    pairs = collect_images_and_annotations(input_dir, annotation_dir)
    if not pairs:
        print("❌ No image-annotation pairs found!")
        return

    random.shuffle(pairs)
    total = len(pairs)
    train_end = int(total * split_ratios[0])
    val_end = train_end + int(total * split_ratios[1])

    splits = {
        'train': pairs[:train_end],
        'val': pairs[train_end:val_end],
        'test': pairs[val_end:]
    }

    for split, pair_list in splits.items():
        print(f"Processing {len(pair_list)} files for {split} split...")

        img_out_dir = os.path.join(output_dir, split, "images")
        label_out_dir = os.path.join(output_dir, split, "labels")
        os.makedirs(img_out_dir, exist_ok=True)
        os.makedirs(label_out_dir, exist_ok=True)

        processed_count = 0
        for img_path, ann_path in pair_list:
            img = cv2.imread(img_path)
            if img is None:
                print(f"⚠️ Can't read {img_path}")
                continue

            # Read annotations
            with open(ann_path, 'r') as f:
                bboxes = []
                labels = []
                for line in f:
                    items = line.strip().split()
                    if len(items) != 5:
                        continue
                    class_id, x, y, w, h = map(float, items)
                    bboxes.append([x, y, w, h])
                    labels.append(int(class_id))

            if not bboxes:
                print(f"⚠️ No valid bboxes found in {ann_path}")
                continue

            # Save original image and annotation
            filename = os.path.basename(img_path)
            base_name = os.path.splitext(filename)[0]
            ext = os.path.splitext(filename)[1]

            # Original
            img_output_path = os.path.join(img_out_dir, filename)
            label_output_path = os.path.join(label_out_dir, base_name + '.txt')

            cv2.imwrite(img_output_path, img)
            with open(label_output_path, 'w') as f:
                for bbox, label in zip(bboxes, labels):
                    f.write(f"{label} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")

            # Generate augmented versions
            for aug_idx in range(augment_count):
                try:
                    augmented = transform(image=img, bboxes=bboxes, class_labels=labels)
                    aug_img = augmented['image']
                    aug_bboxes = augmented['bboxes']
                    aug_labels = augmented['class_labels']

                    if not aug_bboxes:  # Skip if augmentation removed all bboxes
                        continue

                    # Save augmented image and annotation
                    aug_filename = f"{base_name}_aug_{aug_idx}{ext}"
                    aug_img_path = os.path.join(img_out_dir, aug_filename)
                    aug_label_path = os.path.join(label_out_dir, f"{base_name}_aug_{aug_idx}.txt")

                    cv2.imwrite(aug_img_path, aug_img)
                    with open(aug_label_path, 'w') as f:
                        for bbox, label in zip(aug_bboxes, aug_labels):
                            f.write(f"{label} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")

                except Exception as e:
                    print(f"⚠️ Augmentation error for {img_path} (aug {aug_idx}): {e}")
                    continue

            processed_count += 1
            if processed_count % 10 == 0:
                print(f"Processed {processed_count}/{len(pair_list)} files for {split}")

        print(f"✅ Completed {split}: {processed_count} files processed")

    print(f"✅ Augmented dataset saved to: {output_dir}")

# Usage example:
if __name__ == "__main__":
    input_dir = "data/frames"
    annotation_dir = "data/annotations"
    output_dir = "data/augmented"

    augment_images(input_dir, annotation_dir, output_dir, augment_count=3)
