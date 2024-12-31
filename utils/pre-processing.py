import os
import cv2
import argparse
from ultralytics import YOLO
from utils import (
    create_keypoint_heatmap,
    create_bbox_keypoint_heatmap,
    adjust_bbox_scale,
    blend_image_with_heatmap,
    find_largest_object_id,
    find_largest_object_excluding_right,
    calculate_bbox_keypoint_difference,
)

def process_pose_data(input_folder, output_folder, model, save_options=None):
    if save_options is None:
        save_options = ['crop', 'keypoints']

    for cam in ['Camera2', 'Camera1']:
        cam_input_path = os.path.join(input_folder, cam)
        cam_output_path = os.path.join(output_folder, cam)
        os.makedirs(cam_output_path, exist_ok=True)

        for label_name in os.listdir(cam_input_path):
            label_input_path = os.path.join(cam_input_path, label_name)
            label_output_path = os.path.join(cam_output_path, label_name)

            if os.path.exists(label_output_path) and os.listdir(label_output_path):
                print(f"Skipping already processed folder: {label_output_path}")
                continue

            os.makedirs(label_output_path, exist_ok=True)
            filenames = sorted([f for f in os.listdir(label_input_path) if f.endswith('.png')])

            processed_count = 0
            object_id = None

            for filename in filenames:
                if processed_count >= 60:
                    break

                img_path = os.path.join(label_input_path, filename)
                img_output_path = os.path.join(label_output_path, filename)

                if not os.path.exists(img_path):
                    print(f"Image not found: {img_path}")
                    continue

                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"Failed to load image: {img_path}")
                        continue

                    results = model.track(source=img_path, persist=True, conf=0.4)

                    if object_id is None:
                        if cam == 'Camera1':
                            object_id = find_largest_object_id(results)
                        elif cam == 'Camera2':
                            object_id = find_largest_object_excluding_right(results, img.shape)

                    if object_id is None:
                        continue

                    best_box = None
                    best_keypoints = None
                    min_diff = float('inf')

                    for result in results:
                        for box, keypoints in zip(result.boxes, result.keypoints):
                            if box.id == object_id:
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                diff = calculate_bbox_keypoint_difference(keypoints, [x1, y1, x2, y2])
                                if diff < min_diff:
                                    min_diff = diff
                                    best_box = box
                                    best_keypoints = keypoints

                    if best_box is not None and best_keypoints is not None:
                        x1, y1, x2, y2 = map(int, best_box.xyxy[0])

                        if 'original' in save_options:
                            img_resized = cv2.resize(img, (60, 60))
                            cv2.imwrite(os.path.join(label_output_path, f'original_{filename}'), img_resized)

                        if 'crop' in save_options:
                            cropped_img = img[y1:y2, x1:x2]
                            cropped_img_resized = cv2.resize(cropped_img, (60, 60))
                            cv2.imwrite(os.path.join(label_output_path, f'crop_{int(best_box.id)}_{filename}'), cropped_img_resized)

                        if 'heatmap' in save_options:
                            heatmap_img = create_keypoint_heatmap(best_keypoints, img.shape)
                            cv2.imwrite(os.path.join(label_output_path, f'heatmap_{int(best_box.id)}_{filename}'), heatmap_img)

                        if 'bbox_heatmap' in save_options:
                            bbox_heatmap_img = create_bbox_keypoint_heatmap(best_keypoints, [x1, y1, x2, y2])
                            cv2.imwrite(os.path.join(label_output_path, f'bbox_heatmap_{int(best_box.id)}_{filename}'), bbox_heatmap_img)

                        if 'keypoints' in save_options:
                            keypoints_list = [[kp[0].item() / img.shape[1], kp[1].item() / img.shape[0], kp[2].item()] for kp in best_keypoints]
                            with open(os.path.join(label_output_path, f'keypoints_{filename[:-4]}.txt'), 'a') as f:
                                f.write(f'{int(best_box.id)}, {keypoints_list}\n')

                    processed_count += 1

                except Exception as e:
                    print(f"Error processing {img_path}: {e}")

    print("Files have been processed successfully.")

def main():
    parser = argparse.ArgumentParser(description="Process pose data from UP-FALL dataset.")
    parser.add_argument('--input_folder', type=str, required=True, help="Path to the input folder.")
    parser.add_argument('--output_folder', type=str, required=True, help="Path to the output folder.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the YOLOv8 model.")
    parser.add_argument('--save_options', type=str, nargs='+', default=['crop', 'keypoints'],
                        choices=['original', 'crop', 'heatmap', 'bbox_heatmap', 'keypoints'],
                        help="List of outputs to save.")

    args = parser.parse_args()

    model = YOLO(args.model_path)
    process_pose_data(args.input_folder, args.output_folder, model, args.save_options)

if __name__ == "__main__":
    main()