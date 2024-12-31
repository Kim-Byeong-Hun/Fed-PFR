import os
import cv2
import numpy as np
from ultralytics import YOLO

def create_keypoint_heatmap(keypoints, img_shape, heatmap_size=60, sigma=1):
    heatmap = np.zeros((heatmap_size, heatmap_size), dtype=np.float32)
    for kp in keypoints:
        x, y, conf = kp[0].item(), kp[1].item(), kp[2].item()
        if x == 0 and y == 0:
            continue
        x = int(x * heatmap_size / img_shape[1])
        y = int(y * heatmap_size / img_shape[0])
        for i in range(heatmap_size):
            for j in range(heatmap_size):
                heatmap[j, i] += conf * np.exp(-((i - x) ** 2 + (j - y) ** 2) / (2 * sigma ** 2))
    if heatmap.max() != 0:
        heatmap = np.clip(heatmap / heatmap.max(), 0, 1) * 255
    return cv2.applyColorMap(heatmap.astype(np.uint8), cv2.COLORMAP_JET)

def create_bbox_keypoint_heatmap(keypoints, bbox, heatmap_size=60, sigma=3):
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    heatmap = np.zeros((heatmap_size, heatmap_size), dtype=np.float32)
    for kp in keypoints:
        x, y, conf = kp[0].item(), kp[1].item(), kp[2].item()
        if x == 0 and y == 0:
            continue
        x = int((x - x1) * heatmap_size / width)
        y = int((y - y1) * heatmap_size / height)
        for i in range(heatmap_size):
            for j in range(heatmap_size):
                heatmap[j, i] += conf * np.exp(-((i - x) ** 2 + (j - y) ** 2) / (2 * sigma ** 2))
    if heatmap.max() != 0:
        heatmap = np.clip(heatmap / heatmap.max(), 0, 1) * 255
    return cv2.applyColorMap(heatmap.astype(np.uint8), cv2.COLORMAP_JET)

def adjust_bbox_scale(x1, y1, x2, y2, img_shape, scale=2):
    width = x2 - x1
    height = y2 - y1
    center_x = x1 + width // 2
    center_y = y1 + height // 2
    new_width = width * scale
    new_height = height * scale
    new_x1 = max(0, center_x - new_width // 2)
    new_y1 = max(0, center_y - new_height // 2)
    new_x2 = min(img_shape[1], center_x + new_width // 2)
    new_y2 = min(img_shape[0], center_y + new_height // 2)
    return int(new_x1), int(new_y1), int(new_x2), int(new_y2)

def blend_image_with_heatmap(image, heatmap):
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    return cv2.addWeighted(image, 0.6, heatmap_resized, 0.4, 0)

def find_largest_object_id(results):
    max_area = 0
    largest_id = -1
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area = (x2 - x1) * (y2 - y1)
            if area > max_area:
                max_area = area
                largest_id = box.id
    return largest_id.item() if largest_id != -1 else None

def find_largest_object_excluding_right(results, img_shape):
    max_area = 0
    largest_id = -1
    exclusion_boundary = img_shape[1] * 3 // 4
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if x1 > exclusion_boundary:
                continue
            area = (x2 - x1) * (y2 - y1)
            if area > max_area:
                max_area = area
                largest_id = box.id
    return largest_id.item() if largest_id != -1 else None

def calculate_bbox_keypoint_difference(keypoints, bbox):
    x1, y1, x2, y2 = bbox
    kp_min_x = min([kp[0].item() for kp in keypoints])
    kp_max_x = max([kp[0].item() for kp in keypoints])
    kp_min_y = min([kp[1].item() for kp in keypoints])
    kp_max_y = max([kp[1].item() for kp in keypoints])
    kp_width = kp_max_x - kp_min_x
    kp_height = kp_max_y - kp_min_y
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    width_diff = abs(bbox_width - kp_width)
    height_diff = abs(bbox_height - kp_height)
    return width_diff + height_diff