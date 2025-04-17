import cv2
import numpy as np
import os
import json

def is_touching_border(contour, width, height):
    for point in contour:
        x, y = point[0]
        if x <= 1 or y <= 1 or x >= width - 2 or y >= height - 2:
            return True
    return False

def warp_perspective(image, corners, output_size=3024):
    dst = np.array([
        [0, 0],
        [output_size - 1, 0],
        [output_size - 1, output_size - 1],
        [0, output_size - 1]
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(np.array(corners, dtype="float32"), dst)
    return cv2.warpPerspective(image, M, (output_size, output_size))

def order_corners(corners):
    corners = corners[np.argsort(corners[:, 0])]
    left = corners[:2]
    right = corners[2:]

    top_left = left[np.argmin(left[:, 1])]
    bottom_left = left[np.argmax(left[:, 1])]
    top_right = right[np.argmin(right[:, 1])]
    bottom_right = right[np.argmax(right[:, 1])]

    return [top_left, top_right, bottom_right, bottom_left]

def find_largest_valid_contour(mask, image_path):
    contours_, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = mask.shape[:2]
    valid_contours = [cnt for cnt in contours_ if not is_touching_border(cnt, width, height)]
    if not valid_contours:
        raise ValueError(f"No valid contours found in {image_path}")
    return max(valid_contours, key=cv2.contourArea)

def extract_approx_quad(contour, image_path):
    approx = cv2.approxPolyDP(contour, 0.06 * cv2.arcLength(contour, True), True)
    if len(approx) != 4:
        raise ValueError(f"{image_path} has {len(approx)} corners instead of 4")
    return order_corners(approx.reshape(4, 2))

# === PIPELINE A ===
def extract_chessboard_pipeline_a(image, image_path):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_brown = np.array([0, 22, 107])
    upper_brown = np.array([30, 255, 255])
    table_mask = cv2.inRange(hsv, lower_brown, upper_brown)
    table_mask = cv2.morphologyEx(table_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    table_mask = cv2.morphologyEx(table_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError(f"No table contours found in {image_path}")

    table_contour = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(image)
    cv2.drawContours(mask, [table_contour], -1, (255, 255, 255), cv2.FILLED)
    masked_image = cv2.bitwise_and(image, mask)

    hsv = cv2.cvtColor(masked_image, cv2.COLOR_BGR2HSV)
    table_color_mask = cv2.inRange(hsv, lower_brown, upper_brown)
    chessboard_mask = cv2.bitwise_not(table_color_mask)
    chessboard_mask = cv2.morphologyEx(chessboard_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    chessboard_mask = cv2.morphologyEx(chessboard_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)

    chessboard_contour = find_largest_valid_contour(chessboard_mask, image_path)
    corners = extract_approx_quad(chessboard_contour, image_path)
    return warp_perspective(image, corners)

# === PIPELINE B ===
def extract_chessboard_pipeline_b(image, image_path):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)
    edges = cv2.erode(edges, np.ones((5, 5), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError(f"No table contours in {image_path}")

    table_contour = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(image)
    cv2.drawContours(mask, [table_contour], -1, (255, 255, 255), cv2.FILLED)
    mask = cv2.dilate(mask, np.ones((75, 75), np.uint8), iterations=1)
    masked_image = cv2.bitwise_and(image, mask)

    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    chessboard_contour = find_largest_valid_contour(mask, image_path)
    corners = extract_approx_quad(chessboard_contour, image_path)
    return warp_perspective(image, corners)

# === MAIN EXECUTION ===
with open("data/input.json", "r") as f:
    image_paths = json.load(f)["image_files"]

output_dir = "brute_output"
os.makedirs(output_dir, exist_ok=True)
error_log = []

for idx, path in enumerate(image_paths):
    image = cv2.imread(path)
    if image is None:
        error_log.append(f"Failed to read image: {path}")
        continue

    try:
        warped = extract_chessboard_pipeline_a(image, path)
        method = "A"
    except Exception as e_a:
        print(f"Pipeline A failed for {path}: {e_a}")
        try:
            warped = extract_chessboard_pipeline_b(image, path)
            method = "B"
        except Exception as e_b:
            error_log.append(f"Both pipelines failed for {path}: A: {e_a} | B: {e_b}")
            continue

    filename = os.path.splitext(os.path.basename(path))[0]
    out_path = os.path.join(output_dir, f"{filename}_warped_{method}.jpg")
    cv2.imwrite(out_path, warped)
    print(f"Saved: {out_path} using Pipeline {method}")

with open(os.path.join(output_dir, "error.txt"), "w") as f:
    for err in error_log:
        f.write(f"{err}\n")
