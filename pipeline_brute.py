import cv2
import numpy as np
import os
import json

with open("data/input.json", "r") as f:
    input_data = json.load(f)

image_paths = input_data["image_files"]

error = []

# Create output root directory
output_dir = "brute_output"
os.makedirs(output_dir, exist_ok=True)

for image_num, image_path in enumerate(image_paths):
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    image_output_dir = os.path.join(output_dir, image_name)
    os.makedirs(image_output_dir, exist_ok=True)

    image = cv2.imread(image_path)
    if image is None:
        error.append(f"Error: Failed to read image {image_path}")
        continue

    # Save original
    cv2.imwrite(f"{image_output_dir}/original.jpg", image)

    # Convert to grayscale and blur slightly
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Morphological operations to clean up edges
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)

    # Save edge image
    cv2.imwrite(f"{image_output_dir}/edges.jpg", edges)

    # Detect table contour
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        error.append(f"Error: {image_path} has no table contour")
        continue

    table_contour = max(contours, key=cv2.contourArea)

    chessboard_mask = np.zeros_like(image)
    cv2.drawContours(chessboard_mask, [table_contour], -1, (255, 255, 255), thickness=cv2.FILLED)
    masked_image = cv2.bitwise_and(image, chessboard_mask)

    # Save masked table
    cv2.imwrite(f"{image_output_dir}/table_masked.jpg", masked_image)
    '''
    # Detect chessboard mask (reusing HSV-based filtering inside table)
    hsv = cv2.cvtColor(masked_image, cv2.COLOR_BGR2HSV)
    lower_brown = np.array([0, 22, 107])
    upper_brown = np.array([30, 255, 255])
    table_color_mask = cv2.inRange(hsv, lower_brown, upper_brown)
    chessboard_mask = cv2.bitwise_not(table_color_mask)

    kernel = np.ones((3, 3), np.uint8)
    chessboard_mask = cv2.morphologyEx(chessboard_mask, cv2.MORPH_CLOSE, kernel)
    chessboard_mask = cv2.morphologyEx(chessboard_mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # Save chessboard mask
    cv2.imwrite(f"{image_output_dir}/chessboard_mask.jpg", chessboard_mask)
    '''
    # Filter contours
    # Ensure chessboard_mask is a single-channel binary image
    if len(chessboard_mask.shape) == 3:
        chessboard_mask = cv2.cvtColor(chessboard_mask, cv2.COLOR_BGR2GRAY)

    contours_, _ = cv2.findContours(chessboard_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = chessboard_mask.shape[:2]

    def is_touching_border(contour, width, height):
        for point in contour:
            x, y = point[0]
            if x <= 1 or y <= 1 or x >= width - 2 or y >= height - 2:
                return True
        return False

    contours = [cnt for cnt in contours_ if not is_touching_border(cnt, width, height)]

    if not contours:
        error.append(f"Error: {image_path} has no valid chessboard contours")
        continue

    try:
        chessboard_contour = max(contours, key=cv2.contourArea)
    except ValueError:
        error.append(f"Error: {image_path} has no contours after filtering")
        continue

    # Save chessboard contour
    contour_image = masked_image.copy()
    cv2.drawContours(contour_image, [chessboard_contour], -1, (0, 255, 0), 20)
    cv2.imwrite(f"{image_output_dir}/chessboard_contour.jpg", contour_image)

    # Approximate corners
    approx = cv2.approxPolyDP(chessboard_contour, 0.06 * cv2.arcLength(chessboard_contour, True), True)

    if len(approx) != 4:
        error.append(f"Error: {image_path} has {len(approx)} corners instead of 4")
        continue

    corners = approx.reshape(4, 2)
    corners = corners[np.argsort(corners[:, 0])]

    left_corners = corners[:2]
    right_corners = corners[2:]

    top_left = left_corners[np.argmin(left_corners[:, 1])]
    bottom_left = left_corners[np.argmax(left_corners[:, 1])]
    top_right = right_corners[np.argmin(right_corners[:, 1])]
    bottom_right = right_corners[np.argmax(right_corners[:, 1])]

    ordered_corners = [top_left, top_right, bottom_right, bottom_left]

    # Save corner visualization
    corner_image = masked_image.copy()
    for corner in ordered_corners:
        cv2.circle(corner_image, tuple(corner), 40, (255, 0, 0), -1)
    cv2.imwrite(f"{image_output_dir}/ordered_corners.jpg", corner_image)

    # Perspective warp
    output_size = 3024
    dst = np.array([
        [0, 0],
        [output_size - 1, 0],
        [output_size - 1, output_size - 1],
        [0, output_size - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(np.array(ordered_corners, dtype="float32"), dst)
    warped = cv2.warpPerspective(image, M, (output_size, output_size))

    # Save final warped image
    cv2.imwrite(f"{image_output_dir}/warped.jpg", warped)

# Save error log
with open(os.path.join(output_dir, "error.txt"), "w") as f:
    for item in error:
        f.write("%s\n" % item)
