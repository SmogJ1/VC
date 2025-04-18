import cv2
import numpy as np
import os
import json
from matplotlib import pyplot as plt
from collections import defaultdict

def get_list(json_path):

    with open(json_path, "r") as f:
        input_data = json.load(f)

    return input_data["image_files"]

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
    warped = cv2.warpPerspective(image, M, (output_size, output_size))

    return warped, M

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
def extract_chessboard_pipeline_a(image, image_path, image_num=-1):
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

    copy = masked_image.copy()
    for corner in corners:
        cv2.circle(copy, tuple(corner), 10, (0, 255, 0), -1)

    if image_num > -1:
        path = "debug/" + str(image_num) + "/masked_image.png"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, masked_image)
        path = "debug/" + str(image_num) + "/chessboard_contour.png"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, chessboard_contour)
        path = "debug/" + str(image_num) + "/corners.png"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, copy)


    warped, M = warp_perspective(image, corners)
    return warped, M

# === PIPELINE B ===
def extract_chessboard_pipeline_b(image, image_path, image_num=-1):
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

    copy = masked_image.copy()
    for corner in corners:
        cv2.circle(copy, tuple(corner), 10, (0, 255, 0), -1)

    if image_num > -1:
        path = "debug/" + str(image_num) + "/masked_image.png"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, masked_image)
        path = "debug/" + str(image_num) + "/chessboard_contour.png"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, chessboard_contour)
        path = "debug/" + str(image_num) + "/corners.png"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, copy)


    warped, M = warp_perspective(image, corners)
    return warped, M


def get_warped_image(image, path, image_num=-1):


    try:
        warped, M = extract_chessboard_pipeline_a(image, path, image_num)
    except Exception as e_a:
        print(f"Pipeline A failed for {path}: {e_a}")
        try:
            warped, M = extract_chessboard_pipeline_b(image, path, image_num)
        except Exception as e_b:
            
            print(f"Pipeline B failed for {path}: {e_b}")
            
    if image_num > -1:
        path = "debug/" + str(image_num) + "/warped_image.png"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, warped)

    return warped, M


def get_knight_position(warped, image_num=-1):
    knight_template = cv2.imread("knight.png", cv2.IMREAD_GRAYSCALE)  # Your knight symbol

    chessboard_img= cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    chessboard_color = warped.copy()

    # Color versions for display
    knight_template_color = cv2.imread("knight.png")


    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Find keypoints and descriptors
    kp1, des1 = sift.detectAndCompute(knight_template, None)
    kp2, des2 = sift.detectAndCompute(chessboard_img, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # Higher = more accurate but slower

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Perform matching
    matches = flann.knnMatch(des1, des2, k=2)

    # Apply ratio test (Lowe's criteria)
    good_matches = []
    ratio_threshold = 0.7  # Adjust this if needed
    for m, n in matches:
        if m.distance < ratio_threshold * n.distance:
            good_matches.append(m)

    # Draw matches
    match_output = cv2.drawMatches(
        knight_template_color, kp1, 
        chessboard_color, kp2,
        good_matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        matchColor=(0, 255, 0)  # Green matches
    )

    # Find the knight location (average of good matches)
    if len(good_matches) > 0:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Calculate average position
        knight_position = np.median(dst_pts, axis=0)[0]
        
        # Draw a circle at the found position
        cv2.circle(chessboard_color, (int(knight_position[0]), int(knight_position[1])), 
                30, (0, 0, 255), 3)
        cv2.putText(chessboard_color, "Knight", 
                (int(knight_position[0]) + 40, int(knight_position[1])),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
    else:
        knight_position = None
        print("No good matches found.")

    if (image_num>-1):
        path = "debug/" + str(image_num) + "/knight_matches.png"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, match_output)
        path = "debug/" + str(image_num) + "/knight_position.png"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, chessboard_color)

    return knight_position


def rotate_knight_to_bottom_left(image, knight_pos):
    h, w = image.shape[:2]
    cx, cy = w // 2, h // 2
    
    x, y = knight_pos 

    print(f"Knight position: ({x}, {y}) | Center: ({cx}, {cy})")

    if x < cx and y < cy:
        angle = 90  
    elif x >= cx and y < cy:
        angle = 180 
    elif x >= cx and y >= cy:
        angle = 270 
    else:
        angle = 0  

    print(f"Rotating {angle}º to place the knight in the bottom-left corner")

    if angle == 90:
        rotated = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif angle == 180:
        rotated = cv2.rotate(image, cv2.ROTATE_180) 
    elif angle == 270:
        rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE) 
    else:
        rotated = image.copy() 

    return rotated, angle


def rotate_board(warped, knight_position, image_num=-1):

    rotated_chessboard, angle = rotate_knight_to_bottom_left(warped, knight_position)
    
    if (image_num>-1):
        path = "debug/" + str(image_num) + "/rotated_chessboard.png"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, rotated_chessboard)

    return angle






def get_cropped_image(image, padding=0, image_num=-1):
    h, w = image.shape[:2]
    top = bottom = left = right = padding

    if padding > 0:
        top = bottom = left = right = padding

    cropped_image = image[top:h-bottom, left:w-right]

    if (image_num>-1):
        path = "debug/" + str(image_num) + "/cropped_image.png"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, cropped_image)




    return cropped_image



def get_grid(image, image_num=-1):
    rows, cols = 8, 8

    image_copy = image.copy()
    h, w = image.shape[:2]
    cell_height = h // rows
    cell_width = w // cols

    cell_coords = []

    for i in range(rows):
        for j in range(cols):
            x1, y1 = j * cell_width, i * cell_height
            x2, y2 = (j + 1) * cell_width, (i + 1) * cell_height

            cell_coords.append({
                "row": i,
                "col": j,
                "bbox": (x1, y1, x2, y2)
            })

            cv2.rectangle(image_copy, (x1, y1), (x2, y2), (0, 0, 255), 10) 
            cv2.putText(image_copy, f"{i},{j}", (x2 - 120, y2 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2, cv2.LINE_AA) 
            
    if (image_num>-1):
        path = "debug/" + str(image_num) + "/grid.png"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, image_copy)

    return cell_coords

def detect_single_piece_per_cell(image, grid_info, central_ratio=0.45, image_num=-1):

    piece_positions = []
    image_copy = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    black_range = ((0, 0, 0), (180, 255, 50))
    cream_range = ((15, 30, 130), (40, 150, 255))

    black_count = 0
    cream_count = 0

    for cell in grid_info:
        x1, y1, x2, y2 = cell["bbox"]
        row, col = cell["row"], cell["col"]

        w = x2 - x1
        h = y2 - y1
        margin_x = int((1 - central_ratio) / 2 * w)
        margin_y = int((1 - central_ratio) / 2 * h)
        x1_c = x1 + margin_x
        y1_c = y1 + margin_y
        x2_c = x2 - margin_x
        y2_c = y2 - margin_y

        cell_hsv_patch = hsv[y1_c:y2_c, x1_c:x2_c]
        total_pixels = cell_hsv_patch.shape[0] * cell_hsv_patch.shape[1]

        mask_black = cv2.inRange(cell_hsv_patch, *black_range)
        mask_cream = cv2.inRange(cell_hsv_patch, *cream_range)

        count_black_pixels = cv2.countNonZero(mask_black)
        count_cream_pixels = cv2.countNonZero(mask_cream)
        occupied_pixels = count_black_pixels + count_cream_pixels

        occupied_ratio = occupied_pixels / total_pixels
        free_ratio = 1 - occupied_ratio

        if free_ratio >= 0.72:
            continue

        cell_patch = gray[y1:y2, x1:x2]
        blurred = cv2.GaussianBlur(cell_patch, (3, 3), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        has_piece = False
        mid_area_detected = False
        top_feature_detected = False

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 1 and area < w * h:
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                point_from_bottom = False
                point_from_middle = False

                for point in cnt:
                    px, py = point[0]
                    px_global = px + x1
                    py_global = py + y1
                    if py <= 170:
                        point_from_middle = True
                    if py >= h - 40:
                        point_from_bottom = True
                    if point_from_bottom and point_from_middle:
                        continue
                    if 20 < py < 60 and 40 < px < 280:
                        top_feature_detected = True
                    if (cx - 80 <= px_global <= cx + 80) and (cy - 120 <= py_global <= cy + 80):
                        mid_area_detected = True
                        break

                if mid_area_detected and top_feature_detected:
                    has_piece = True
                    break

        if has_piece:  
            cv2.rectangle(image_copy, (x1, y1), (x2, y2), (0, 255, 0), 50)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(image_copy, (cx, cy), 3, (0, 0, 255), 40)

            if count_black_pixels > count_cream_pixels:
                piece_positions.append({"row": row, "col": col, "color": "Black"})
                black_count += 1
            else:
                piece_positions.append({"row": row, "col": col, "color": "White"})
                cream_count += 1
        
        matrix = np.zeros((8, 8), dtype=int)

        for piece in piece_positions:
                matrix[piece["row"], piece["col"]] = 1

        if image_num > -1:
            path = "debug/" + str(image_num) + "/detected_pieces.png"
            os.makedirs(os.path.dirname(path), exist_ok=True)
            cv2.imwrite(path, image_copy)
            print(f"Black pieces: {black_count}")
            print(f"White pieces: {cream_count}")

            for piece in piece_positions:
                print(f"Piece found at ({piece['row']}, {piece['col']}) - Color: {piece['color']}")


    return image_copy, piece_positions, black_count, cream_count, matrix



def rotate_matrix(matrix, rotation_angle, image_num=-1):

    if image_num > -1:
        print(f"Original matrix for image {image_num}:\n{matrix}")

    if rotation_angle == 90:
        matrix = np.rot90(matrix, k=1)
    elif rotation_angle == 180:
        matrix = np.rot90(matrix, k=2)
    elif rotation_angle == 270:
        matrix = np.rot90(matrix, k=3)

    if image_num > -1:
        print(f"Rotated matrix for image {image_num}:\n{matrix}")

    
    return matrix

def mirror_matrix(matrix, image_num=-1):
    matrix = np.flipud(matrix)

    if image_num > -1:
        print(f"Mirrored matrix for image {image_num}:\n{matrix}")

    return matrix

def create_piece_mask(img, lower_hsv, upper_hsv):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, lower_hsv, upper_hsv)

def clean_mask(mask, kernel_size=(5, 5), open_iter=2, close_iter=1):
    kernel = np.ones(kernel_size, np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=open_iter)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=close_iter)
    return mask

def preprocess_for_segmentation(img, blur_kernel=(11, 11)):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v_blurred = cv2.GaussianBlur(v, blur_kernel, 0)
    return cv2.merge([h, s, v_blurred])

def get_piece_masks(image, image_num=-1):
    WHITE_LOWER = np.array([10, 30, 100])
    WHITE_UPPER = np.array([40, 180, 255])

    BLACK_LOWER = np.array([0, 0, 0])
    BLACK_UPPER = np.array([180, 255, 40])

    smoothed = cv2.cvtColor(preprocess_for_segmentation(image), cv2.COLOR_HSV2BGR)

    white_mask = clean_mask(create_piece_mask(smoothed, WHITE_LOWER, WHITE_UPPER))
    black_mask = clean_mask(create_piece_mask(smoothed, BLACK_LOWER, BLACK_UPPER))

    if image_num > -1:
        path = "debug/" + str(image_num) + "/white_mask.png"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, white_mask)
        path = "debug/" + str(image_num) + "/black_mask.png"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, black_mask)

    return white_mask, black_mask

# --- Detect Stacked Pieces ---
def detect_stacked_pieces(pieces):
    col_to_rows = defaultdict(list)
    for piece in pieces:
        col_to_rows[piece["col"]].append(piece["row"])

    stacked = set()
    for col, rows in col_to_rows.items():
        rows.sort()
        for i in range(len(rows) - 1):
            if rows[i + 1] == rows[i] + 1:
                stacked.add((rows[i], col))
                stacked.add((rows[i + 1], col))
    return stacked

# --- Detect Pieces Using Contours ---
def detect_pieces_using_contours(mask, min_area=25000, dilate_iter=2):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fused_mask = np.zeros_like(mask)
    cv2.drawContours(fused_mask, contours, -1, 255, thickness=cv2.FILLED)

    kernel = np.ones((5, 5), np.uint8)
    fused_mask = cv2.dilate(fused_mask, kernel, iterations=dilate_iter)

    final_contours, _ = cv2.findContours(fused_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in final_contours:
        if cv2.contourArea(cnt) > min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append([x, y, x + w, y + h])
    return boxes

# --- Fallback for Undetected Pieces ---
def fallback_for_piece(piece, grid_info, w, h):
    cell = next(c for c in grid_info if c["row"] == piece["row"] and c["col"] == piece["col"])
    x1, y1, x2, y2 = cell["bbox"]
    return [
        max(0, x1 + 50),
        max(0, y1 - 200),
        min(w, x2 - 50),
        min(h, y2 - 50)
    ]

# --- Get The Bounding Boxes using the auxiliary functions (as this is a big and kinda complex function, we added comments to explain it) ---
def get_clean_bounding_boxes(mask, pieces=None, grid_info=None, min_area=25000, dilate_iter=2):
    h, w = mask.shape

    # Identify which pieces are stacked (same column, consecutive rows)
    stacked = detect_stacked_pieces(pieces) if pieces else set()

    # Detect bounding boxes using contour detection and dilation
    contour_boxes = detect_pieces_using_contours(mask, min_area, dilate_iter)

    # Initialize a dictionary to track which pieces have been detected
    detected = {(p["row"], p["col"]): False for p in pieces}

    final_boxes = []

    # Loop through each detected bounding box
    for bx in contour_boxes:
        x1b, y1b, x2b, y2b = bx
        overlaps_valid = False   # True if this box overlaps with a valid (non-stacked) piece
        discard_box = False      # True if this box overlaps with a stacked piece (we want to avoid that)

        # Check overlap with each piece on the board
        for p in pieces:
            coord = (p["row"], p["col"])
            # Find the corresponding grid cell
            cell = next(c for c in grid_info if c["row"] == p["row"] and c["col"] == p["col"])
            x1, y1, x2, y2 = cell["bbox"]

            # Calculate intersection area between bounding box and cell
            inter_area = max(0, min(x2, x2b) - max(x1, x1b)) * max(0, min(y2, y2b) - max(y1, y1b))
            cell_area = (x2 - x1) * (y2 - y1)

            # If there's significant overlap (>5% of the cell area), evaluate the match
            if inter_area / (cell_area + 1e-5) > 0.05:
                if coord in stacked:
                    # If it overlaps with a stacked piece, mark it for discard
                    discard_box = True
                    break
                else:
                    # Otherwise, it's a valid match
                    overlaps_valid = True
                    detected[coord] = True

        # If the box is valid and not discarded, keep it
        if overlaps_valid and not discard_box:
            final_boxes.append(bx)

    # Add fallback boxes for pieces that were not detected or are stacked
    for p in pieces:
        coord = (p["row"], p["col"])
        if coord in stacked or not detected[coord]:
            final_boxes.append(fallback_for_piece(p, grid_info, w, h))

    return final_boxes



def merge_close_bounding_boxes(boxes, max_distance=125):
    if not boxes:
        return []

    boxes = [np.array(box) for box in boxes]
    used = [False] * len(boxes)
    merged = []

    for i, box in enumerate(boxes):
        if used[i]:
            continue
        x1, y1, x2, y2 = box
        current = box.copy()
        used[i] = True

        for j in range(i + 1, len(boxes)):
            if used[j]:
                continue
            xx1, yy1, xx2, yy2 = boxes[j]
            if (abs(x1 - xx1) < max_distance or abs(x2 - xx2) < max_distance) and \
               (abs(y1 - yy1) < max_distance or abs(y2 - yy2) < max_distance):
                current[0] = min(current[0], xx1)
                current[1] = min(current[1], yy1)
                current[2] = max(current[2], xx2)
                current[3] = max(current[3], yy2)
                used[j] = True

        merged.append(current.tolist())

    return merged


def remove_weird_shapes(boxes, max_ratio=1.3):
    return [
        box for box in boxes
        if (box[2] - box[0]) <= (box[3] - box[1]) * max_ratio
    ]

# --- Pre-processing ---

def get_bounding_boxes(grid_info, white_mask, black_mask, pieces, warped, image_num=-1):

    grid_info_with_padding = [
        {
            "row": cell["row"],
            "col": cell["col"],
            "bbox": tuple(v + 220 for v in cell["bbox"])
        }
        for cell in grid_info
    ]

    white_pieces = [p for p in pieces if p["color"] == "White"]
    black_pieces = [p for p in pieces if p["color"] == "Black"]

    # --- Detect bounding boxes for pieces ---

    white_boxes = get_clean_bounding_boxes(white_mask, white_pieces, grid_info_with_padding)
    white_boxes = merge_close_bounding_boxes(white_boxes)
    white_boxes = remove_weird_shapes(white_boxes)

    black_boxes = get_clean_bounding_boxes(black_mask, black_pieces, grid_info_with_padding)
    black_boxes = merge_close_bounding_boxes(black_boxes)
    black_boxes = remove_weird_shapes(black_boxes)

    # --- Draw bounding boxes on the warped image ---

    warped_copy = warped.copy()
    for box in white_boxes:
        cv2.rectangle(warped_copy, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 20)
    for box in black_boxes:
        cv2.rectangle(warped_copy, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 20)

    if image_num > -1:
        path = "debug/" + str(image_num) + "/bounding_boxes.png"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, warped_copy)


    return white_boxes, black_boxes
    
# Revert bounding boxes to original image space
def warp_boxes_back_to_original(bboxes, M_inv):
    all_pts = []
    for x1, y1, x2, y2 in bboxes:
        all_pts.extend([
            [x1, y1], [x2, y1],
            [x2, y2], [x1, y2]
        ])

    pts_array = np.array(all_pts, dtype=np.float32).reshape(-1, 1, 2)
    M_inv = M_inv.astype(np.float32)

    warped_back = cv2.perspectiveTransform(pts_array, M_inv)

    recovered_boxes = []
    for i in range(0, len(warped_back), 4):
        pts = warped_back[i:i + 4].reshape(-1, 2)
        xs, ys = pts[:, 0], pts[:, 1]
        recovered_boxes.append([
            int(xs.min()), int(ys.min()),
            int(xs.max()), int(ys.max())
        ])

    return recovered_boxes

# Shrink box width to avoid excessive width
def shrink_box_width(box, factor=0.60):
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2
    new_half_width = (x2 - x1) * factor / 2
    return [int(cx - new_half_width), y1, int(cx + new_half_width), y2]

def revert_image(image, white_boxes, black_boxes, M, image_num=-1):

    # Calculate inverse of the transformation matrix
    M_inv = np.linalg.inv(M).astype(np.float32)

    # Recover boxes in original image space
    combined_boxes = white_boxes + black_boxes
    recovered_boxes = warp_boxes_back_to_original(combined_boxes, M_inv)

    # Remove boxes that are too wide
    recovered_boxes = [shrink_box_width(box) for box in recovered_boxes]

    recovered_image = image.copy()
    for box in recovered_boxes:
        x1, y1, x2, y2 = box
        print(f"X1: {x1}, Y1: {y1}, X2: {x2}, Y2: {y2}")
        cv2.rectangle(recovered_image, (x1, y1), (x2, y2), (255, 0, 0), 10)

    if image_num > -1:
        path = "debug/" + str(image_num) + "/recovered_boxes.png"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, recovered_image)


    return recovered_boxes

def get_box_coordinates(boxes):
    boxes_list_dict = []
    for box in boxes:
        x1, y1, x2, y2 = box
        boxes_list_dict.append({
            "xmin": x1,
            "ymin": y1,
            "xmax": x2,
            "ymax": y2
        })

    return boxes_list_dict

def get_image_output(image_path, i=-1):

    results = dict()
    results["image"] = image_path

    image = cv2.imread(image_path)

    warped, M = get_warped_image(image, image_path, image_num=i)

    print(f"Image {i} warped successfully")
    #save warped image

 
    cropped = get_cropped_image(warped, padding=220, image_num=i)
    grid = get_grid(cropped, image_num=i)
    piece_image, piece_positions, black_count, cream_count, matrix = detect_single_piece_per_cell(cropped, grid, image_num=i)

    white_mask, black_mask = get_piece_masks(warped, image_num=i)
    white_boxes, black_boxes = get_bounding_boxes(grid, white_mask, black_mask, piece_positions, warped, image_num=i)
    new_boxes = revert_image(image, white_boxes, black_boxes, M, image_num=i)
    boxes_list_dict = get_box_coordinates(new_boxes)

    knight_position = get_knight_position(warped, image_num=i)
    angle = rotate_board(warped, knight_position, image_num=i)

    rotated_matrix = rotate_matrix(matrix, angle, image_num=i)
    mirrored_matrix = mirror_matrix(rotated_matrix, image_num=i)

    results["num_pieces"] = black_count + cream_count
    results["black_pieces"] = black_count
    results["white_pieces"] = cream_count
    results["board"] = mirrored_matrix.tolist()
    results["detected_pieces"] = boxes_list_dict


    return results



def main(json_path):

    final_results = []
    image_paths = get_list(json_path)

    for i in range(len(image_paths)):
        try:
            # result_dict = get_image_output(image_paths[i]) # USE THIS FOR ONLY FINAL OUTPUT
            result_dict = get_image_output(image_paths[i], i=i) # USE THIS FOR SAVING IMAGES TOO
            final_results.append(result_dict)

        except Exception as e:

            result_dict = dict()
            result_dict["image"] = image_paths[i]
            result_dict["num_pieces"] = 0
            result_dict["black_pieces"] = 0
            result_dict["white_pieces"] = 0
            board = np.zeros((8, 8), dtype=int)
            result_dict["board"] = board.tolist()
            result_dict["detected_pieces"] = []
            final_results.append(result_dict)

            # append error on image number to debug/error.txt
            error_path = "debug/error.txt"
            os.makedirs(os.path.dirname(error_path), exist_ok=True)
            with open(error_path, "a") as f:
                f.write(f"Error on image {i}: {e}\n")
            print(f"Error on image {i}: {e}")
            continue

    # save final results to json
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "final_results.json")
    with open(output_path, "w") as f:
        json.dump(final_results, f, indent=4)
    print(f"Final results saved to {output_path}")

json_path = "data/json_example_task1/input.json"
main(json_path)