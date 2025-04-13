import cv2
import numpy as np
import os
import json
from matplotlib import pyplot as plt

def get_list(json_path):

    with open(json_path, "r") as f:
        input_data = json.load(f)

    return input_data["image_files"]
       

def get_table_contour(image, image_num=-1):

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_brown = np.array([0, 22, 107])
    upper_brown = np.array([30, 255, 255])
    mask = cv2.inRange(hsv, lower_brown, upper_brown)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    table_contour = max(contours, key=cv2.contourArea)
    contour_image = image.copy()
    cv2.drawContours(contour_image, [table_contour], -1, (0, 255, 0), 20)

    if(image_num>-1):
        path = "debug/" + str(image_num) + "/table_contour.png"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, contour_image)

    return table_contour

def get_masked_image(image, table_contour, image_num=-1):

    table_mask = np.zeros_like(image)
    cv2.drawContours(table_mask, [table_contour], -1, (255, 255, 255), thickness=cv2.FILLED)
    masked_image = cv2.bitwise_and(image, table_mask)

    if(image_num>-1):
        path = "debug/" + str(image_num) + "/masked_image.png"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, masked_image)

    return masked_image

def get_contours(masked_image, image_num=-1):
    hsv = cv2.cvtColor(masked_image, cv2.COLOR_BGR2HSV)

    lower_brown = np.array([0, 22, 107])
    upper_brown = np.array([30, 255, 255])
    table_color_mask = cv2.inRange(hsv, lower_brown, upper_brown)
    chessboard_mask = cv2.bitwise_not(table_color_mask)

    kernel = np.ones((3, 3), np.uint8)
    chessboard_mask = cv2.morphologyEx(chessboard_mask, cv2.MORPH_CLOSE, kernel)
    chessboard_mask = cv2.morphologyEx(chessboard_mask, cv2.MORPH_OPEN, kernel, iterations=2)

    contours_, _ = cv2.findContours(chessboard_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    contour_image = masked_image.copy()
    for contour in contours_:
        random_color = tuple(np.random.randint(0, 255, 3).tolist())
        cv2.drawContours(contour_image, [contour], -1, random_color, 30)

    if(image_num>-1):
        path = "debug/" + str(image_num) + "/contours.png"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, contour_image)
        path = "debug/" + str(image_num) + "/chessboard_mask.png"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, chessboard_mask)

    return contours_


def is_touching_border(contour, width, height):
    for point in contour:
        x, y = point[0]
        if x <= 1 or y <= 1 or x >= width - 2 or y >= height - 2:
            return True
    return False

def remove_boarder_contours(contours, masked_image, image_num=-1):
    height, width = masked_image.shape[:2]
    contours = [cnt for cnt in contours if not is_touching_border(cnt, width, height)]

    contour_image = masked_image.copy()
    for contour in contours:
        cv2.drawContours(contour_image, [contour], -1, (0, 255, 0), 30)

    if(image_num>-1):
        path = "debug/" + str(image_num) + "/no_border_contours.png"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, contour_image)

    return contours

def get_chessboard_contours(image, original, contours, image_num=-1):
    chessboard_countor = max(contours, key=cv2.contourArea)
    contour_image = image.copy()
    cv2.drawContours(contour_image, [chessboard_countor], -1, (0, 255, 0), 30)

    chessboard_mask = np.zeros_like(image)
    cv2.drawContours(chessboard_mask, [chessboard_countor], -1, (255, 255, 255), thickness=cv2.FILLED)
    masked_image = cv2.bitwise_and(image, chessboard_mask)

    if(image_num>-1):
        path = "debug/" + str(image_num) + "/chessboard_contour.png"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, contour_image)
        path = "debug/" + str(image_num) + "/chessboard_only.png"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, masked_image)

    return chessboard_countor

def approx_chessboard_contour(chessboard_countor, masked_image, image_num=-1):
    approx = cv2.approxPolyDP(chessboard_countor, 0.06 * cv2.arcLength(chessboard_countor, True), True)

    # draw contours on the original image
    contour_image = masked_image.copy()
    cv2.drawContours(contour_image, [approx], -1, (0, 255, 0), 30)

    if(image_num>-1):
        path = "debug/" + str(image_num) + "/approx_chessboard_contour.png"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, contour_image)

    return approx

def get_corners(chessboard_countor, image, image_num=-1):

    corners = chessboard_countor.reshape(4, 2)
    corners = corners[np.argsort(corners[:, 0])]


    left_corners = corners[:2]

    top_left = left_corners[np.argmin(left_corners[:, 1])]
    bottom_left = left_corners[np.argmax(left_corners[:, 1])]

    right_corners = corners[2:]
    top_right = right_corners[np.argmin(right_corners[:, 1])]
    bottom_right = right_corners[np.argmax(right_corners[:, 1])]


    ordered_corners = [top_left, top_right, bottom_right, bottom_left]

    # draw contours on the original image
    corner_image = image.copy()
    for corner in ordered_corners:
        cv2.circle(corner_image, tuple(corner), 50, (255,0,0), -1)

    if(image_num>-1):
        path = "debug/" + str(image_num) + "/corners.png"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, corner_image)
    return ordered_corners

def get_warped_image(image, corners, image_num=-1):
    output_size = 3024

    dst = np.array([
            [0, 0],
            [output_size - 1, 0],
            [output_size - 1, output_size - 1],
            [0, output_size - 1]
        ], dtype="float32")

    M = cv2.getPerspectiveTransform(np.array(corners, dtype="float32"), dst)
    warped = cv2.warpPerspective(image, M, (output_size, output_size))
    
    if(image_num>-1):
        path = "debug/" + str(image_num) + "/warped_image.png"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, warped)
    return warped

def get_knight_position(warped, image_num=-1):
    knight_template = cv2.imread("knight2.png", cv2.IMREAD_GRAYSCALE)  # Your knight symbol

    chessboard_img= cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    chessboard_color = warped.copy()

    # Color versions for display
    knight_template_color = cv2.imread("knight2.png")


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

    if x < cx and y < cy:
        angle = 90  
    elif x >= cx and y < cy:
        angle = 180 
    elif x >= cx and y >= cy:
        angle = 270 
    else:
        angle = 0  


    if angle == 90:
        rotated = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif angle == 180:
        rotated = cv2.rotate(image, cv2.ROTATE_180) 
    elif angle == 270:
        rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE) 
    else:
        rotated = image.copy() 

    return rotated

def rotate_board(warped, knight_position, image_num=-1):

    rotated_chessboard = rotate_knight_to_bottom_left(warped, knight_position)
    
    if (image_num>-1):
        path = "debug/" + str(image_num) + "/rotated_chessboard.png"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, rotated_chessboard)

    return rotated_chessboard


def main(json_path):

    image_paths = get_list(json_path)

    for i in range(len(image_paths)):

        try:
            path = image_paths[i]

            # i = -1 #UNCOMMENT THIS LINE TO NOT SAVE DEBUG IMAGES

            image = cv2.imread(path)
            table_contour = get_table_contour(image, image_num=i)
            masked_image = get_masked_image(image, table_contour, image_num=i)
            countours_ = get_contours(masked_image, image_num=i)
            contours = remove_boarder_contours(countours_, masked_image, image_num=i)
            chessboard_contour_ = get_chessboard_contours(masked_image, image, contours, image_num=i)
            chessboard_contour = approx_chessboard_contour(chessboard_contour_, image, image_num=i)
            corners = get_corners(chessboard_contour,image, image_num=i)
            warped = get_warped_image(image, corners, image_num=i)
            knight_position = get_knight_position(warped, image_num=i)
            rotated_image = rotate_board(warped, knight_position, image_num=i)

        except Exception as e:
            # append error on image number to debug/error.txt
            error_path = "debug/error.txt"
            os.makedirs(os.path.dirname(error_path), exist_ok=True)
            with open(error_path, "a") as f:
                f.write(f"Error on image {i}: {e}\n")
            print(f"Error on image {i}: {e}")
            continue



json_path = "data/input.json"
main(json_path)