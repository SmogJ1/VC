import cv2
import numpy as np
import os
import json
from matplotlib import pyplot as plt

with open("data/input.json", "r") as f:
    input_data = json.load(f)

image_paths = input_data["image_files"]

error = []

for image_num in range(len(image_paths)):




    image = cv2.imread(image_paths[image_num])

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_brown = np.array([0, 25, 100])
    upper_brown = np.array([30, 255, 255])

    mask = cv2.inRange(hsv, lower_brown, upper_brown)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    table_contour = max(contours, key=cv2.contourArea)

    contour_image = image.copy()
    cv2.drawContours(contour_image, [table_contour], -1, (0, 255, 0), 3)

    contour_rgb = cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB)



    table_mask = np.zeros_like(image)
    cv2.drawContours(table_mask, [table_contour], -1, (255, 255, 255), thickness=cv2.FILLED)
    masked_image = cv2.bitwise_and(image, table_mask)

    masked_rgb = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)


    hsv = cv2.cvtColor(masked_image, cv2.COLOR_BGR2HSV)

    table_color_mask = cv2.inRange(hsv, lower_brown, upper_brown)
    chessboard_mask = cv2.bitwise_not(table_color_mask)

    kernel = np.ones((3, 3), np.uint8)
    chessboard_mask = cv2.morphologyEx(chessboard_mask, cv2.MORPH_CLOSE, kernel)
    chessboard_mask = cv2.morphologyEx(chessboard_mask, cv2.MORPH_OPEN, kernel, iterations=2)

    contours_, _ = cv2.findContours(chessboard_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = chessboard_mask.shape[:2]

    def is_touching_border(contour, width, height):
        for point in contour:
            x, y = point[0]
            if x <= 1 or y <= 1 or x >= width - 2 or y >= height - 2:
                return True
        return False

    contours = [cnt for cnt in contours_ if not is_touching_border(cnt, width, height)]

    # draw contours on the original image
    contour_image = masked_image.copy()
    for contour in contours:
        cv2.drawContours(contour_image, [contour], -1, (0, 255, 0), 30)
    contour_rgb = cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB)

    try :
        chessboard_countor = max(contours, key=cv2.contourArea)
    except ValueError:
        string = f"Error: {image_paths[image_num]} of num {image_num} has no contours"
        error.append(string)
        print(string)
        continue

    # draw contours on the original image
    contour_image = masked_image.copy()
    cv2.drawContours(contour_image, [chessboard_countor], -1, (0, 255, 0), 30)
    contour_rgb = cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB)


    ## aply mask to the image
    chessboard_mask = np.zeros_like(image)
    cv2.drawContours(chessboard_mask, [chessboard_countor], -1, (255, 255, 255), thickness=cv2.FILLED)
    masked_image = cv2.bitwise_and(image, chessboard_mask)
    masked_rgb = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)

    approx = cv2.approxPolyDP(chessboard_countor, 0.06 * cv2.arcLength(chessboard_countor, True), True)

    # draw contours on the original image
    contour_image = masked_image.copy()
    cv2.drawContours(contour_image, [approx], -1, (0, 255, 0), 30)
    contour_rgb = cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB)


    if(len(approx) != 4):
        string = f"Error: {image_paths[image_num]} of num {image_num} has {len(approx)} corners"
        error.append(string)
        print(string)
        continue


    corners = approx.reshape(4, 2)
    corners = corners[np.argsort(corners[:, 0])]


    left_corners = corners[:2]

    top_left = left_corners[np.argmin(left_corners[:, 1])]
    bottom_left = left_corners[np.argmax(left_corners[:, 1])]

    right_corners = corners[2:]
    top_right = right_corners[np.argmin(right_corners[:, 1])]
    bottom_right = right_corners[np.argmax(right_corners[:, 1])]


    ordered_corners = [top_left, top_right, bottom_right, bottom_left]

    # draw contours on the original image
    corner_image = masked_image.copy()
    for corner in ordered_corners:
        cv2.circle(corner_image, tuple(corner), 50, (255,0,0), -1)

    corner_rgb = cv2.cvtColor(corner_image, cv2.COLOR_BGR2RGB)



    output_size = 3024

    dst = np.array([
            [0, 0],
            [output_size - 1, 0],
            [output_size - 1, output_size - 1],
            [0, output_size - 1]
        ], dtype="float32")

    M = cv2.getPerspectiveTransform(np.array(ordered_corners, dtype="float32"), dst)
    warped = cv2.warpPerspective(image, M, (output_size, output_size))
    warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
    

    # save the warped image
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"warped_{image_num}.jpg")
    cv2.imwrite(output_path, warped)
    
# save error list to txt

with open("output/error.txt", "w") as f:
    for item in error:
        f.write("%s\n" % item)
