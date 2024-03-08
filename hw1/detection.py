import os
from unittest import result
import cv2
import utils
import numpy as np
import matplotlib.pyplot as plt


def detect(dataPath, clf):
    """
    Please read detectData.txt to understand the format. Load the image and get
    the face images. Transfer the face images to 19 x 19 and grayscale images.
    Use clf.classify() function to detect faces. Show face detection results.
    If the result is True, draw the green box on the image. Otherwise, draw
    the red box on the image.
      Parameters:A
        dataPath: the path of detectData.txt
      Returns:
        No returns.
    """
    # Begin your code (Part 4)
    dataPath = "data/detect/detectData.txt"
    with open(dataPath, 'r') as file:
        line_list = [line.rstrip().split() for line in file]

    line_idx = 0
    while line_idx < len(line_list):
        image = cv2.imread(os.path.join("data/detect", line_list[line_idx][0]))
        img_gray = cv2.cvtColor(image, cv2.IMREAD_GRAYSCALE)
        num_faces = int(line_list[line_idx][1])

        # Crop face region using the ground truth label
        box_list = []
        for i in range(num_faces):
            # get boxes
            x, y = int(line_list[line_idx + 1 + i][0]), int(line_list[line_idx + 1 + i][1])
            w, h = int(line_list[line_idx + 1 + i][2]), int(line_list[line_idx + 1 + i][3])
            left_top = (max(x, 0), max(y, 0))
            right_bottom = (min(x + w, img_gray.shape[1]), min(y + h, img_gray.shape[0]))
            img_crop = img_gray[left_top[1]:right_bottom[1], left_top[0]:right_bottom[0]].copy()

            # classify
            if clf.classify(img_crop) == 1:
                box_list.append((left_top, right_bottom, 1))
            else:
                box_list.append((left_top, right_bottom, 0))

        for left_top, right_bottom, label in box_list:
            if label == 1:
                cv2.rectangle(image, left_top, right_bottom, (0, 255, 0), 2)
            else:
                cv2.rectangle(image, left_top, right_bottom, (255, 0, 0), 2)

        line_idx += num_faces + 1

    # End your code (Part 4)
