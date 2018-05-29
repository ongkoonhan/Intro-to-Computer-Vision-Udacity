import os
import numpy as np
import cv2 as cv2

from Intro_To_Computer_Vision_Utils import utils


input_folder = os.path.join("input")
output_folder = os.path.join("output")

show_img = True


### Q1
# (a)
img_path = os.path.join(input_folder, "ps1-input0.png")
img = cv2.imread(img_path, flags=cv2.IMREAD_GRAYSCALE)

utils.print_image_details_and_show(img, "img", show_img)

img_edges = cv2.Canny(img, 100, 200)
img_edges = np.float32(cv2.normalize(img_edges, img_edges, norm_type=cv2.NORM_MINMAX))   # Normalize to [0,1]

utils.print_image_details_and_show(img_edges, "img_edges", show_img)

















