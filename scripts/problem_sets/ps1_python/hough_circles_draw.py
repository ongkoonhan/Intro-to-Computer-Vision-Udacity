import cv2
import numpy as np


def hough_circles_draw(img, peaks_arr, radius, radius_type):
    # Draw lines found in an image using Hough transform.

    # img: Image on top of which to draw lines
    # peaks: Qx2 matrix containing row, column indices of the Q peaks found in accumulator
    # radius: radius used to generate Circle Hough Transform


    img = img.copy()


    ### Checks

    radius_type_list = ["numeric", "array"]
    assert radius_type in radius_type_list, \
        "radius_type must be one of the following: {0}".format(radius_type_list)

    if radius_type == "array":
        assert len(peaks_arr) == len(radius), \
            "peaks_arr and radius must have the same length for radius_type: 'array'"


    ### Draw Circles

    if radius_type == "numeric":
        for b_idx, a_idx in peaks_arr:
            r = radius
            center_pt = (a_idx, b_idx,)
            img = cv2.circle(img, center_pt, r, (0, 255, 0,))   # Draw line from 2 pts

    elif radius_type == "array":
        for centers_idx, r in zip(peaks_arr, radius):
            b_idx, a_idx = centers_idx
            center_pt = (a_idx, b_idx,)
            img = cv2.circle(img, center_pt, r, (0, 255, 0,))   # Draw line from 2 pts


    return img

















