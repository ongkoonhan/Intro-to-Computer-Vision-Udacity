import cv2
import numpy as np


def print_image_details_and_show(img, chart_title, show=True):
    print("Image: " + chart_title)
    print(img.dtype)
    print(img.shape)

    if len(img.shape) == 2:
        print("grayscale")
    elif len(img.shape) == 3:
        print("colour")

    print()

    if show:
        cv2.imshow(chart_title, img)
        cv2.waitKey(0)


def ROI_index_range_from_center(center_pixel, ROI_size, orig_img_size, allow_partial=False):

    ### Checks

    assert type(center_pixel) is tuple, \
        "center_pixel must be a tuple"

    assert type(ROI_size) is tuple, \
        "ROI_size must be a tuple"

    assert type(orig_img_size) is tuple, \
        "orig_img_size must be a tuple"

    assert len(center_pixel) == len(ROI_size) == len(orig_img_size), \
        "center_pixel, ROI_size, and orig_img_size must have the same size"

    for elem in ROI_size:
        assert elem % 2 != 0, \
            "ROI_size must contain odd numbers only"


    ### Get ROI index range
    range_tup_list = []
    for i in range(len(ROI_size)):   # Loop through each dim
        ROI_dim = ROI_size[i]
        orig_img_dim = orig_img_size[i]
        center_pix_loc = center_pixel[i]

        dim_l = center_pix_loc - (ROI_dim -1)//2
        dim_u = center_pix_loc + (ROI_dim -1)//2 +1

        # Allow for partial ROI at boundaries
        if allow_partial:
            if dim_l < 0:
                dim_l = 0
            if dim_u > orig_img_dim:
                dim_u = orig_img_dim

        assert dim_l >= 0 and dim_u <= orig_img_dim, \
            "ROI is out of range"

        range_arr = np.arange(dim_l, dim_u)
        range_tup_list.append(range_arr)

    idx_range_tup = tuple(range_tup_list)
    idx = np.ix_(*idx_range_tup)   # Converts n-tuple of 1-D ranges into numpy n-D index; Input individual tuple elems into ix_ function

    return idx























