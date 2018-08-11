import numpy as np

from hough_circles_acc import hough_circles_acc
from hough_peaks import hough_peaks


def find_circles(BW, radius_range, n_peaks, radius_step=1, threshold=None, nHoodSize=None):
    # Find circles in given radius range using Hough transform.
    # BW: Binary (black and white) image containing edge pixels
    # radius_range: Range of circle radii [min max] to look for, in pixels


    img = BW


    ### Checks

    # radius_range
    assert type(radius_range) is tuple and len(radius_range) == 2, \
        "radius_range must be a 2-tuple"

    for elem in radius_range:
        assert (elem > 0), \
            "nHoodSize tuple elements must be positive ints"
    new_tup_list = [int(elem) for elem in radius_range]
    radius_range = tuple(new_tup_list)

    assert radius_step >= 1, \
        "radius_step must be >= 1"


    ### Initialization

    radius_arr = np.arange(radius_range[0], radius_range[1] +1, step=radius_step)
    hough_acc_shape = (len(radius_arr), img.shape[0], img.shape[1],)   # (radius, img_rows, img_cols)
    hough_space_accum = np.zeros(hough_acc_shape, dtype=np.uint64)


    ### Circle Hough Transform

    for i in range(len(radius_arr)):
        radius = radius_arr[i]
        hough_circle_centres_accum = hough_circles_acc(img, radius)
        hough_space_accum[i] = hough_circle_centres_accum

    # threshold
    if threshold is None:
        threshold = 0.50 * np.amax(hough_space_accum, axis=None)

    # nHoodSize
    if nHoodSize is None:
        nHoodSize = img.shape
        new_tup_list = [(int(elem / 80) * 2) +1 for elem in nHoodSize]  # b,a space, create odd values >= size(H) / 50
        nHoodSize_radius = int(5/radius_step)
        nHoodSize_radius = nHoodSize_radius +1 if nHoodSize_radius % 2 == 0 else nHoodSize_radius
        new_tup_list.insert(0, nHoodSize_radius)  # Set nHoodSize for radius parameter to be 3
        nHoodSize = tuple(new_tup_list)


    peaks_arr = hough_peaks(hough_space_accum, numpeaks=n_peaks, nHoodSize=nHoodSize)

    centers_arr = peaks_arr[:, 1:3]
    radii_arr = peaks_arr[:, 0]
    radii_arr = radius_arr[radii_arr]

    return centers_arr, radii_arr















