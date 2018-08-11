import cv2
import numpy as np

from Intro_To_Computer_Vision_Utils import utils


def hough_peaks(H, numpeaks=1, threshold=None, nHoodSize=None):
    # Find peaks in a Hough accumulator array.
    # Threshold (optional): Threshold at which values of H are considered to be peaks
    # NHoodSize (optional): Size of the suppression neighborhood, [M N]

    # Please see the Matlab documentation for houghpeaks():
    # http://www.mathworks.com/help/images/ref/houghpeaks.html
    # Your code should imitate the matlab implementation.


    hough_acc = H.copy()


    ### Checks

    # Numpeaks check
    assert numpeaks > 0, \
        "numpeaks must be an int value > 0"
    numpeaks = int(numpeaks)   # Force to int

    # Threshold check
    if threshold is None:
        threshold = 0.5 * np.amax(hough_acc, axis=None)
    else:
        assert threshold > 0, \
            "threshold must be > 0"

    # nHoodSize check
    if nHoodSize is None:
        nHoodSize = hough_acc.shape
        new_tup_list = [(int(elem /100) *2) +1 for elem in nHoodSize]   # create odd values >= size(H) / 50
        nHoodSize = tuple(new_tup_list)
    else:
        assert type(nHoodSize) is tuple, \
            "nHoodSize must be a tuple"

        assert len(hough_acc.shape) == len(nHoodSize), \
            "hough_acc and nHoodSize dimensions do not match"

        for elem in nHoodSize:
            assert (elem > 0) and (elem % 2 != 0), \
                "nHoodSize tuple elements must be odd and positive ints"

        new_tup_list = [int(elem) for elem in nHoodSize]
        nHoodSize = tuple(new_tup_list)


    ### Get top n peaks

    hough_acc[hough_acc < threshold] = 0   # Remove low values

    peaks_list = []
    for i in range(numpeaks):   # Loop numpeaks times
        peak_idx_tup = np.unravel_index(np.argmax(hough_acc, axis=None), hough_acc.shape)

        if hough_acc[peak_idx_tup] == 0:   # Break if there are no peaks remaining
            break

        peaks_list.append(peak_idx_tup)

        # Set nHood to zero
        idx = utils.ROI_index_range_from_center(peak_idx_tup, nHoodSize, hough_acc.shape, allow_partial=True)
        hough_acc[idx] = 0


    peaks_arr = np.asarray(peaks_list)

    return peaks_arr


def mark_hough_peaks(img, peaks_arr, rect_size, copy=True):
    assert type(rect_size) is tuple and len(rect_size) == 2, \
        "rect_size must be a 2-tuple"

    if copy:
        img = img.copy()

    for peak in peaks_arr:
        idx = utils.ROI_index_range_from_center(tuple(peak), rect_size, img.shape, allow_partial=True)

        idx_rows = idx[0].flatten()
        idx_cols = idx[1].flatten()

        pt1 = (idx_cols[0], idx_rows[0])  # cv2.rectangle uses (x,y) instead of (row,col)
        pt2 = (idx_cols[-1], idx_rows[-1])  # cv2.rectangle uses (x,y) instead of (row,col)
        img = cv2.rectangle(img, pt1, pt2, 255)

    return img

































