import cv2 as cv
import numpy as np
from tabulate import tabulate

compare_methods = ['Correlation', 'Chi-Square', 'Intersection', 'Bhattacharyya']

if __name__ == "__main__":
    src_base = cv.resize(cv.imread("image.jpg"), (1200, 446))
    src_test1 = cv.resize(cv.imread("many_lines.jpg"), (1200, 446))
    src_test2 = cv.resize(cv.imread("many_circles.jpg"), (1200, 446))

    hsv_base = cv.cvtColor(src_base, cv.COLOR_BGR2HSV)
    hsv_test1 = cv.cvtColor(src_test1, cv.COLOR_BGR2HSV)
    hsv_test2 = cv.cvtColor(src_test2, cv.COLOR_BGR2HSV)

    hsv_half_down = hsv_base[hsv_base.shape[0] // 2:, :]

    h_bins = 50
    s_bins = 60

    histSize = [h_bins, s_bins]

    # hue varies from 0 to 179, saturation from 0 to 255
    h_ranges = [0, 180]
    s_ranges = [0, 256]

    ranges = h_ranges + s_ranges  # concat lists

    # Use the 0-th and 1-st channels
    channels = [0, 1]

    hist_base = cv.calcHist([hsv_base], channels, None, histSize, ranges, accumulate=False)
    cv.normalize(hist_base, hist_base, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)

    hist_half_down = cv.calcHist([hsv_half_down], channels, None, histSize, ranges, accumulate=False)
    cv.normalize(hist_half_down, hist_half_down, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)

    hist_test1 = cv.calcHist([hsv_test1], channels, None, histSize, ranges, accumulate=False)
    cv.normalize(hist_test1, hist_test1, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)

    hist_test2 = cv.calcHist([hsv_test2], channels, None, histSize, ranges, accumulate=False)
    cv.normalize(hist_test2, hist_test2, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)

    head_of_table = ['Method', 'Base-base', 'Base-half', 'Base-lines', 'Base-circles']
    body_of_table = []

    for compare_method in range(4):
        base_base = cv.compareHist(hist_base, hist_base, compare_method)
        base_half = cv.compareHist(hist_base, hist_half_down, compare_method)
        base_test1 = cv.compareHist(hist_base, hist_test1, compare_method)
        base_test2 = cv.compareHist(hist_base, hist_test2, compare_method)

        body_of_table.append([compare_methods[compare_method], base_base, base_half, base_test1, base_test2])

    print(tabulate(body_of_table, head_of_table, tablefmt="grid"))

    cv.imshow("Original", hsv_base)
    cv.imshow("Half", hsv_half_down)
    cv.imshow("Test 1", hsv_test1)
    cv.imshow("Test 2", hsv_test2)

    cv.waitKey(0)
