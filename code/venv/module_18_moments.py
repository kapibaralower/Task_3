import cv2 as cv
import numpy as np
from random import randint


def thresh_callback(val):
    threshold = val

    canny_output = cv.Canny(src_gray, threshold, threshold * 2)

    contours, _ = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    mu = [None] * len(contours)
    for i in range(len(contours)):
        mu[i] = cv.moments(contours[i])

    mc = [None] * len(contours)
    for i in range(len(contours)):
        mc[i] = (mu[i]['m10'] / (mu[i]['m00'] + 1e-5), mu[i]['m01'] / (mu[i]['m00'] + 1e-5))

    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)

    for i in range(len(contours)):
        color = (randint(0, 256), randint(0, 256), randint(0, 256))
        cv.drawContours(drawing, contours, i, color, 2)
        cv.circle(drawing, (int(mc[i][0]), int(mc[i][1])), 4, color, -1)

    cv.imshow('Contours', drawing)

    for i in range(len(contours)):
        print(' * Contour[%d] - Area (M_00) = %.2f - Area OpenCV: %.2f - Length: %.2f' % (
        i, mu[i]['m00'], cv.contourArea(contours[i]), cv.arcLength(contours[i], True)))


if __name__ == "__main__":
    src = cv.imread("random_figures.jpg")

    src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

    src_gray = cv.blur(src_gray, (3, 3))

    source_window = 'Source'

    cv.namedWindow(source_window)

    cv.imshow(source_window, src)

    max_thresh = 255
    thresh = 100

    cv.createTrackbar('Canny Thresh:', source_window, thresh, max_thresh, thresh_callback)

    thresh_callback(thresh)
    cv.waitKey()
