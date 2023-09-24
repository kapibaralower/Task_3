import cv2 as cv
import numpy as np
from random import randint


def thresh_callback(val):
    threshold = val

    canny_output = cv.Canny(src_gray, threshold, threshold * 2)

    contours, _ = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    hull_list = []
    for i in range(len(contours)):
        hull = cv.convexHull(contours[i])
        hull_list.append(hull)

    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)

    for i in range(len(contours)):
        color = (randint(0,256), randint(0,256), randint(0,256))
        cv.drawContours(drawing, contours, i, color)
        cv.drawContours(drawing, hull_list, i, color)

    cv.imshow('Contours', drawing)

if __name__ == "__main__":
    src = cv.imread("many_figures.jpg")

    src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    src_gray = cv.blur(src_gray, (3,3))

    source_window = 'Source'

    cv.namedWindow(source_window)
    cv.imshow(source_window, src)

    max_thresh = 255
    thresh = 100

    cv.createTrackbar('Canny thresh:', source_window, thresh, max_thresh, thresh_callback)

    thresh_callback(thresh)

    cv.waitKey()
