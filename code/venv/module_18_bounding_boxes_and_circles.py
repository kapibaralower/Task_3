import cv2 as cv
import numpy as np
from random import randint


def thresh_callback(val):
    threshold = val

    canny_output = cv.Canny(src_gray, threshold, threshold * 2)

    contours, _ = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    contours_poly = [None] * len(contours)
    bound_rect = [None] * len(contours)
    centers = [None] * len(contours)
    radius = [None] * len(contours)

    for i, c in enumerate(contours):
        contours_poly[i] = cv.approxPolyDP(c, 3, True)
        bound_rect[i] = cv.boundingRect(contours_poly[i])
        centers[i], radius[i] = cv.minEnclosingCircle(contours_poly[i])

    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)

    for i in range(len(contours)):
        color = (randint(0, 256), randint(0, 256), randint(0, 256))
        cv.drawContours(drawing, contours_poly, i, color)
        cv.rectangle(drawing, (int(bound_rect[i][0]), int(bound_rect[i][1])),
                     (int(bound_rect[i][0] + bound_rect[i][2]), int(bound_rect[i][1] + bound_rect[i][3])), color, 2)

        cv.circle(drawing, (int(centers[i][0]), int(centers[i][1])), int(radius[i]), color, 2)

    cv.imshow('Contours', drawing)


if __name__ == "__main__":
    src = cv.imread("many_figures.jpg")

    src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

    src_gray = cv.blur(src_gray, (3, 3))

    source_window = 'Source'

    cv.namedWindow(source_window)

    cv.imshow(source_window, src)

    max_thresh = 255
    thresh = 100

    cv.createTrackbar('Canny thresh:', source_window, thresh, max_thresh, thresh_callback)

    thresh_callback(thresh)
    cv.waitKey()
