import cv2 as cv
import numpy as np
from random import randint


if __name__ == "__main__":
    src = cv.imread("layers.jpg")

    cv.imshow('Source Image', src)
    src[np.all(src == 255, axis=2)] = 0

    kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)

    imgLaplacian = cv.filter2D(src, cv.CV_32F, kernel)

    sharp = np.float32(src)

    imgResult = sharp - imgLaplacian

    imgResult = np.clip(imgResult, 0, 255)
    imgResult = imgResult.astype('uint8')

    imgLaplacian = np.clip(imgLaplacian, 0, 255)
    imgLaplacian = np.uint8(imgLaplacian)

    bw = cv.cvtColor(imgResult, cv.COLOR_BGR2GRAY)

    _, bw = cv.threshold(bw, 90, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

    dist = cv.distanceTransform(bw, cv.DIST_L2, 3)

    cv.normalize(dist, dist, 0, 1.0, cv.NORM_MINMAX)

    _, dist = cv.threshold(dist, 0.4, 1.0, cv.THRESH_BINARY)
    # Dilate a bit the dist image
    kernel1 = np.ones((3,3), dtype=np.uint8)
    dist = cv.dilate(dist, kernel1)

    dist_8u = dist.astype('uint8')
    # Find total markers
    contours, _ = cv.findContours(dist_8u, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # Create the marker image for the watershed algorithm
    markers = np.zeros(dist.shape, dtype=np.int32)
    # Draw the foreground markers
    for i in range(len(contours)):
        cv.drawContours(markers, contours, i, (i+1), -1)
    # Draw the background marker
    cv.circle(markers, (5,5), 3, (255,255,255), -1)

    markers_8u = (markers * 10).astype('uint8')

    cv.watershed(imgResult, markers)

    mark = markers.astype('uint8')
    mark = cv.bitwise_not(mark)

    colors = []
    for contour in contours:
        colors.append((randint(0,256), randint(0,256), randint(0,256)))
    # Create the result image
    dst = np.zeros((markers.shape[0], markers.shape[1], 3), dtype=np.uint8)
    # Fill labeled objects with random colors
    for i in range(markers.shape[0]):
        for j in range(markers.shape[1]):
            index = markers[i,j]
            if index > 0 and index <= len(contours):
                dst[i,j,:] = colors[index-1]


    cv.imshow('Final Result', dst)
    cv.waitKey()
