import cv2 as cv
import numpy as np
from random import uniform

if __name__ == "__main__":
    image = cv.imread("image.jpg")

    kernel_size = 4
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)

    for i in range(kernel_size):
        for j in range(kernel_size):
            kernel[i][j] = round(uniform(0, 0.12), 2)

    print(kernel)

    dst = cv.filter2D(image, -1, kernel, anchor=(3, 2), delta=-65)

    cv.imshow("Filtered image", dst)

    cv.waitKey(0)
