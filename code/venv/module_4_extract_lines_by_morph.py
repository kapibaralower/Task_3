import cv2 as cv
import numpy as np


if __name__ == "__main__":
    # read and resize image
    image = cv.resize(cv.imread("image.jpg"), [1200, 446])

    # change color paliter
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    cv.imshow("Original", image)
    cv.imshow("Gray", gray)

    cv.waitKey(0)
    cv.destroyWindow("Original")

    # invert image
    gray = cv.bitwise_not(gray)

    # add binary threshhold on image
    gray_thrash_binary = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C,
                                              cv.THRESH_BINARY, 11, -2)

    cv.imshow("Gray bitwise", gray)

    cv.waitKey(0)
    cv.destroyWindow("Gray")

    cv.imshow("Thrash binary", gray_thrash_binary)

    cv.waitKey(0)
    cv.destroyAllWindows()

    # extract horizontal and vertical information
    horizontal = np.copy(gray_thrash_binary)
    vertical = np.copy(gray_thrash_binary)

    cols = horizontal.shape[1]
    horizontal_size = cols // 230

    # create horizontal line as kernel
    horizontal_structure = cv.getStructuringElement(cv.MORPH_RECT, (horizontal_size, 1))

    # decrease size of all objects on image
    horizontal = cv.erode(horizontal, horizontal_structure)
    # increase size of all horisontal lines of objects on image
    horizontal = cv.dilate(horizontal, horizontal_structure)

    rows = vertical.shape[0]
    vertical_size = rows // 75

    # create vertical line as kernel
    vertical_structure = cv.getStructuringElement(cv.MORPH_RECT, (1, vertical_size))

    # decrease size of all objects on image
    vertical = cv.erode(vertical, vertical_structure)
    # increase size of all vertical lines of objects on image
    vertical = cv.dilate(vertical, vertical_structure)

    cv.imshow("Horizontal", horizontal)
    cv.imshow("Vertical", vertical)

    cv.waitKey(0)
    cv.destroyAllWindows()

    # create edges of objects on "gray_thrash_binary"
    edges = cv.adaptiveThreshold(gray_thrash_binary, 255, cv.ADAPTIVE_THRESH_MEAN_C,
                                 cv.THRESH_BINARY, 3, -2)

    cv.imshow("Edges", edges)
    cv.imshow("Gray", gray_thrash_binary)

    cv.waitKey(0)
    cv.destroyAllWindows()
