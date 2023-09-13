import cv2 as cv
from random import randint

if __name__ == "__main__":
    image = cv.imread("image.jpg")

    borderType = cv.BORDER_CONSTANT
    window_name = "Custom border"

    cv.namedWindow(window_name, cv.WINDOW_AUTOSIZE)

    top = int(0.03 * image.shape[0])
    bottom = top
    left = int(0.015 * image.shape[1])
    right = left

    print("ESC - exit\nc - random color border\nr - replicate border based on image")

    while 1:

        value = [randint(0, 255), randint(0, 255), randint(0, 255)]

        dst = cv.copyMakeBorder(image, top, bottom, left, right, borderType, None, value)

        cv.imshow(window_name, dst)

        c = cv.waitKey(0)
        if c == 27:
            break
        elif chr(c) == 'c':
            borderType = cv.BORDER_CONSTANT
        elif chr(c) == 'r':
            borderType = cv.BORDER_REPLICATE