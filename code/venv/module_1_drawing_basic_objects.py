import cv2 as cv
import numpy as np
import random as rand


# function to draw 10 ellipses in different positions
def many_ellipses(img):
    angle = 0

    for i in range(10):
        cv.ellipse(img, (500, 500), (400, 150), angle, 0, 360,
                   (rand.randint(0, 255), rand.randint(0, 255), rand.randint(0, 255)),
                   1, 100)
        angle += 36

    return 0


# generate a dot with coordinates in range of window size
def random_dot(w):
    dot = np.array([w // rand.randint(1, 10), w // rand.randint(1, 10)])

    return dot


# draw a polygons in random position based on dots from "random_dot()" func
def many_polygons(img, w):
    for i in range(4):
        points = np.array([random_dot(w), random_dot(w), random_dot(w)],
                          dtype=np.uint64)
        cv.fillPoly(img, [points],
                    (rand.randint(0, 255), rand.randint(0, 255), rand.randint(0, 255)),
                    8)

    return 0


if __name__ == "__main__":
    W = 1000
    size = W, W, 3

    atom_image = np.zeros(size, dtype=np.uint8)
    poly_image = np.zeros(size, dtype=np.uint8)

    many_ellipses(atom_image)

    # draw a filled circle in center of "atom_image"
    cv.circle(atom_image, (500, 500), 60, (255, 255, 255), -1, 1)

    cv.imshow("atom?", atom_image)

    cv.waitKey(0)

    many_polygons(poly_image, W)

    # draw a line on "poly_image"
    cv.line(poly_image, (0, 1000), (1000, 0), (255, 255, 255), 5, 1)

    # draw a filled rectangle on "poly_image"
    cv.rectangle(poly_image, (500, 500), (750, 750), (255, 255, 0), -1, 1)

    cv.imshow("not atom", poly_image)

    cv.waitKey(0)
    cv.destroyAllWindows()
