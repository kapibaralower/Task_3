import cv2 as cv
import numpy as np


def zoom(img):
    print("ESC - exit zoom \ni - in zoom \no - out zoom \n_______________")

    # menu of zooming func
    while 1:
        # copy information about size and channels count of original image
        rows, cols, _channels = map(int, img.shape)

        cv.imshow("Zoom", img)

        k = cv.waitKey(0)

        if k == 27:
            break

        # zoom in image (size x2)
        elif chr(k) == 'i':
            img = cv.pyrUp(img, dstsize=(2 * cols, 2 * rows))

        # zoom out image (size /2)
        elif chr(k) == 'o':
            img = cv.pyrDown(img, dstsize=(cols // 2, rows // 2))

    cv.destroyAllWindows()


def re_map(map_x, map_y, img):
    print("ESC - exit \nt - top-down rotate \nr - right-left rotate \n_______________")

    # menu of remap func
    while 1:
        cv.imshow("Remap", img)

        k = cv.waitKey(0)

        if k == 27:
            break

        # create maps as templates to swap topsides and downsides of image
        elif chr(k) == 't':
            for i in range(map_x.shape[0]):
                map_x[i, :] = [x for x in range(map_x.shape[1])]
                for j in range(map_y.shape[1]):
                    map_y[:, j] = [map_y.shape[0] - y for y in range(map_y.shape[0])]

        # create maps as templates to swap right and left sides of image
        elif chr(k) == 'r':
            for i in range(map_x.shape[0]):
                map_x[i, :] = [map_x.shape[1] - x for x in range(map_x.shape[1])]
            for j in range(map_y.shape[1]):
                map_y[:, j] = [y for y in range(map_y.shape[0])]

        # use template maps to remap image
        img = cv.remap(img, map_x, map_y, cv.INTER_LINEAR)
        cv.destroyAllWindows()

    cv.destroyAllWindows()


if __name__ == "__main__":
    image = cv.resize(cv.imread("image.jpg"), [800, 332])

    # create arrays with original image size
    map_x = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
    map_y = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)

    # primitive menu
    while 1:
        print("ESC - exit \nz - zoom \nr - remap \n_______________")

        cv.imshow("Image", image)

        k = cv.waitKey(0)

        if k == 27:
            break

        elif chr(k) == 'z':
            cv.destroyAllWindows()
            zoom(image)

        elif chr(k) == 'r':
            cv.destroyAllWindows()
            re_map(map_x, map_y, image)
