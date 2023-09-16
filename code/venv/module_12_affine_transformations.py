import cv2 as cv
import numpy as np

if __name__ == "__main__":
    image = cv.imread("image.jpg")

    img_t = np.array([[0, 0], [image.shape[1] - 1, 0], [0, image.shape[0] - 1]]).astype(np.float32)
    dst_t = np.array([[0, image.shape[1] * 0.13], [image.shape[1] * 0.85, image.shape[0] * 0.25],
                       [image.shape[1] * 0.15, image.shape[0] * 0.7]]).astype(np.float32)

    warp_mat = cv.getAffineTransform(img_t, dst_t)

    warp_dst = cv.warpAffine(image, warp_mat, (image.shape[1], image.shape[0]))

    cv.imshow("warped", warp_dst)

    center = (warp_dst.shape[1] // 2, warp_dst.shape[0] // 2)
    angle = -30
    scale = 0.5

    rot_mat = cv.getRotationMatrix2D(center, angle, scale)

    warp_rotate_dst = cv.warpAffine(warp_dst, rot_mat, (warp_dst.shape[1], warp_dst.shape[0]))

    cv.imshow("Warp and rotate", warp_rotate_dst)

    cv.waitKey(0)