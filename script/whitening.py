import cv2
import numpy
import math
def whitening(img):
    img = numpy.array(img)
    img = img / 255.0
    mu, sigma = cv2.meanStdDev(img)  # 返回均值和方差，分别对应3个通道
    img[:, :, 0] = (img[:, :, 0] - mu[0]) / (sigma[0] + 1e-6)
    img[:, :, 1] = (img[:, :, 1] - mu[1]) / (sigma[1] + 1e-6)
    img[:, :, 2] = (img[:, :, 2] - mu[2]) / (sigma[2] + 1e-6)
    #色彩拉伸
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    # 将 像素值 高于 值域区间[0, 255] 的 像素点 置255
    img = img * (img <= 255) + 255 * (img > 255)
    img = img.astype(numpy.uint8)
    return img


# path = './imdemos/taidao.jpg'
# img = cv2.imread(path)
# cv2.imshow('origin', img)
# img = whitening(img)
# cv2.imshow('witened', img)
# cv2.waitKey(0)
