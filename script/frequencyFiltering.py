import cv2
import numpy as np
from numpy import fft

def Ideal_LowPass_Filter(img, p, radius):
    B, G, R = cv2.split(img.copy())
    rows = img.shape[0]
    cols = img.shape[1]
    centerX = int((rows+1)/2)
    centerY = int((cols+1)/2)
    H = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            D = np.sqrt((i-centerX)**2 + (j-centerY)**2)
            He = p if D < radius else not p
            H[i, j] = He

    Layer_list = [B, G, R]
    mapList = []
    for i in range(3):
        Layer_list[i] = fft.fft2(Layer_list[i])
        Layer_list[i] = fft.fftshift(Layer_list[i])
        map = np.log(np.abs(Layer_list[i]) + 1)
        map = cv2.normalize(map, None, 0, 255, cv2.NORM_MINMAX)
        Layer_list[i] = Layer_list[i] * H
        map = np.round(map * H)
        map = map.astype(np.uint8)
        mapList.append(map)
        Layer_list[i] = fft.ifft2(Layer_list[i])
        Layer_list[i] = np.abs(Layer_list[i])
        Layer_list[i] = np.where(Layer_list[i] >255, 255, Layer_list[i])
        Layer_list[i] = np.where(Layer_list[i] <0, 0, Layer_list[i])
        Layer_list[i] = Layer_list[i].astype(np.uint8)

    img_shifted = cv2.merge(Layer_list)
    return mapList, img_shifted


def Ideal_BandPass_Filter(img, p, radius, width):
    B, G, R = cv2.split(img.copy())
    rows = img.shape[0]
    cols = img.shape[1]
    centerX = int((rows+1)/2)
    centerY = int((cols+1)/2)
    H = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            D = np.sqrt((i-centerX)**2 + (j-centerY)**2)
            He = p if radius-width/2 < D <radius+width/2 else not p
            H[i, j] = He

    Layer_list = [B, G, R]
    mapList = []
    for i in range(3):
        Layer_list[i] = fft.fft2(Layer_list[i])
        Layer_list[i] = fft.fftshift(Layer_list[i])
        map = np.log(np.abs(Layer_list[i]) + 1)
        map = cv2.normalize(map, None, 0, 255, cv2.NORM_MINMAX)
        Layer_list[i] = Layer_list[i] * H
        map = np.round(map * H)
        map = map.astype(np.uint8)
        mapList.append(map)
        Layer_list[i] = fft.ifft2(Layer_list[i])
        Layer_list[i] = np.abs(Layer_list[i])
        Layer_list[i] = np.where(Layer_list[i] >255, 255, Layer_list[i])
        Layer_list[i] = np.where(Layer_list[i] <0, 0, Layer_list[i])
        Layer_list[i] = Layer_list[i].astype(np.uint8)

    img_shifted = cv2.merge(Layer_list)
    return mapList, img_shifted


def Butterworth_BandPass_Filter(img, p, radius, width, n):
    B, G, R = cv2.split(img.copy())
    rows = img.shape[0]
    cols = img.shape[1]
    centerX = int((rows+1)/2)
    centerY = int((cols+1)/2)
    H = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            D = np.sqrt((i-centerX)**2 + (j-centerY)**2)
            frac = D*D - radius*radius
            frac = np.abs(frac)+0.0001
            He = 1 / (1+(((D*width)/(frac))**(2*n)))
            if p :
                He = 1 - He
            H[i, j] = He
    Layer_list = [B, G, R]
    mapList = []
    for i in range(3):
        Layer_list[i] = fft.fft2(Layer_list[i])
        Layer_list[i] = fft.fftshift(Layer_list[i])
        map = np.log(np.abs(Layer_list[i])+1)
        map = cv2.normalize(map, None, 0, 255, cv2.NORM_MINMAX)
        Layer_list[i] = Layer_list[i] * H
        map = np.round(map * H)
        map = map.astype(np.uint8)
        mapList.append(map)
        Layer_list[i] = fft.ifft2(Layer_list[i])
        Layer_list[i] = np.abs(Layer_list[i])
        Layer_list[i] = np.where(Layer_list[i] >255, 255, Layer_list[i])
        Layer_list[i] = np.where(Layer_list[i] <0, 0, Layer_list[i])
        Layer_list[i] = Layer_list[i].astype(np.uint8)

    img_shifted = cv2.merge(Layer_list)
    return mapList, img_shifted


def Gaussian_BandPass_Filter(img, p, radius, width):
    B, G, R = cv2.split(img.copy())
    rows = img.shape[0]
    cols = img.shape[1]
    centerX = int((rows+1)/2)
    centerY = int((cols+1)/2)
    H = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            D = np.sqrt((i-centerX)**2 + (j-centerY)**2)
            frac = D*width
            if(frac < 0.0001):
                frac = 0.0001
            He = np.exp(-((D*D-radius*radius)/(frac))**2)
            if not p :
                He = 1 - He
            H[i, j] = He
    Layer_list = [B, G, R]
    mapList = []
    for i in range(3):
        Layer_list[i] = fft.fft2(Layer_list[i])
        Layer_list[i] = fft.fftshift(Layer_list[i])
        map = np.log(np.abs(Layer_list[i])+1)
        map = cv2.normalize(map, None, 0, 255, cv2.NORM_MINMAX)
        Layer_list[i] = Layer_list[i] * H
        map = np.round(map * H)
        map = map.astype(np.uint8)
        mapList.append(map)
        Layer_list[i] = fft.ifft2(Layer_list[i])
        Layer_list[i] = np.abs(Layer_list[i])
        Layer_list[i] = np.where(Layer_list[i] >255, 255, Layer_list[i])
        Layer_list[i] = np.where(Layer_list[i] <0, 0, Layer_list[i])
        Layer_list[i] = Layer_list[i].astype(np.uint8)

    img_shifted = cv2.merge(Layer_list)
    return mapList, img_shifted


# if __name__ == '__main__':
    # img = cv2.imread("./imdemos/lenna.bmp")
    # img = Butterworth_BandPass_Filter(img, 0, 200, 20, 5)
    # cv2.imshow('Butter', img)
    # cv2.waitKey(0)