import cv2
import numpy as np
import math

PI = np.pi
eps = 0.0001


def rgb2hsi(img_bgr):
    b, g, r = cv2.split(img_bgr/255.)
    fenmu = ((r-g)*(r-g))+(r-b)*(g-b)
    TT = np.where(2*np.sqrt(fenmu)!=0, (2*r-b-g)/(2*np.sqrt(fenmu)), 0)
    TT = np.where(TT>1, 1, TT)
    TT = np.where(TT<-1, -1, TT)
    Tdata = np.where((r-g)**2+(r-b)*(g-b)!=0, np.arccos(TT),0)
    Hdata = np.where(g >= b,Tdata, 2*PI-Tdata) 
    Hdata = Hdata / (2*PI)
    Sdata = np.where((b+g+r) != 0, 1-3.*(np.minimum(np.minimum(b,g), r))/(b+g+r),0)
    Idata = np.array((b+g+r)/3.)
    img_hsi = np.zeros((img_bgr.shape[0], img_bgr.shape[1], 3))
    img_hsi[:,:,0] = Hdata*255
    img_hsi[:,:,1] = Sdata*255
    img_hsi[:,:,2] = Idata*255
    img_hsi = np.where(img_hsi < 0, 0, img_hsi)
    img_hsi = np.where(img_hsi > 255, 255, img_hsi)
    img_hsi = np.array(np.round(img_hsi)).astype(np.uint8)
    return img_hsi


def hsi2rgb(img_hsi):
    H, S, I = cv2.split(img_hsi)
    H = H / 255.0 * 2 * PI
    S = S / 255.0
    I = I / 255.0
    rows = img_hsi.shape[0]
    cols = img_hsi.shape[1]
    Bdata = np.zeros((rows, cols))
    Gdata = np.zeros((rows, cols))
    Rdata = np.zeros((rows, cols))
    Rdata = np.where((0<=H) & (H<2*PI/3),       I*(1+S*np.cos(H)/np.cos(PI/3-H)),           Rdata)
    Bdata = np.where((0<=H) & (H<2*PI/3),       I*(1-S),                                    Bdata)
    Gdata = np.where((0<=H) & (H<2*PI/3),       3*I-(Rdata+Bdata),                          Gdata)
    Gdata = np.where((2*PI/3<=H) & (H<4*PI/3),  I*(1+S*np.cos(H-2*PI/3)/np.cos(PI-H)),      Gdata)
    Rdata = np.where((2*PI/3<=H) & (H<4*PI/3),  I*(1-S),                                    Rdata)
    Bdata = np.where((2*PI/3<=H) & (H<4*PI/3),  3*I-(Rdata+Gdata),                          Bdata)
    Bdata = np.where((4*PI/3<=H) & (H<2*PI),    I*(1+S*np.cos(H-4*PI/3)/np.cos(5*PI/3-H)),  Bdata)
    Gdata = np.where((4*PI/3<=H) & (H<2*PI),    I*(1-S),                                    Gdata)
    Rdata = np.where((4*PI/3<=H) & (H<2*PI),    3*I-(Bdata+Gdata),                          Rdata)
    img_bgr = cv2.merge([Bdata, Gdata, Rdata]) * 255
    img_bgr = np.where(img_bgr < 0, 0, img_bgr)
    img_bgr = np.where(img_bgr > 255, 255, img_bgr)
    img_bgr = np.round(img_bgr).astype(np.uint8)
    return img_bgr


def gammaShift(img, gamma = 1):
    if(len(cv2.split(img)) == 3):
        B, G, R = cv2.split(img.copy())
        B = B / 255.0
        G = G / 255.0
        R = R / 255.0
        rows = int(img.shape[0])
        cols = int(img.shape[1])
        for i in range(rows):
            for j in range(cols):
                B[i, j] = B[i, j] ** gamma
                G[i, j] = G[i, j] ** gamma
                R[i, j] = R[i, j] ** gamma
        gammaShifted = cv2.merge([B, G, R]) * 255
        gammaShifted = np.round(gammaShifted).astype(np.uint8)
        return gammaShifted
    elif(len(cv2.split(img)) == 1):
        rows = int(img.shape[0])
        cols = int(img.shape[1])
        for i in range(rows):
            for j in range(cols):
                img[i, j] = img[i, j] ** gamma
        img = np.round(img * 255).astype(np.uint8)
        return img


def retinexShift(img, gamma=0.5):
    rows = int(img.shape[0])
    cols = int(img.shape[1])
    hsi_img = rgb2hsi(img)
    H, S, I = cv2.split(hsi_img)
    I  = I / 255
    I_Gaussian = cv2.GaussianBlur(I, ksize=(101, 101), sigmaX=0)
    for i in range(rows):
        for j in range(cols):
            if(I[i,j] < 0) | math.isinf(I[i,j]) | math.isnan(I[i,j]):
                print(I[i, j], i, j)
    I = np.log(I+1)
    I_Gaussian = np.log(I_Gaussian+1)
    I = I - gamma*I_Gaussian
    min = I.min()
    max = I.max()
    print(max, min)
    I = 255*(I - min) /(max - min)
    I = np.where(I > 255, 255, I)
    I = np.where(I < 0, 0, I)
    I = np.round(I).astype(np.uint8)
    hsi_img = cv2.merge([H, S, I])
    rgb_img = hsi2rgb(hsi_img)
    return rgb_img


if __name__ == '__main__':
    img = cv2.imread('F:\imageProcessing\imdemos\out67.bmp')
    cv2.imshow('origin', img)
    cv2.imshow('retinex', retinexShift(img, 0.5))
    # cv2.imshow('gamma', gammaShift(img, 0.7))
    cv2.waitKey(0)
