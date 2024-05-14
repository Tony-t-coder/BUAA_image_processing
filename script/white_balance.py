import cv2
import numpy as np

def white_balance(img, k):
    cnt = np.zeros(256)
    B, G, R = cv2.split(img.copy())
    img_h, img_w = B.shape[:2]
    for i in range(img_h):
        for j in range(img_w):
            cnt[B[i,j]] += 1
            cnt[G[i,j]] += 1
            cnt[R[i,j]] += 1
    gate = k*img_h*img_w*3
    T = 0
    now = 0
    for i in range(255, -1, -1):
        now += cnt[i]
        if(now > gate):
            T = i
            break
    avg_B = 0
    avg_G = 0
    avg_R = 0
    cntB = 0
    cntG = 0
    cntR = 0
    for i in range(img_h):
        for j in range(img_w):
            if(B[i,j]>T):
                cntB += 1
                avg_B += B[i, j]
            if(G[i,j]>T):
                cntG += 1
                avg_G += G[i, j]
            if(R[i,j]>T):
                cntR += 1
                avg_R += R[i, j]
    avg_B /= cntB
    avg_G /= cntG
    avg_R /= cntR
    array = np.array([avg_B, avg_G, avg_R], dtype=np.float64)
    M = array.max()
    Kb = M / avg_B
    Kg = M / avg_G
    Kr = M / avg_R
    B_ = np.round(Kb * B)
    G_ = np.round(Kg * G)
    R_ = np.round(Kr * R)
    B_ = np.where(B_<0, 0, B_)
    G_ = np.where(G_<0, 0, G_)
    R_ = np.where(R_<0, 0, R_)
    B_ = np.where(B_>255, 255, B_)
    G_ = np.where(G_>255, 255, G_)
    R_ = np.where(R_>255, 255, R_)
    B_ = B_.astype(np.uint8)
    G_ = G_.astype(np.uint8)
    R_ = R_.astype(np.uint8)
    balanced = cv2.merge([B_, G_, R_])
    return balanced


# img = cv2.imread("./imdemos/Link.jpg")
# balanced = white_balance(img, 0.5)
# cv2.imshow("balance", balanced)
# cv2.waitKey(0)
