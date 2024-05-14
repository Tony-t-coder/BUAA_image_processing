import matplotlib.pyplot as plt
import numpy as np
from numpy import fft
import math
import cv2


# 仿真运动模糊
def motion_process(image_size, motion_angle, motion_degree):
    PSF = np.zeros(image_size)
    print(image_size)
    center_position = (image_size[0] - 1) / 2
    print(center_position)

    slope_tan = math.tan(motion_angle * math.pi / 180)
    slope_cot = 1 / slope_tan
    if slope_tan <= 1:
        for i in range(motion_degree):
            offset = round(i * slope_tan)  # ((center_position-i)*slope_tan)
            PSF[int(center_position + offset), int(center_position - offset)] = 1
        return PSF / PSF.sum()             # 对点扩散函数进行归一化亮度
    else:
        for i in range(motion_degree):
            offset = round(i * slope_cot)
            PSF[int(center_position - offset), int(center_position + offset)] = 1
        return PSF / PSF.sum()

# 对图片进行运动模糊
def make_blurred(input, PSF, eps):
    input_fft = fft.fft2(input)             # 进行二维数组的傅里叶变换
    PSF_fft = fft.fft2(PSF) + eps
    blurred = fft.ifft2(input_fft * PSF_fft)
    blurred = np.abs(fft.fftshift(blurred))
    return blurred

def inverse(input, PSF, eps):                # 逆滤波
    input_fft = fft.fft2(input)
    PSF_fft = fft.fft2(PSF) + eps            # 噪声功率，这是已知的，考虑epsilon
    result = fft.ifft2(input_fft / PSF_fft)  # 计算F(u,v)的傅里叶反变换
    result = np.abs(fft.fftshift(result))
    return result

def wiener(input, PSF, eps, K=0.001):        # 维纳滤波，K=0.01
    input_fft = fft.fft2(input)
    PSF_fft = fft.fft2(PSF) + eps
    PSF_fft_1 = np.conj(PSF_fft) / (np.abs(PSF_fft) ** 2 + K)
    result = fft.ifft2(input_fft * PSF_fft_1)
    result = np.abs(fft.fftshift(result))
    return result
    
def normal(array):
    array = np.where(array < 0,  0, array)
    array = np.where(array > 255, 255, array)
    array = array.astype(np.int16)
    return array

def wiener_filter(img, motion_angle=0, motion_degree=1, noise_degree=0, k=0.001):
    b_gray, g_gray, r_gray = cv2.split(img.copy())
    img_h, img_w = b_gray.shape[:2]
    PSF = motion_process((img_h, img_w), motion_angle, motion_degree)
    Result = []
    for gray in [b_gray, g_gray, r_gray]:
        channel = wiener(gray, PSF, noise_degree + 1e-3, k)
        Result.append(normal(channel))
    print(Result)
    result_wiener = cv2.merge([Result[0], Result[1], Result[2]])
    return result_wiener


def inverse_filter(img, motion_angle=0, motion_degree=1, noise_degree=0):
    b_gray, g_gray, r_gray = cv2.split(img.copy())
    img_h, img_w = b_gray.shape[:2]
    PSF = motion_process((img_h, img_w), motion_angle, motion_degree)
    Result = []
    for gray in [b_gray, g_gray, r_gray]:
        channel = inverse(gray, PSF, noise_degree + 1e-3)
        Result.append(normal(channel))
    print(Result)
    result_inverse = cv2.merge([Result[0], Result[1], Result[2]])
    return result_inverse

def main(gray):
    channel = []
    img_h, img_w = gray.shape[:2]
    PSF = motion_process((img_h, img_w), motion_angle=30, motion_degree=40)      # 进行运动模糊处理
    blurred = np.abs(make_blurred(gray, PSF, 1e-3))

    result_blurred = inverse(blurred, PSF, 1e-3)  # 逆滤波
    result_wiener = wiener(blurred, PSF, 1e-3)    # 维纳滤波

    blurred_noisy = blurred + 0.1 * blurred.std() * \
                    np.random.standard_normal(blurred.shape)  # 添加噪声,standard_normal产生随机的函数
    inverse_mo2no = inverse(blurred_noisy, PSF, 0.1 + 1e-3)   # 对添加噪声的图像进行逆滤波
    wiener_mo2no = wiener(blurred_noisy, PSF, 0.1 + 1e-3)     # 对添加噪声的图像进行维纳滤波
    channel.append((normal(blurred),normal(result_blurred),normal(result_wiener),
                    normal(blurred_noisy),normal(inverse_mo2no),normal(wiener_mo2no)))
    return channel

if __name__ == '__main__':
    image = cv2.imread('../imdemos/lenna.bmp')
    b_gray, g_gray, r_gray = cv2.split(image.copy())

    Result = []
    for gray in [b_gray, g_gray, r_gray]:
        channel = main(gray)
        Result.append(channel)
    blurred = cv2.merge([Result[0][0][0], Result[1][0][0], Result[2][0][0]])
    cv2.imwrite("blurred.png", blurred)
    result_blurred = cv2.merge([Result[0][0][1], Result[1][0][1], Result[2][0][1]])
    result_wiener = cv2.merge([Result[0][0][2], Result[1][0][2], Result[2][0][2]])
    blurred_noisy = cv2.merge([Result[0][0][3], Result[1][0][3], Result[2][0][3]])
    cv2.imwrite("blurredNoisy.png", blurred_noisy)
    inverse_mo2no = inverse_filter(blurred_noisy, 30, 10, 0.1)
    wiener_mo2no = wiener_filter(blurred_noisy, 30, 10, 0.1)

    plt.figure(1)
    plt.xlabel("Original Image")
    plt.imshow(np.flip(image, axis=2))

    plt.figure(2, figsize=(8, 6.5))
    imgNames = {"Motion blurred":blurred,
                "inverse deblurred":result_blurred,
                "wiener deblurred(k=0.01)":result_wiener,
                "motion & noisy blurred":blurred_noisy,
                "inverse_mo2no":inverse_mo2no,
                'wiener_mo2no':wiener_mo2no}
    for i,(key,imgName) in enumerate(imgNames.items()):
        plt.subplot(231+i)
        plt.xlabel(key)
        plt.imshow(np.flip(imgName, axis=2))
    plt.show()
