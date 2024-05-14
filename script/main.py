import cv2
import numpy as np
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import os
import whitening
import white_balance
import wiener
import lightness
import frequencyFiltering
import style_transfer
os.environ['KMP_DUPLICATE_LIB_OK']= 'TRUE'

app = Flask(__name__)
CORS(app)


@app.route('/process', methods=['POST'])
def process_image():
    # 从请求中获取上传的图片文件
    print('Received request from frontend.')
    uploaded_image = request.files['image']
    option = request.form['option']
    print("The user chose the option:",option)
    image_bytes = uploaded_image.read()
    image_np = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    # 在这里对图片进行处理，这里只是一个示例
    if option == "option1":
        denoisingOption = request.form['denoisingoption']
        if denoisingOption == 'denoisingOption1':
            print("均值滤波！")
            kernel_size = (int)(request.form['kernel_size'])
            image = meanFilter(image,kernel_size)
        elif denoisingOption == 'denoisingOption2':
            print("中值滤波！")
            kernel_size = (int)(request.form['kernel_size'])
            image = medianFilter(image, kernel_size)
        elif denoisingOption == 'denoisingOption3':
            print("高斯滤波！")
            kernel_size = (int)(request.form['kernel_size'])
            image = gaussianFilter(image, kernel_size)
        elif denoisingOption == 'denoisingOption4':
            print("维纳滤波！")
            motion_angle = int(request.form['motion_angle'])
            motion_degree = int(request.form['motion_degree'])
            noise_degree = float(request.form['noise_degree'])
            k = float(request.form['k'])
            image = weinerFilter(image, motion_angle, motion_degree, noise_degree, k)
        elif denoisingOption == 'denoisingOption5':
            print("逆滤波！")
            motion_angle = int(request.form['motion_angle'])
            motion_degree = int(request.form['motion_degree'])
            noise_degree = float(request.form['noise_degree'])
            print(motion_angle," ",motion_degree," ",noise_degree)
            image = inverseFilter(image, motion_angle, motion_degree, noise_degree)
    elif option == "option2":
        print("提取轮廓！")
        image = edge(image)
    elif option == "option3":
        print("人像磨皮！")
        kernel_size = int(request.form['kernel_size'])
        image = bilateralFilter(image, kernel_size)
    elif option == "option4":
        print("风格迁移！")
        StyleUploadimage = request.files['Styleimage']
        Styleimage_bytes = StyleUploadimage.read()
        Styleimage_np = np.frombuffer(Styleimage_bytes, dtype=np.uint8)
        Styleimage = cv2.imdecode(Styleimage_np, cv2.IMREAD_COLOR)
        s_weight = float(request.form['s_weight'])
        c_weight = float(request.form['c_weight'])
        resolution = int(request.form['resolution'])
        epoch = int(request.form['epoch'])
        Style_trans(Styleimage, image, s_weight, c_weight, resolution, epoch)
    elif option == "option5":
        print("白化！")
        image = whitenProcess(image)
    elif option == "option6":
        print("白平衡！")
        k = float(request.form['k'])
        image = whiteBalance(image, k)
    elif option == 'option7':
        print("锐化！")
        intensity = float(request.form['intensity'])
        image = sharpen(image, intensity)
    elif option == 'option8':
        print('图像增强！')
        enhanceoption = request.form['enhanceoption']
        if enhanceoption == 'EnhanceOption1':
            print('gamma变换！')
            gamma = float(request.form['gamma'])
            print(gamma)
            image = gammaShift(image, gamma)
        elif enhanceoption == 'EnhanceOption2':
            print('retinex算法！')
            gamma = float(request.form['gamma'])
            image = retinexShift(image, gamma)
    elif option == 'option9':
        print('频率域滤波！')
        frequencyoption = request.form['frequencyoption']
        if frequencyoption == 'FrequencyOption1':
            print('理想低通/带通！')
            p = int(request.form['p'])
            radius = int(request.form['radius'])
            image = Ideal_LowPass_Filter(image, p, radius)
        elif frequencyoption == 'FrequencyOption2':
            print('理想带通/带阻！')
            p = int(request.form['p'])
            radius = int(request.form['radius'])
            width = int(request.form['width'])
            image = Ideal_BandPass_Filter(image, p, radius, width)
        elif frequencyoption == 'FrequencyOption3':
            print('巴特沃斯带通/带阻！')
            p = int(request.form['p'])
            radius = int(request.form['radius'])
            width = int(request.form['width'])
            n = float(request.form['n'])
            image = ButterWorth_BandPass_Filter(image, p, radius, width, n)
        elif frequencyoption == 'FrequencyOption4':
            print('高斯带通/带阻！')
            p = int(request.form['p'])
            radius = int(request.form['radius'])
            width = int(request.form['width'])
            image = Gaussian_BandPass_Filter(image, p, radius, width)
    path1 = os.getcwd()
    path2 = os.path.dirname(path1)
    path = path2 + '/result/image.png'
    if option != 'option4':
        cv2.imwrite(path, image)
    # 假设处理完成后，将处理后的图片保存到服务器，并返回图片的URL
    print("保存成功，返回前端")
    # 返回处理后的图片URL
    # result = {'message': 'Data processed successfully'}
    return jsonify({'resultImageUrl': path})

#均值滤波
def meanFilter(img, kernel_size):
    kernel_size = kernel_size if (kernel_size % 2)  else kernel_size+1
    img_filtered = cv2.blur(img, ksize=(kernel_size, kernel_size))
    return img_filtered


#中值滤波_处理椒盐噪声
def medianFilter(img, kernel_size):
    kernel_size = kernel_size if (kernel_size % 2)  else kernel_size+1
    img_filtered = cv2.medianBlur(img, ksize=kernel_size)
    return img_filtered

#高斯滤波_高斯噪声
def gaussianFilter(img, kernel_size):
    kernel_size = kernel_size if (kernel_size % 2)  else kernel_size+1
    img_filtered = cv2.GaussianBlur(img, ksize=(kernel_size, kernel_size), sigmaX=0)
    return img_filtered

#维纳滤波_运动模糊
def weinerFilter(img, motion_angle, motion_degree, noise_degree, k):
    img_filtered=wiener.wiener_filter(img, motion_angle, motion_degree, noise_degree, k)
    return img_filtered

#逆滤波_不加噪声的运动模糊
def inverseFilter(img, motion_angle, motion_degree, noise_degree):
    img_filtered=wiener.inverse_filter(img, motion_angle, motion_degree, noise_degree)
    return img_filtered

#双边滤波_人像磨皮
def bilateralFilter(img, kernel_size):
    kernel_size = kernel_size if (kernel_size % 2)  else kernel_size+1
    img_filtered=cv2.bilateralFilter(img, kernel_size, 40, 10)
    return img_filtered

#白化
def whitenProcess(img):
    img_processed = whitening.whitening(img)
    return img_processed

#白平衡(色彩校正)
def whiteBalance(img, k):
    img_balanced = white_balance.white_balance(img, k)
    return img_balanced

#边缘提取_灰度图:
def edge(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_edge = cv2.Sobel(img, -1, 1, 0)
    img_edge += cv2.Sobel(img, -1, 0, 1)
    return img_edge

#锐化:
def sharpen(img, intensity):
    img_edge = cv2.Sobel(img, -1, 1, 0, ksize=3)
    img_edge += cv2.Sobel(img, -1, 0, 1, ksize=3)
    img_sharpen = intensity*img_edge + img
    img_sharpen = np.round(img_sharpen)
    img_sharpen = np.where(img_sharpen > 255, 255, img_sharpen)
    img_sharpen = np.where(img_sharpen < 0, 0, img_sharpen)
    img_sharpen = img_sharpen.astype(np.uint8)
    return img_sharpen

# 以下为新增部分
#一、图像增强
#图像增强：1.gamma变换
def gammaShift(img, gamma):  #gamma > 0 (0-1)提亮，(1-无穷)变暗
    return lightness.gammaShift(img, gamma)


#图像增强：2.retinex算法 亮度均衡
def retinexShift(img, gamma):  #gamma (0-1)
    return lightness.retinexShift(img, gamma)

#二、频率域滤波
#频率域滤波有显示中间频域图像的需求

#频率域滤波：1.理想低通/高通滤波
def Ideal_LowPass_Filter(img, p, radius): #p=0低通，p=1高通,  radius (0,300)为滤波半径
    mapList, result = frequencyFiltering.Ideal_LowPass_Filter(img, p, radius)
    for i in range(3):
        cv2.imwrite("../result/" + str(i) + ".png", mapList[i])
    return result
    
#频率域滤波：2.理想带通/带阻滤波
def Ideal_BandPass_Filter(img, p, radius, width): #p=0带阻，p=1带通, radius (0,100)为滤波带半径, width为滤波带宽
    mapList, result = frequencyFiltering.Ideal_BandPass_Filter(img, p, radius, width)
    for i in range(3):
        cv2.imwrite("../result/" + str(i) + ".png", mapList[i])
    return result

#频率域滤波：3.巴特沃斯带通/带阻滤波
def ButterWorth_BandPass_Filter(img, p, radius, width, n): 
    #p=0带阻，p=1带通, radius (0,100)为滤波带半径, width为滤波带宽, n为参数(强度>0)
    mapList, result = frequencyFiltering.Butterworth_BandPass_Filter(img, p, radius, width, n)
    for i in range(3):
        cv2.imwrite("../result/" + str(i) + ".png", mapList[i])
    return result

#频率域滤波：4.高斯带通/带阻滤波
def Gaussian_BandPass_Filter(img, p, radius, width): 
    #p=0带阻，p=1带通, radius (0,100)为滤波带半径, width为滤波带宽
    mapList, result = frequencyFiltering.Gaussian_BandPass_Filter(img, p, radius, width)
    for i in range(3):
        cv2.imwrite("../result/" + str(i) + ".png", mapList[i])
    return result

#风格化
def Style_trans(style, content, s_weight=1, c_weight=1, resolution=512, epoch = 700):
    #s_weight(0-2, default = 1, step=0.1)
    #c_weight(0-2, default = 1, step=0.1)
    #resolution(2的倍数吧, 128-1024, step=2)
    # epoch(>500, default=700, step=10)
    cv2.imwrite('../work/style.png', style)
    cv2.imwrite('../work/content.png', content)
    style_transfer.style_transfer('../work/style.png', '../work/content.png', s_weight, c_weight, resolution, epoch)


if __name__ == '__main__':
    app.run(port=5000, debug=True)
    # img1 = cv2.imread("../work/style.png")
    # img2 = cv2.imread('../work/content.png')
    # Style_trans(img1, img2, 1, 1, 512, 700)