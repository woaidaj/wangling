import sys
from PyQt5.QtGui import (QPixmap, QImage, qRed, qGreen, qBlue)
from PyQt5.QtCore import (QThread, pyqtSignal, QObject, QRect)
import numpy as np
import math
import cv2
import copy
import pywt
import os
import time
from matplotlib import pyplot as plt

'''
***************************************************
                图像处理线程
功能：
     1，图像格式预处理；    2，ROI区域分割(待考究，应该划分为另一个线程)；
     3，图像特征提取（待考究）；4，图像清晰度评价
———————————————————————————————————————————————————
使用方式：
    1，在主界面初始化线程实例a = imgPro()；
    2，然后，对a的属性赋值, 进行赋值和初始化；
    3，开始线程a.start()
---------------------------------------------------
PS：线程.start()后会运行线程中run（）代码段，run代码return
后会结束线程。此时，还可以修改线程中的参数，再start()重新开始
线程。至于怎么样才算把线程销毁了，不知道，在这里也不重要啦。
***************************************************
'''



class imgProThread(QThread, QObject):

    valueSend = pyqtSignal(float)            # 发出的图像清晰度评价值
    ROISend = pyqtSignal(object)             # 发出的ROI信号

    def __init__(self):
        super().__init__()
        self.__bgr = None                   # QImage转opencv类型的图像数据
        self.__gray = None                  # bgr灰度化的灰度图像
        self.roi = None                     # ROI区域
        self.img_roi = None                 # ROI分割的图像
        self.height = None                  # 图像的高
        self.width = None                   # 图像的宽
        self.deep = None                    # 图像的通道数
        self.coe = 0                        # 小波函数所占权重
        self.sigma = 0.2                    # 自适应Canny算子的阈值

    def readImage(self):
        '''读取要处理的图像'''
        dir = os.path.dirname(os.path.realpath(__file__))
        newdir = os.path.join(dir, 'image', 'process.png')
        self.__bgr = cv2.imread(newdir, 1)      # 读取彩色图像

    def grayScale(self):
        '''彩色图像转为灰度图像'''
        self.__gray = cv2.cvtColor(self.__bgr, cv2.COLOR_BGR2GRAY)

    def imFeature(self):
        '''小波变换提取图像特征'''
        cA, (cH, cV, cD) = pywt.dwt2(self.__gray, 'db6')
        result = np.std(abs(cH)) ** 2 + np.std(abs(cV)) ** 2 + np.std(abs(cD))
        return result

    def ROI(self):
        # Sobel滤波
        x = cv2.Sobel(self.__gray, cv2.CV_16S, 1, 0)
        y = cv2.Sobel(self.__gray, cv2.CV_16S, 0, 1)
        absX = cv2.convertScaleAbs(x)  # 转回uint8
        absY = cv2.convertScaleAbs(y)
        dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
        # 图像叠加
        image_add = cv2.add(self.__gray, dst)
        # 自适应阈值Canny滤波
        # 计算单通道像素强度的中位数
        v = np.median(image_add)
        # 选择合适的lower和upper值，然后应用它们
        lower = int(max(0, (1.0 - self.sigma) * v))
        upper = int(min(255, (1.0 + self.sigma) * v))
        image_gaus = cv2.GaussianBlur(image_add, (5, 5), 0)
        edged = cv2.Canny(image_gaus, lower, upper)
        # 得到边缘的最小矩形
        por = cv2.boundingRect(edged)
        # 判断ROI的大小
        x, y, w, h = por
        if w < 10 and h < 10:
            self.roi = None
            self.ROISend.emit(QRect(0, 0, 0, 0))
        else:
            self.roi = por
            self.ROISend.emit(QRect(por[0], por[1], por[2], por[3]))

    def zonaPellucidaDivision(self, center, radius):
        '''
            进行透明带分割（PS：进行体模测试的时候，这一步应该跳过）
        '''
        # 极坐标变换
        img_polar = cv2.warpPolar(self.img_roi, (300, 600), center, radius, cv2.INTER_LINEAR | cv2.WARP_POLAR_LINEAR)
        # Soble边缘检测
        sob = self.sobel(img_polar)
        # LOG变换
        log = self.LOG(sob)
        # 阈值分割
        log[log < 40] = 0
        log[log >= 40] = 255
        # 极坐标反变换
        img_new = cv2.warpPolar(log, (self.width, self.height), center, radius, cv2.WARP_INVERSE_MAP)
        # 图像开操作
        thd = cv2.dilate(img_new, None, iterations=30)
        thd = cv2.erode(thd, None, iterations=30)
        contours, _ = cv2.findContours(thd, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours.sort(key=lambda c: cv2.contourArea(c), reverse=True)  # 将轮廓按面积进行排序
        cv2.drawContours(img_new, contours, -1, (0, 0, 255), 3)
        # 椭圆拟合
        ellipse1 = cv2.fitEllipse(contours[0])  # 返回元组：1，（x,y）中心位置；2，(a,b)短长轴直径，非半径；3，angle角度
        blank_crop = np.zeros(np.shape(img_new))
        cv2.ellipse(blank_crop, ellipse1, (255, 255, 0), 10)
        # 将椭圆覆盖的像素清除
        zeros_index = np.argwhere(blank_crop == 255)
        for i in zeros_index:
            img_new[i[0], i[1]] = 0
        # 图像开操作
        thd = cv2.dilate(img_new, None, iterations=20)
        thd = cv2.erode(thd, None, iterations=20)
        contours, _ = cv2.findContours(thd, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours.sort(key=lambda c: cv2.contourArea(c), reverse=True)  # 将轮廓按面积进行排序
        cv2.drawContours(img_new, contours, -1, (0, 0, 255), 3)
        # 再一次椭圆拟合
        ellipse2 = cv2.fitEllipse(contours[0])
        blank_crop[:, :] = 0
        cv2.ellipse(blank_crop, ellipse2, (255, 255, 0), 10)
        return ellipse1, ellipse2


    @staticmethod
    def detectCircle(image):
        # 霍夫圆检测
        dst = cv2.pyrMeanShiftFiltering(image, 10, 80)  # 边缘保留滤波EPF(参数：图像，空间窗口半径，颜色窗口半径)
        cimage = cv2.cvtColor(dst, cv2.COLOR_RGB2GRAY)
        # 霍夫圆参数：图像，霍夫圆检测法，图像像素分辨率与参数空间分辨率的比值，圆心最小距离，canny双阈值的高阈值（低阈值是它的一半），最小投票数，要检测的最小半径，最大半径
        circles = cv2.HoughCircles(cimage, cv2.HOUGH_GRADIENT, 1, 40, param1=50, param2=30, minRadius=40, maxRadius=80)
        return circles

    @staticmethod
    def sobel(image):
        # 静态方法：Sobel边缘检测
        x = cv2.Sobel(image, cv2.CV_16S, 1, 0)
        y = cv2.Sobel(image, cv2.CV_16S, 0, 1)
        absX = cv2.convertScaleAbs(x)  # 转回uint8
        absY = cv2.convertScaleAbs(y)
        dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
        return dst

    @staticmethod
    def LOG(image):
        # 静态方法：LOG边缘检测
        # 先通过高斯滤波降噪
        gaussian = cv2.GaussianBlur(image, (3, 3), 0)
        # 再通过拉普拉斯算子做边缘检测
        # dst = cv2.Laplacian(gaussian, cv2.CV_16S, ksize=3)
        kernel = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]])
        dst = cv2.filter2D(gaussian, -1, kernel)
        LOG = cv2.convertScaleAbs(dst)
        return LOG

    @staticmethod
    def db6(image):
        '''小波函数'''
        cA, (cH, cV, cD) = pywt.dwt2(image, 'db6')
        result = np.std(abs(cH)) ** 2 + np.std(abs(cV)) ** 2 + np.std(abs(cD))
        return result

    @staticmethod
    def variance(image):
        '''归一化方差'''
        u = np.mean(image)
        result = np.square(image - u)
        result = np.sum(result) / (image.shape[0] * image.shape[1])
        return result

    def run(self):
        self.readImage()
        self.grayScale()  # 灰度化
        [self.height, self.width, self.deep] = self.__bgr.shape  # 图像大小
        v1 = self.coe * self.db6(self.__gray) + (1 - self.coe) * self.variance(self.__gray)
        # 判断ROI
        if self.roi is None:
            self.valueSend.emit(v1)
            return
        else:
            x, y, w, h = self.roi
            self.img_roi = copy.copy(self.__gray[y:y+h, x:x+w])
        '''确定圆心和半径'''
        center = (x + int(w / 2), y + int(h / 2))
        radius = max(int(w / 2), int(h / 2))

        '''进行透明带分割'''
        if (radius * 2) < (self.width - center[0]) and (radius * 2) < (self.height - center[1]) and (radius * 2) < \
                center[0] and (radius * 2) < center[1]:
            e1, e2 = self.zonaPellucidaDivision(center, radius * 2)
        else:
            e1, e2 = self.zonaPellucidaDivision(center, min((self.width - center[0]), (self.height - center[1]), center[0], center[1]))
        # 计算透明带宽度
        short1, long1 = e1[1]
        short2, long2 = e2[1]
        w = min(abs(short1 - short2), abs(long1 - long2))
        # 计算ROI的清晰度评价值
        v2 = self.coe * self.db6(self.img_roi) + (1 - self.coe) * self.variance(self.__gray)
        v2 = v2 / w + v1
        '''霍夫圆检测'''
        h_crop = self.sobel(self.img_roi)
        h_crop = self.LOG(h_crop)
        h = cv2.cvtColor(h_crop, cv2.COLOR_GRAY2RGB)
        circles = self.detectCircle(h)
        # 判断圆的个数
        if circles is None:
            self.valueSend.emit(v2)
            return
        else:
            circles = np.uint16(np.around(circles))  # 把circles包含的圆心和半径的值变成整数
            h = []  # 保存每个圆的清晰度链表
            # 对每个圆进行处理
            for i in circles[0, :]:
                c = self.img_roi[i[1] - i[2]:i[1] + i[2], i[0] - i[2]:i[0] + i[2]]
                # 计算当前圆的清晰度
                temp = self.coe * self.db6(c) + (1 - self.coe) * self.variance(self.__gray)
                v3 = temp
                h.append(v3)
            # 累加
            v3 = sum(h) / len(h) + v2
            self.valueSend.emit(v3)
            return

# 测试代码：重写cv2.imshow,方便显示图像测试
def imshow(img):
    cv2.namedWindow('image')
    cv2.imshow('image', img)
    cv2.waitKey()
    cv2.destroyWindow('image')


if __name__ == '__main__':
    import tkinter as tk
    from tkinter import filedialog
    '''选择文件'''
    root = tk.Tk()
    root.withdraw()
    filePath = filedialog.askopenfilename(title='Select Image', initialdir='D:/Python_Script/autofocus/test')

    _, imgs = cv2.imreadmulti(filePath)

    print('测试结束')


