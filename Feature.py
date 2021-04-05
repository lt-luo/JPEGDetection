'''
    空间域图片做DCT 转DCT矩阵，64*64 -》 64 个 （8*8）DCT块
'''

'''
    取8*8 DCT 块的前9个系数 -》 统计（-50，50）间 每个频率的个数 eg：【0，1，0，4，0...】 （dim：101） （64个块64个数，打在 101 的范围内）
    结果为 9 * 101 的矩阵，拉成一维向量得 101 * 9 维 
'''
import numpy as np
import math
import os
import tensorflow as tf
from PIL import Image
import PIC2DCT

#输入 NHWC
def getSPAFeature(picBlock=None):
    kernel3 = [[[0, 0.05, 0], [0, -0.1, 0], [0, 0.05, 0]],
               [[0, -0.05, 0], [0, 0.1, 0], [0, -0.05, 0]],
               [[0, 0.2, 0], [0, -0.4, 0], [0, 0.2, 0]],
               [[0, -0.2, 0], [0, 0.4, 0], [0, -0.2, 0]],
               [[0, 0.5, 0], [0, -1, 0], [0, 0.5, 0]],
               [[0, 0, 0], [0.35, -0.7, 0.35], [0, 0, 0]],
               [[0, 0, 0], [-0.35, 0.7, -0.35], [0, 0, 0]],
               [[0, 0, 0], [0.5, -1, 0.5], [0, 0, 0]],
               [[0, 0, 0], [-0.5, 1, -0.5], [0, 0, 0]]]
    kernel5 = [[[-0.7, 1.7, -1.8, 1.7, -0.7],
                [1.7, -5.2, 7, -5.2, 1.7],
                [-1.8, 7, -10.8, 7, -1.8],
                [1.7, -5.2, 7, -5.2, 1.7],
                [-0.7, 1.7, -1.8, 1.7, -0.7]],
               [[-0.8, 1.9, -2, 1.9, -0.8],
                [1.9, -5.8, 7.8, -5.8, 1.9],
                [-2, 7.8, -12, 7.8, -2],
                [1.9, -5.8, 7.8, -5.8, 1.9],
                [-0.8, 1.9, -2, 1.9, -0.8]],
               [[-0.9, 2.1, -2.2, 2.1, -0.9],
                [2.1, -6.4, 8.6, -6.4, 2.1],
                [-2.2, 8.6, -13.2, 8.6, -2.2],
                [2.1, -6.4, 8.6, -6.4, 2.1],
                [-0.9, 2.1, -2.2, 2.1, -0.9]]]
    kernel3 = np.array(kernel3).reshape((3, 3, 1, 9))
    kernel5 = np.array(kernel5).reshape((5, 5, 1, 3))
    #print(kernel3.shape)
    #print(kernel5.shape)

    a = tf.nn.conv2d(picBlock, kernel3, strides=(1, 1), padding='SAME')
    b = tf.nn.conv2d(picBlock, kernel5, strides=(1, 1), padding='SAME')
    c = tf.concat([a, b], axis=3)
    # a = tf.nn.conv2d(picBlock, kernel3[0], strides=(1, 1), padding='SAME')
    # for i in range(1, 9):
    #     b = tf.nn.conv2d(picBlock, kernel3[i], strides=(1, 1), padding='SAME')
    #     a = tf.concat([a, b], axis=3)
    #
    # for kernel in kernel5:
    #     b = tf.nn.conv2d(picBlock, kernel, strides=(1, 1), padding='SAME')
    #     a = tf.concat([a, b], axis=3)

    return tf.squeeze(c)

def getDCTBlockFeature(picBlock, feature):
    zigzag = [9, 2, 17, 10, 3, 25, 18,
              11, 4, 33, 26, 19, 12, 5, 41,
              34, 27, 20, 13, 6, 49, 42, 35,
              28, 21, 14, 7, 57, 50, 43, 36,
              29, 22, 15, 8, 58, 51, 44, 37,
              30, 23, 16, 52, 59, 45, 38, 31,
              24, 60, 53, 46, 39, 32, 61, 54,
              47, 40, 62, 55, 48, 63, 56, 64]

    picVec = np.reshape(picBlock, (1, 8 * 8))
    #print(picVec.shape)
    #获得前9个AC系数
    for index in range(9):
        coefficient = int(picVec[0][zigzag[index]])
        #clip
        if coefficient > 50:
            coefficient = 50
        if coefficient < -50:
            coefficient = -50

        #print(coefficient)
        feature[index][coefficient + 50] += 1

#retuen numOf64 * (101 * 9)
def getPicFeature(pic, DCTFeature, SPAFeature, blockSize=64):
    print(pic.shape)
    x, y = pic.shape
    #64 * 64 分块
    x = math.ceil(x / blockSize)
    y = math.ceil(y / blockSize)

    #64 * 64 分块
    for subX in range(x):
        for subY in range(y):
            subPic = abs(pic[subX * blockSize: (subX + 1) * blockSize, subY * blockSize: (subY + 1) * blockSize])
            #空间域特征
            spaFeature = getSPAFeature(subPic.reshape((1, blockSize, blockSize, 1)))
            SPAFeature.append(spaFeature)

            #DCT特征
            dctFeature = np.zeros((9, 101))
            #8*8 分块
            for bX in range(8):
                for bY in range(8):
                    block = abs(subPic[bX * 8: (bX + 1) * 8, bY * 8: (bY + 1) * 8])
                    getDCTBlockFeature(block, dctFeature)
            dctFeature = np.reshape(dctFeature, (101 * 9, 1))
            DCTFeature.append(dctFeature)

def readDataSet(path, QF):
    DCTfeature = []
    SPAfeature = []

    for file in os.listdir(path):
        print(file)
        imagePath = os.path.join("%s%s", path, file)
        #print(imagePath)
        img = Image.open(imagePath)
        img = np.array(img)
        r = img[:, :, 0]
        g = img[:, :, 1]
        b = img[:, :, 2]
        Y = 0.299 * r + 0.587 * g + 0.114 * b
        #拓展维度
        Y = np.expand_dims(Y, axis=-1)
        Y = np.expand_dims(Y, axis=0)
        Y = tf.constant(Y - 128, dtype=tf.float32)
        dct_Y = PIC2DCT.dct_layer(Y, qtable=PIC2DCT.jpeg_qtable(QF))
        dct_Y_np = np.array(dct_Y)[0][0]
        getPicFeature(dct_Y_np, DCTfeature, SPAfeature)

    DCTfeature = np.array(DCTfeature)
    SPAfeature = np.array(SPAfeature)
    print(DCTfeature.shape)
    print(SPAfeature.shape)
    return DCTfeature, SPAfeature


if __name__ == '__main__':

    QF1 = [60, 65, 70, 75, 80, 85, 90, 95]
    QF2 = 75
    dir = ['UCID', 'UCID_Single', 'UCID_Double60_', 'UCID_Double65_', 'UCID_Double70_', 'UCID_Double75_',
           'UCID_Double80_', 'UCID_Double85_', 'UCID_Double90_', 'UCID_Double95_']

    # originPath = 'D:\\双重压缩检测\\MBDF&SPAM\\dataset\\UCID'
    # originDCT, originSPA = readDataSet(originPath, QF2)
    # np.save("QF75/originDCT", originDCT)
    # np.save("QF75/originSPA", originSPA)
    #
    # singlePath = 'D:\\双重压缩检测\\MBDF&SPAM\\dataset\\UCID_Single75'
    # singleDCT, singleSPA = readDataSet(singlePath, QF2)
    # np.save("QF75/singleDCT", singleDCT)
    # np.save("QF75/singleSPA", singleSPA)
    #
    # for qf in QF1:
    #     if qf == QF2:
    #         continue
    #     doublePath = "D:\\双重压缩检测\\MBDF&SPAM\\dataset\\UCID_Double" + str(qf) + "_" + str(QF2)
    #     doubleDCT, doubleSPA = readDataSet(doublePath, QF2)
    #     np.save("QF75/doubleDCT" + str(qf), doubleDCT)
    #     np.save("QF75/doubleSPA" + str(qf), doubleSPA)


    # path = 'D:\\双重压缩检测\\MBDF&SPAM\\dataset\\test'
    # dctFeature, spaFeature = readDataSet(path, 75)
    # np.save("QF75/originDCT", dctFeature)
    # np.save("QF75/originSPA", spaFeature)

    # originPath = 'D:\\双重压缩检测\\MBDF&SPAM\\dataset\\test'
    # originDCT, originSPA = readDataSet(originPath, QF2)
    # np.save("QF75/originDCT", originDCT)
    # np.save("QF75/originSPA", originSPA)
    #
    # singlePath = 'D:\\双重压缩检测\\MBDF&SPAM\\dataset\\test_single75'
    # singleDCT, singleSPA = readDataSet(singlePath, QF2)
    # np.save("QF75/singleDCT", singleDCT)
    # np.save("QF75/singleSPA", singleSPA)

    # for qf in QF1:
    #     if qf == QF2:
    #         continue
    qf = 95
    doublePath = "D:\\双重压缩检测\\MBDF&SPAM\\dataset\\test_double" + str(qf) + "-" + str(QF2)
    doubleDCT, doubleSPA = readDataSet(doublePath, QF2)
    np.save("QF75/doubleDCT" + str(qf), doubleDCT)
    np.save("QF75/doubleSPA" + str(qf), doubleSPA)
