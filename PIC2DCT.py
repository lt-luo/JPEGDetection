import numpy as np
import math
import tensorflow as tf
from PIL import JpegImagePlugin
from PIL import Image

PI = math.pi

#像素重组，低分辨->高分辨
def _phase_shift(I, r):
    bsize, a, b, c = I.get_shape().as_list()
    X = tf.reshape(I, (bsize, a, b, r, r))
    #print(X.shape)
    X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, 1, 1
    X = tf.split(X, a, 1)  # a, [bsize, b, r, r]
    X = tf.concat([tf.squeeze(x) for x in X], 1)  # bsize, b, a*r, r
    #print(X.shape)
    X = tf.split(X, b, 0)  # b, [bsize, a*r, r]
    X = tf.concat([tf.squeeze(x) for x in X], 1)  #
    #print(X.shape)
    #（bsize, 1, a*r, b*r）
    return tf.reshape(X, (bsize, 1, a*r, b*r))

# 利用卷积运算将像素矩阵（需转换为tensorflow张量）变换为DCT系数矩阵
# input_x：输入的待转换张量
# qtable：所用的量化表

def dct_layer(input_x, qtable=None):

    # 计算DCT基变换核
    k_h = 8
    k_w = 8
    d_h = 8
    d_w = 8

    dct_base = np.zeros([k_h, k_w, 1, k_h * k_w], dtype=np.float32)
    a = np.ones([k_h], dtype=np.float32) * math.sqrt(2.0 / k_h)
    a[0] = math.sqrt(1.0 / k_h)
    for u in range(0, k_h):
        for v in range(0, k_w):
            for x in range(0, k_h):
                for y in range(0, k_w):
                    dct_base[x, y, :, u * k_h + v] = a[u] * a[v] * math.cos(math.pi * u * (
                            2 * x + 1) / (k_h * 2.0)) * math.cos(math.pi * v * (2 * y + 1) / (k_w * 2.0))

    DCTBase = np.transpose(dct_base, (1, 0, 2, 3))      #(8, 8, 1, 64)，卷积需要做一个翻转
    DCTKernel = tf.constant(DCTBase, dtype=tf.float32)

    # 输入像素矩阵与基变换核进行卷积，并变换矩阵形状，阵此时得到的矩阵为未量化的DCT系数矩，并转成eg：（1，1，384，512）
    DCT = tf.nn.conv2d(input_x, DCTKernel, strides=(d_h, d_w), padding='SAME')
    DCT = _phase_shift(DCT, 8)

    bs = input_x.shape[0]

    # 量化，将输入的量化表变换为合适的形状，将上述DCT系数矩阵按元素除以量化表，并取整
    if qtable is not None:

        if len(qtable.shape) > 2:
            qtable = np.tile(qtable, [1, 1, input_x.shape[1] // 8, input_x.shape[2] // 8])
        else:
            qtable = np.tile(qtable, [bs, 1, input_x.shape[1] // 8, input_x.shape[2] // 8])

        QT = tf.constant(qtable, dtype=tf.float32)
        DCT = tf.round(tf.math.divide(DCT, QT))

    return DCT


# 由质量因子计算量化表（分为亮度量化表和色度量化表）
# qfactor：输入的质量因子
# chroma：是否计算色度通道的量化表
def jpeg_qtable(qfactor, chroma=False):
    t1 = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]])

    t2 = np.array([
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99]])

    if qfactor == -1:
        return None
    if qfactor < 50:
        qfactor = 5000 / qfactor
    else:
        qfactor = 200 - qfactor * 2
    if chroma:
        t = t2
    else:
        t = t1
    qtable = np.floor((t * qfactor + 50) / 100)

    qtable[qtable < 1] = 1
    qtable[qtable > 32767] = 32767
    return qtable

# 根据输入JPEG图像得到其量化表（亮度量化表）
def read_q_table(file_name):
    jpg = JpegImagePlugin.JpegImageFile(file_name)
    qtable = JpegImagePlugin.convert_dict_qtables(jpg.quantization)
    Y_qtable = qtable[0]
    Y_qtable_2d = np.zeros((8, 8))

    qtable_idx = 0
    for i in range(0, 8):
        for j in range(0, 8):
            Y_qtable_2d[i, j] = Y_qtable[qtable_idx]
            qtable_idx = qtable_idx + 1

    return Y_qtable_2d

if __name__ == '__main__':
    path = 'D:\\双重压缩检测\\MBDF&SPAM\\dataset\\UCID_Single75\\ucid00001.jpg'
    #PIL 读的结果是RGB
    img = Image.open(path)
    img = np.array(img)
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]
    Y = 0.299 * r + 0.587 * g + 0.114 * b
    #拓展维度
    Y = np.expand_dims(Y, axis=-1)
    Y = np.expand_dims(Y, axis=0)
    Y = tf.constant(Y - 128, dtype=tf.float32)
    dct_Y = dct_layer(Y, qtable=jpeg_qtable(75))
    print(dct_Y.shape)
    dct_Y_np = np.array(dct_Y)

    #test
    print(dct_Y_np[0, 0, 64: 72, 64: 72])