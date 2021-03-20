import tensorflow as tf
import numpy as np

drop_rate = 0.5

class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, num_channels):
        super(ConvBlock, self).__init__()
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.conv = tf.keras.layers.Conv2D(filters=num_channels,
                                           kernel_size=(3, 3),
                                           padding='same')
        self.listLayers = [self.bn, self.relu, self.conv]

    def call(self, x):
        y = x
        for layer in self.listLayers.layers:
            y = layer(y)
        y = tf.keras.layers.concatenate([x, y], axis=-1)
        return y

class DenseBlock(tf.keras.layers.Layer):
    def __init__(self, num_convs, num_channels):
        super(DenseBlock, self).__init__()
        self.listLayers = []
        for i in range(num_convs):
            self.listLayers.append(ConvBlock(num_channels[i]))

    def call(self, x):
        for layer in self.listLayers.layers:
            x = layer(x)
        return x

class TransitionBlock(tf.keras.layers.Layer):
    def __init__(self, num_channels, **kwargs):
        super(TransitionBlock, self).__init__(**kwargs)
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.conv = tf.keras.layers.Conv2D(num_channels, kernel_size=1)
        self.avg_pool = tf.keras.layers.AvgPool2D(pool_size=2, strides=2, padding='same')

    def call(self, x):
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.conv(x)
        return self.avg_pool(x)

def spatial_block_1():
    return tf.keras.Sequential([
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(32, kernel_size=3, strides=1, padding='same'),
        #原文没说这是max or avg
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='same')
    ], name='spatialNet')

def spatial_block_2():
    net = spatial_block_1()

    trans_channels = 32
    num_conv_channels = [32, 32, 16, 16, 32]
    num_convs_in_dense_block = [5, 5, 5, 5]

    for i, num_convs in enumerate(num_convs_in_dense_block):
        net.add(DenseBlock(num_convs, num_conv_channels))
        net.add(TransitionBlock(trans_channels))

    return net

def spatialBlock():
    net = spatial_block_2()
    net.add(tf.keras.layers.BatchNormalization())
    net.add(tf.keras.layers.ReLU())
    net.add(tf.keras.layers.GlobalAvgPool2D())
    net.add(tf.keras.layers.Flatten())
    net.add(tf.keras.layers.Dense(256))
    net.add(tf.keras.layers.Dropout(drop_rate))

    return net

def frequencyBlcok():
    #一维卷积 or 二维卷积？
    return tf.keras.Sequential([
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv1D(100, kernel_size=3),
        tf.keras.layers.MaxPool1D(strides=2, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv1D(100, kernel_size=3),
        tf.keras.layers.MaxPool1D(strides=2, padding='same'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256),
        tf.keras.layers.Dropout(drop_rate)
    ], name='frequencyNet')

def net():
    spatialInput = tf.keras.Input(shape=(64, 64, 1))
    frequencyInput = tf.keras.Input(shape=(909, 1))

    spatialModel = spatialBlock()(spatialInput)
    frequencyModel = frequencyBlcok()(frequencyInput)

    outputs = tf.keras.layers.concatenate([spatialModel, frequencyModel], axis=-1)
    outputs = tf.keras.layers.Dense(256)(outputs)
    outputs = tf.keras.layers.Dropout(drop_rate)(outputs)
    outputs = tf.keras.layers.Dense(10)(outputs)

    model = tf.keras.models.Model(inputs=[spatialInput, frequencyInput], outputs=outputs)
    # model.add(tf.keras.layers.Dense(256))
    # model.add(tf.keras.layers.Dropout(drop_rate))
    # model.add(tf.keras.layers.Dense(10))

    return model

model = net()
model.summary()


# fashion_mnist = tf.keras.datasets.fashion_mnist
# (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
# # train_images = tf.expand_dims(train_images / 255.0, -1)
# # test_images = tf.expand_dims(test_images / 255.0, -1)
# # train_labels = tf.convert_to_tensor(train_labels)
# # test_labels = tf.convert_to_tensor(test_labels)
# train_images = np.expand_dims(train_images / 255.0, 3)
# test_images = np.expand_dims(test_images / 255.0, 3)
# train_images = tf.constant(train_images, dtype=tf.float32)
# test_images = tf.constant(test_images, dtype=tf.float32)
# train_labels = tf.constant(train_labels, dtype=tf.float32)
# test_labels = tf.constant(test_labels, dtype=tf.float32)
#
# model = net()
# model.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])
# model.fit(train_images, train_labels, epochs=1)
# test_loss, test_acc = model.evaluate(test_images,  test_labels)
# print('\nTest accuracy:', test_acc)

