from __future__ import print_function, division
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
"""
DCGAN效果 < GAN
特徵提取 -> 由特徵轉成圖像
Gan 第一得先拉長延展，這麼做不太合理，於是就加入了反卷積去做 = DCGAN
原本使用卷積是利用卷積層去對圖像做特徵提取，但現在我們要做的是生成。必須先取得noise讓他去生成圖像。就是把整個過程給反過來了->反卷積 ->由特徵轉成一個實際的圖像
可選反卷積 也可選反池化，原本是做特徵壓縮，這邊給他做特徵放大
"""
class DCGAN():
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        optimizer = Adam(0.0002)

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        self.generator = self.build_generator()

        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        self.discriminator.trainable = False

        valid = self.discriminator(img)

        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(128 * 7 * 7, activation="relu", input_dim=self.latent_dim)) #將100維noise轉換為128*7*7的向量，h=7 w=7 的特徵圖有128個 (6272)
        model.add(Reshape((7, 7, 128))) #(7,7,128)
        model.add(UpSampling2D()) #反池化操作使特徵圖放大 (14,14,128)
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D()) #(28,28,128) 已經剛好與圖像(28,28)一樣大 ，接下目標就是將 128轉換成1
        model.add(Conv2D(64, kernel_size=3, padding="same")) #(28,28,64)
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(self.channels, kernel_size=3, padding="same")) #(28,28,1)
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same")) #(28,28,1)->(14,14,32)
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same")) #(7,7,64)
        model.add(ZeroPadding2D(padding=((0,1),(0,1)))) #(8,8,64)
        model.add(BatchNormalization(momentum=0.8)) #(8,8,64)
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same")) #(4,4,128)
        model.add(BatchNormalization(momentum=0.8)) #(4,4,128)
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same")) #(4,4,256)
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten()) #4096
        model.add(Dense(1, activation='sigmoid')) #1

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, iter, batch_size=128, save_interval=50):

        (X_train, _), (_, _) = mnist.load_data() #(60000,28,28)

        X_train = X_train / 127.5 - 1.
        X_train = np.expand_dims(X_train, axis=3)

        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for i in range(iter):

            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            g_loss = self.combined.train_on_batch(noise, valid)

            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (i, d_loss[0], 100*d_loss[1], g_loss))

            if i % save_interval == 0:
                self.save_imgs(i)

    def save_imgs(self, iter):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/mnist_%d.png" % iter)
        plt.close()


if __name__ == '__main__':
    dcgan = DCGAN()
    dcgan.train(iter=10000, batch_size=32, save_interval=50)
