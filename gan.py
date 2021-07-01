from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt
import numpy as np

class GAN():
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100  #100維的噪音向量

        optimizer = Adam(0.0002)

        # 構建判別器
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # 構建生成器
        self.generator = self.build_generator()

        # 生成器輸入：噪音數據，輸出：生成的圖像數據
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # 在combined中只訓練生成器
        self.discriminator.trainable = False

        # 最後由判別器來判斷真假
        validity = self.discriminator(img)

        # 訓練生成器去騙過判別器
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)


    def build_generator(self):

        model = Sequential()

        model.add(Dense(256, input_dim=self.latent_dim)) #100維
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8)) #momentum: float >= 0. 參數，用於加速 SGD 在相關方向上前進，並抑制震盪。
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='tanh')) # 100 -> (784)28*28*1 np.prod->計算所有元素的乘積，預處理完後 值壓縮為-1 ~ 1之前 所以使用tanh
        model.add(Reshape(self.img_shape))
        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img) # input->100維 output->28*28*1的圖像

    def build_discriminator(self):

        model = Sequential()

        model.add(Flatten(input_shape=self.img_shape)) #拉平延展 -> 28*28*1
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid')) #LR 接sigmoid
        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity) #img->input 、 validity->output  https://keras.io/api/models/model/

    def train(self, iter, batch_size=128, sample_interval=50):

        # 加載數據集
        (X_train, _), (_, _) = mnist.load_data() #Return (x_train,y_train),(x_test,y_test) 但在這邊用不到標籤以及test，所以用_替代
        print('X_train.shape:',X_train.shape) # (60000,28,28) 3維 而剛剛定義的tensor是4維
        # 數據預處理
        X_train = X_train / 127.5 - 1.  #127.5->255/2 而127.5-1. 是為了將值限制在 -1 ~ 1 之間      -> 255/127.5 = 2  2-1 = 1
        X_train = np.expand_dims(X_train, axis=3)
        print('X_train.shape:',X_train.shape) # (60000,28,28,1)
        # 制作標簽
        valid = np.ones((batch_size, 1)) #1
        fake = np.zeros((batch_size, 1)) #0

        for i in range(iter):

            #訓練判別器
            #訓練一個batch數據
            idx = np.random.randint(0, X_train.shape[0], batch_size) #在60000個數據中隨機選32個(batch_size=32)當作這一批的輸入
            imgs = X_train[idx] #通過索引找到對應的數據 -> (64,28,28,1) 真數據

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim)) #np.random.normal(loc,scale,size) loc=機率分布的平均 scale=機率分布的標準差 size=返回數組的形狀 ->np.random.normal(0, 1, (32, 100))

            # 獲取生成數據
            gen_imgs = self.generator.predict(noise) #預測出來的數據即為生成數據

            # 訓練判別器
            d_loss_real = self.discriminator.train_on_batch(imgs, valid) #train_on_batch ->執行一次梯度更新在一個batch的數據上 ，也就是進行一次迭代更新。 把真實數據判別為真
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake) #生成出來的數據判別為假
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # 訓練生成器
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim)) 

            # 讓生成器騙過判別器
            g_loss = self.combined.train_on_batch(noise, valid) #聯合訓練，但不訓練D，希望將生成出來的數據判別為真。達到以假亂真的效果

            # 打印訓練結果
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (i, d_loss[0], 100*d_loss[1], g_loss)) #隨著迭代次數上升 [D loss往上 acc往下]這邊的acc為判別出造假的機率 G loss往下

            # 保存部分迭代效果
            if i % sample_interval == 0:
                self.sample_images(i) #每隔i次將當前的結果儲存下來，這邊i=500

    def sample_images(self, iter):
        r, c = 5, 5 #換一子圖 5*5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise) #這裡為-1 ~ 1之前，所以必須將值還原回去

        # 預處理還原
        gen_imgs = 0.5 * gen_imgs + 0.5 #先還原到0 ~ 1之間

        fig, axs = plt.subplots(r, c)
        cnt = 0 #當前位置
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/%d.png" % iter)
        plt.close()


if __name__ == '__main__':
    gan = GAN()
    gan.train(iter=50000, batch_size=32, sample_interval=500)
