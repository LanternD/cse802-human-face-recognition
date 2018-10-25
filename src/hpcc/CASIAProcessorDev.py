import os
import math
import csv
import time
import pickle
from pathlib import Path
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras import regularizers
from keras.optimizers import SGD, Adadelta, RMSprop
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D, MaxPooling1D
from keras.layers import ZeroPadding2D
from keras.utils import np_utils
from keras import backend as K
from keras import callbacks
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
# import PIL.Image
from keras.preprocessing import image as ki

ROOT_PATH = '/mnt/home/yangdeli/cse802proj/cnn_dev/'
FILE_PATH = './dev_dataset/' # '/mnt/research/CSE_802_SPR_17/casia_mtcnn_cropped/' # 'F:/cse802_data/casia_mtcnn_cropped/' 
LFW_PATH = 'F:/cse802_data/lfw_mtcnn_cropped2/'

TOTAL_CLS_NUM = 10575


class CNNModelDev(object):

    def __init__(self):

        self.model = Sequential()
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

        start_time = time.time()
        self.timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime(start_time))


    def load_dev_set(self):
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        with open(ROOT_PATH + 'dev_train_test.pkl', 'rb') as f_pkl:
            all_data = pickle.load(f_pkl)
            self.x_train = all_data[0][0]
            self.y_train = all_data[0][1]
            self.x_test = all_data[1][0]
            self.y_test = all_data[1][1]

            self.x_train = self.x_train.astype('float32')
            self.x_test = self.x_test.astype('float32')
            self.x_train /= 255
            self.x_test /= 255

            # print(self.x_train[0].tolist())
            # img = Image.fromarray(self.x_train[0], 'P')
            # img.show()

            self.y_train = np_utils.to_categorical(self.y_train, 100)  # convert to one-hot vector
            self.y_test = np_utils.to_categorical(self.y_test, 100)
            print('Training samples:', self.x_train.shape[0])
            # print(self.y_train.shape)
            del all_data
            f_pkl.close()

    def network_dev(self):
        input_shape = (110, 110, 3)
        kernel_size = (3, 3)
        self.model.add(Convolution2D(32, kernel_size,
                                     border_mode='same',
                                     input_shape=input_shape))
        self.model.add(Convolution2D(64, kernel_size))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D())

        self.model.add(Convolution2D(64, kernel_size))
        self.model.add(Convolution2D(128, kernel_size))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D())

        self.model.add(Convolution2D(96, kernel_size))
        self.model.add(Convolution2D(192, kernel_size))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D())

        self.model.add(Convolution2D(160, kernel_size))
        self.model.add(Convolution2D(320, kernel_size))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D())

        self.model.add(Convolution2D(512, kernel_size))
        # self.model.add(Convolution2D(512, kernel_size))
        self.model.add(Activation('relu'))
        # self.model.add(AveragePooling2D())  # avg pooling

        self.model.add(Flatten())
        self.model.add(Dropout(0.60))
        self.model.add(Dense(100))
        self.model.add(Activation('softmax'))

        my_opt = SGD(lr=0.005)

        self.model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
        print(self.model.summary())
        
        json_string = self.model.to_json()
        with open(ROOT_PATH + 'dev_result_output/' + self.timestamp + '_dev_CASIA_model.json', 'w') as f_json:
            f_json.write(json_string)
            f_json.close()

    def dev_fitting(self):
        self.load_dev_set()
        self.network_dev()

        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            featurewise_center=True,
            featurewise_std_normalization=True,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True)

        train_datagen.fit(self.x_train)

        test_datagen = ImageDataGenerator(rescale=1./255)
        test_datagen.fit(self.x_test)

        csv_path = ROOT_PATH + 'dev_result_output/' + self.timestamp + '_dev_log.log'
        csv_logger = callbacks.CSVLogger(csv_path, append=True)

        self.model.fit_generator(
            train_datagen.flow(self.x_train, self.y_train, batch_size=128),
            steps_per_epoch=115,
            verbose=1,
            validation_data=test_datagen.flow(self.x_test, self.y_test, batch_size=128),
            validation_steps=29,
            epochs=20, callbacks=[csv_logger])

# ==================================
#         batch_size = 128
#         nb_epoch = 20
#         self.load_dev_set()
#         self.network_dev()
        
#         csv_path = ROOT_PATH + 'dev_result_output/' + self.timestamp + '_dev_log.log'
#         csv_logger = callbacks.CSVLogger(csv_path, append=True)
#         self.model.fit(self.x_train, self.y_train,
#                        batch_size=batch_size,
#                        epochs=nb_epoch,
#                        verbose=2,
#                        validation_data=(self.x_test, self.y_test),
#                        callbacks=[csv_logger]
#                        )
