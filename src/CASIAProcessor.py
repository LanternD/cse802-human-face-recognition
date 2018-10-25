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

from keras.applications.resnet50 import ResNet50

ROOT_PATH = './'#'/mnt/home/yangdeli/cse802proj/'
FILE_PATH = 'F:/cse802_data/casia_mtcnn_cropped/' # '/mnt/research/CSE_802_SPR_17/casia_mtcnn_cropped/'
LFW_PATH = 'F:/cse802_data/lfw_mtcnn_cropped2/'

TOTAL_CLS_NUM = 10575


class DataPreprocessor(object):
    """
    Task:
    1. decode jpg file into 3-layer tensor
    2. split the images in every folder into training and testing set
    3. load every images into RAM
    """
    def __init__(self, start_index, end_index):
        self.data = []
        self.dirs = []
        self.file_list = []
        self.file_num_list = []
        self.train_total_num = 0
        self.start_index = start_index
        self.end_index = end_index
        self.X_train = []
        self.Y_train = []

    def dir_list_gen(self):
        self.dirs = sorted(os.listdir(FILE_PATH))
        print('Number of subdirectories in ' + FILE_PATH + ': ' + str(len(self.dirs)))

    def get_file_list(self):
        self.dir_list_gen()

        self.file_list = []
        for dirss in self.dirs[self.start_index: self.end_index]:
            file_path = FILE_PATH + dirss + '/'
            files = []
            for root, dirs, files in os.walk(file_path):
                # print('Number of files in PATH \'' + file_path + '\': ' + str(len(files)))
                pass
            self.file_list.append(files)
            self.file_num_list.append(len(files))  # record the num of jpgs in each folder.
        # print(self.file_num_list)
        self.train_total_num = sum(self.file_num_list)

    def load_pic_data(self):
        # with train-test split
        train_img_all = []

        # print('Number of classes: ' + str(self.end_index - self.start_index))
        for i in range(self.start_index, self.end_index):
            # traverse all the classes
            file_path_queue = []
            for pic in self.file_list[i-self.start_index]:
                file_path_queue.append(FILE_PATH + self.dirs[i] + '/' + pic)

            for img_name in file_path_queue:
                img = ki.load_img(img_name, target_size=(110, 110))
                xx_trn = ki.img_to_array(img)
                # xx_trn = xx_trn.reshape(3, 110, 110)
                # print(xx.shape)
                train_img_all.append(xx_trn)
            self.Y_train += self.file_num_list[i-self.start_index] * [i]
        
        self.X_train = np.asarray(train_img_all)
        # print(all_img)
        # print(self.X_train.shape)
        # print(len(self.Y_train))

    def load_pic_dev_pkl(self):
        # with train-test split
        train_img_all = []
        test_img_all = []
        train_tag_all = []
        test_tag_all = []
        print(len(self.file_list))
        # print('Number of classes: ' + str(self.end_index - self.start_index))
        for i in range(self.end_index - self.start_index):
            # traverse all the classes
            file_path_queue = []
            for pic in self.file_list[i]:
                file_path_queue.append(FILE_PATH + self.dirs[self.start_index + i] + '/' + pic)

            test_size = int(len(file_path_queue)/5)
            print('Class:', self.dirs[self.start_index + i], 'total pics:', len(file_path_queue),
                  'train size:', 4 * test_size, 'test size:', test_size)
            for img_name in file_path_queue[:4 * test_size]:
                img = ki.load_img(img_name, target_size=(110, 110))
                xx_trn = ki.img_to_array(img)
                # xx_trn = xx_trn.reshape(3, 110, 110)
                # print(xx.shape)
                train_img_all.append(xx_trn)
                train_tag_all.append([i])
            for img_name in file_path_queue[-test_size:]:
                img = ki.load_img(img_name, target_size=(110, 110))
                # xx_tst = np.array(img)
                xx_tst = ki.img_to_array(img)
                test_img_all.append(xx_tst)
                test_tag_all.append([i])
        X_train = np.asarray(train_img_all)
        X_test = np.asarray(test_img_all)
        Y_train = np.asarray(train_tag_all)
        Y_test = np.asarray(test_tag_all)
        print('Total train size:', X_train.shape, Y_train.shape)
        print('Total test size:', X_test.shape, Y_test.shape)

        data_set = ((X_train, Y_train), (X_test, Y_test))
        f_save_pkl = open(ROOT_PATH + 'cnn_dev/dev_train_test.pkl', 'wb')
        pickle.dump(data_set, f_save_pkl, pickle.HIGHEST_PROTOCOL)
        f_save_pkl.close()

    def normalizer(self):
        self.X_train = self.X_train.astype('float32')
        self.X_train /= 255

    def run(self):
        self.get_file_list()
        self.load_pic_data()
        self.normalizer()


class CNNModel(object):

    def __init__(self, X_train, Y_train, model_name):
        self.X_train = X_train
        self.Y_train = Y_train
        self.model = Sequential()
        self.batch_size = 128
        self.nb_epoch = 10
        self.model_h5_name = model_name
        # self.start_time_stamp = time.strftime('%Y%m%d_%H%M%S', time.localtime(start_time))
        # K.set_image_dim_ordering('th')

    def network(self):

        # input image dimensions
        img_rows, img_cols = 110, 110

        input_shape = (3, img_rows, img_cols)
        # print(self.X_train.shape[0], 'train samples')

        # convert class vectors to binary class matrices
        # self.Y_train = np_utils.to_categorical(self.Y_train, TOTAL_CLS_NUM)  # convert to one-hot vector
        # print(self.Y_train[2])
        model_h5_file = Path(ROOT_PATH + self.model_h5_name + '_model_weights.h5')
        if model_h5_file.is_file():
            self.model = load_model(ROOT_PATH + self.model_h5_name + '_model_weights.h5')
        else:
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

            self.model.add(Convolution2D(128, kernel_size))
            self.model.add(Convolution2D(256, kernel_size))
            self.model.add(Activation('relu'))
            self.model.add(MaxPooling2D())

            self.model.add(Convolution2D(256, kernel_size))
            self.model.add(Convolution2D(512, kernel_size))
            self.model.add(Activation('relu'))
            self.model.add(AveragePooling2D())  # avg pooling

            self.model.add(Flatten())
            self.model.add(Dropout(0.60))
            self.model.add(Dense(TOTAL_CLS_NUM))
            self.model.add(Activation('softmax'))

            self.model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

            # print('Saving the model as .json and .h5 file')
            json_string = self.model.to_json()
            with open(ROOT_PATH + 'result_output/' + self.model_h5_name + '_CASIA_model.json', 'w') as f_json:
                f_json.write(json_string)
                f_json.close()
            self.model.save(ROOT_PATH + self.model_h5_name + '_model_weights.h5')

    def model_fitting(self):
        csv_path = ROOT_PATH + 'result_output/' + self.model_h5_name + '_train_log.log'

        csv_logger = callbacks.CSVLogger(csv_path, append=True)

        self.model.fit(self.X_train, self.Y_train,
                       batch_size=self.batch_size,
                       epochs=self.nb_epoch,
                       verbose=2,
                       # validation_data=(self.X_test, self.Y_test),
                       callbacks=[csv_logger])
        self.model.save(ROOT_PATH + self.model_h5_name + '_model_weights.h5')

    def run(self):
        self.network()
        self.model_fitting()
        # K.clear_session()

    def inter_output(self):
        model = load_model('./' + self.model_h5_name + '_model_weights.h5')

        layer_name = 'flatten_1'
        intermediate_layer_model = Model(inputs=model.input,
                                         outputs=model.get_layer(layer_name).output)
        intermediate_output = intermediate_layer_model.predict(self.X_train[0:2]).tolist()
        print(intermediate_output)

    def flow_from_dir_fitting(self, progress):
        self.network()
        train_datagen = ImageDataGenerator(
            rescale=1. / 255)
        train_generator = train_datagen.flow_from_directory(
            FILE_PATH,
            target_size=(110, 110),
            batch_size=128,
            class_mode='categorical')

        csv_path = ROOT_PATH + 'result_output/' + self.model_h5_name + '_train_log.log'
        csv_logger = callbacks.CSVLogger(csv_path, append=True)

        self.model.fit_generator(
            train_generator,
            steps_per_epoch=3750,
            verbose=2,
            epochs=5, callbacks=[csv_logger])

        self.model.save(ROOT_PATH + self.model_h5_name + '_model_weights.h5')
        self.model.save(ROOT_PATH + self.model_h5_name + '_p' + progress + '_model_weights.h5')


class LFWProcessor(object):
    def __init__(self, model_name):
        self.dirs = []
        self.file_list = []
        self.file_num_list = []
        self.train_total_num = 0
        self.X_train = []
        self.model_name = model_name

    def dir_list_gen(self):
        self.dirs = os.listdir(LFW_PATH)
        print('Number of subdirectories in ' + LFW_PATH + ': ' + str(len(self.dirs)))

    def get_file_list(self):
        self.dir_list_gen()
        self.file_list = []
        for dirss in self.dirs:
            file_path = LFW_PATH + dirss + '/'
            files = []
            for root, dirs, files in os.walk(file_path):
                # print('Number of files in PATH \'' + file_path + '\': ' + str(len(files)))
                pass
            self.file_list.append(files)
            self.file_num_list.append(len(files))  # record the num of jpgs in each folder.
        # print(self.file_num_list)
        self.train_total_num = sum(self.file_num_list)

    def load_pic_data(self):
        # with train-test split
        train_img_all = []

        # print('Number of classes: ' + str(self.end_index - self.start_index))
        for i in range(len(self.file_list)):
            # traverse all the classes
            file_path_queue = []
            for pic in self.file_list[i]:

                file_path_queue.append(LFW_PATH + self.dirs[i] + '/' + pic)
            for img_name in file_path_queue:
                img = ki.load_img(img_name, target_size=(110, 110))
                xx_trn = ki.img_to_array(img)
                # xx_trn = xx_trn.reshape(3, 110, 110)
                # print(xx.shape)
                train_img_all.append(xx_trn)
        self.X_train = np.asarray(train_img_all)
        # print(all_img)
        print('LFW data shape:', self.X_train.shape)
        # print(len(self.Y_train))

    def normalizer(self):
        self.X_train = self.X_train.astype('float32')
        self.X_train /= 255

    def csv_generator(self):

        with open('./lfw_feature_matrix.csv', 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            model = load_model('./' + self.model_name + '.h5')
            # layer_config = model.get_layer("flatten_1").get_config()
            # print(layer_config)
            print(model.summary())
            layer_name = 'dropout_1'
            intermediate_layer_model = Model(inputs=model.input,
                                             outputs=model.get_layer(layer_name).output)
            # for pics in self.X_train:
            intermediate_output = intermediate_layer_model.predict(self.X_train)
            print(intermediate_output.shape)
            for element in intermediate_output:
                csv_writer.writerow(element)
            # print(intermediate_output)
            csvfile.close()
        print('Feature matrix CSV has been created.')

    def run(self):
        self.get_file_list()
        self.load_pic_data()
        self.normalizer()
        self.csv_generator()


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
        with open(ROOT_PATH + 'cnn_dev/dev_train_test.pkl', 'rb') as f_pkl:
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

        self.model.add(Convolution2D(128, kernel_size))
        self.model.add(Convolution2D(256, kernel_size))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D())

        self.model.add(Convolution2D(192, kernel_size))
        self.model.add(Convolution2D(320, kernel_size))
        self.model.add(Activation('relu'))
        self.model.add(AveragePooling2D((6, 6)))  # avg pooling

        self.model.add(Flatten())
        self.model.add(Dropout(0.60))
        self.model.add(Dense(100))
        self.model.add(Activation('softmax'))

        self.model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
        # print('Saving the model as .json and .h5 file')
        print(self.model.summary())
        # json_string = self.model.to_json()
        # with open(ROOT_PATH + 'result_output/' + self.timestamp + '_dev_CASIA_model.json', 'w') as f_json:
        #     f_json.write(json_string)
        #     f_json.close()

    def dev_fitting(self):
        
        batch_size = 128
        nb_epoch = 20
        self.load_dev_set()
        self.network_dev()
        print(self.model.summary())
        csv_path = ROOT_PATH + 'result_output/' + self.timestamp + '_dev_log.log'
        csv_logger = callbacks.CSVLogger(csv_path, append=True)
        self.model.fit(self.x_train, self.y_train,
                       batch_size=batch_size,
                       epochs=nb_epoch,
                       verbose=2,
                       validation_data=(self.x_test, self.y_test),
                       callbacks=[csv_logger]
                       )


class ResNetLFWProcessor(LFWProcessor):

    def load_pic_data(self):
        # with train-test split
        train_img_all = []

        # print('Number of classes: ' + str(self.end_index - self.start_index))
        for i in range(len(self.file_list)):
            # traverse all the classes
            file_path_queue = []
            for pic in self.file_list[i]:

                file_path_queue.append(LFW_PATH + self.dirs[i] + '/' + pic)
            for img_name in file_path_queue:
                img = ki.load_img(img_name, target_size=(224, 224))
                xx_trn = ki.img_to_array(img)
                # xx_trn = xx_trn.reshape(3, 110, 110)
                # print(xx.shape)
                train_img_all.append(xx_trn)
        self.X_train = np.asarray(train_img_all)
        # print(all_img)
        print('LFW data shape:', self.X_train.shape)

    def csv_generator(self):

        with open('./lfw_feature_matrix_res.csv', 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            model = ResNet50(weights='imagenet')
            # layer_config = model.get_layer("flatten_1").get_config()
            # print(layer_config)
            print(model.summary())
            datagen = ImageDataGenerator(rescale=1. / 255)
            lfw_generator = datagen.flow_from_directory(
                LFW_PATH,
                shuffle=False,
                target_size=(224, 224),
                batch_size=64,
                class_mode='categorical')
            # model.predict_generator()
            layer_name = 'flatten_1'
            intermediate_layer_model = Model(inputs=model.input,
                                             outputs=model.get_layer(layer_name).output)
            # # for pics in self.X_train:
            intermediate_output = intermediate_layer_model.predict_generator(lfw_generator, steps=207,
            verbose=1)
            print(intermediate_output.shape)
            # for element in intermediate_output:
            #     csv_writer.writerow(element)
            # print(intermediate_output)
            csvfile.close()
        print('Feature matrix CSV has been created.')
