from CASIAProcessor import DataPreprocessor
from CASIAProcessor import CNNModel, CNNModelDev
from CASIAProcessor import LFWProcessor
from CASIAProcessor import ResNetLFWProcessor
from keras import backend as K
import time

__author__ = 'Deliang Yang'
# FILE_PATH = 'F:/cse802_data/casia_mtcnn_cropped2/'

CLS_NUM = 225  # number of classes running every large step


def run():
    start_time = time.time()
    current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))
    print('Task begins. Time stamp: ' + current_time)

    function_flag = 2

    if function_flag == 0:
        
        for i in range(0, 47):
            epoch_start_time = time.time()
            start_index = CLS_NUM * i
            end_index = CLS_NUM * i + CLS_NUM
            print('Training range:', start_index, end_index)
            dp0 = DataPreprocessor(start_index, end_index)
            dp0.run()
            cnn_mdl0 = CNNModel(dp0.X_train, dp0.Y_train, 'v3')
            cnn_mdl0.run()
            if i in [8, 16, 24, 32, 40]:
                K.clear_session()
            del dp0
            del cnn_mdl0

            end_time = time.time()
            print('20 epochs, session time: ' + '%.3f' % (end_time - epoch_start_time) + ' s')
            print('---')

    elif function_flag == 1:
        dp0 = DataPreprocessor(0, 3)
        dp0.run()
        cnn_mdl0 = CNNModel(dp0.X_train, dp0.Y_train, 'v3')
        cnn_mdl0.inter_output()

    elif function_flag == 2:
        # convert LFW database to feature matrix CSV file
        lfwp0 = LFWProcessor('v7_p9_model_weights')
        lfwp0.run()

    elif function_flag == 3:
        # generating the development data set
        dp0 = DataPreprocessor(0, 100)
        dp0.get_file_list()
        dp0.load_pic_dev_pkl()

    elif function_flag == 4:
        cnn_mdl_dev0 = CNNModelDev()
        cnn_mdl_dev0.network_dev()
        # cnn_mdl_dev0.dev_fitting()

    elif function_flag == 5:
        cnn_mdl0 = CNNModel(None, None, 'v6_pc')
        for j in range(5):
            epoch_start_time = time.time()

            cnn_mdl0.flow_from_dir_fitting(str(j))

            # K.clear_session()

            end_time = time.time()

            print('5 epochs done, ' + '%.3f' % (end_time - epoch_start_time) + ' s. ', 'Total time lapse:',
                  '%.3f' % (end_time - start_time))
            print('---')

    elif function_flag == 6:
        # print model structure
        cnn_mdl0 = CNNModel(None, None, 'v7_p6')
        cnn_mdl0.network()
        print(cnn_mdl0.model.summary())

    elif function_flag == 7:
        resnet0 = ResNetLFWProcessor('res')
        resnet0.csv_generator()

    end_time = time.time()
    current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))

    print(current_time)
    print('Total execution time: ' + '%.3f' % (end_time - start_time) + ' s')


if __name__ == '__main__':
    run()
