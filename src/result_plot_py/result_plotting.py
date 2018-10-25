from matplotlib import pyplot as plt
import numpy as np
import csv

"""
This code primarily plot the affect of the dropout rate.
"""
__author__ = 'Deliang Yang & Mengying Sun'
RST_PATH = './'  # result path


def csv_log_reader(data_file_name):
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []

    # data_file_name = 'basic_mnist_log_wrong_c'
    print('Processing file: ' + data_file_name)
    with open(RST_PATH + data_file_name + '.log', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        next(spamreader, None)  # skip header
        for line_buf in spamreader:

            train_loss.append(float(line_buf[2]))
            train_acc.append(float(line_buf[1]))
            if len(line_buf) > 4:
                test_loss.append(float(line_buf[4]))
                test_acc.append(float(line_buf[3]))

        csvfile.close()
    if False:
        test_acc = smoothing(test_acc, 0.6)
        test_loss = smoothing(test_loss, 0.6)

    train_loss = np.asarray(train_loss)
    train_acc = np.asarray(train_acc)
    if len(test_loss) > 2:
        test_loss = np.asarray(test_loss)
        test_acc = np.asarray(test_acc)
    if False:
        train_loss = np.log10(train_loss)
        test_loss = np.log10(test_loss)
    if len(test_loss) > 2:
        return train_loss, train_acc, test_loss, test_acc
    else:
        return train_loss, train_acc


def dp_rate_result_plot():
    e_trn_loss, e_trn_acc, e_tst_loss, e_tst_acc = csv_log_reader('20170425_145632_dev_log')
    s_trn_loss, s_trn_acc, s_tst_loss, s_tst_acc = csv_log_reader('20170422_171138_dev_log')
    l_trn_loss, l_trn_acc, l_tst_loss, l_tst_acc = csv_log_reader('20170422_222212_dev_log')
    b_trn_loss, b_trn_acc, b_tst_loss, b_tst_acc = csv_log_reader('20170422_224636_dev_log')

    xx = np.linspace(0, 19, 20)

    f, axarr = plt.subplots(2, 2, figsize=(12, 10))
    axarr[0, 0].plot(xx, e_trn_loss, '-.')
    axarr[0, 0].plot(xx, s_trn_loss, '--')
    axarr[0, 0].plot(xx, l_trn_loss, '-')
    axarr[0, 0].plot(xx, b_trn_loss[:100], ':')
    axarr[0, 0].set_xlabel('Epoch #', fontsize='large')
    axarr[0, 0].set_ylabel('Train loss', fontsize='large')
    axarr[0, 0].grid(True)
    # axarr[0, 0].set_xlim(1, 21)

    axarr[0, 1].plot(xx, e_trn_acc, '-.')
    axarr[0, 1].plot(xx, s_trn_acc, '--')
    axarr[0, 1].plot(xx, l_trn_acc, '-')
    axarr[0, 1].plot(xx, b_trn_acc, ':')
    axarr[0, 1].set_xlabel('Epoch #', fontsize='large')
    axarr[0, 1].set_ylabel('Train accuracy', fontsize='large')
    axarr[0, 1].grid(True)

    axarr[1, 0].plot(xx, e_tst_loss, '-.')
    axarr[1, 0].plot(xx, s_tst_loss, '--')
    axarr[1, 0].plot(xx, l_tst_loss, '-')
    axarr[1, 0].plot(xx, b_tst_loss, ':')
    axarr[1, 0].set_xlabel('Epoch #', fontsize='large')
    axarr[1, 0].set_ylabel('Test loss', fontsize='large')
    axarr[1, 0].grid(True)

    axarr[1, 1].plot(xx, e_tst_acc, '-.', label='Dropout=0.2')
    axarr[1, 1].plot(xx, s_tst_acc, '--', label='Dropout=0.4')
    axarr[1, 1].plot(xx, l_tst_acc, '-', label='Dropout=0.6')
    axarr[1, 1].plot(xx, b_tst_acc, ':', label='Dropout=0.8')
    axarr[1, 1].set_xlabel('Epoch #', fontsize='large')
    axarr[1, 1].set_ylabel('Test accuracy', fontsize='large')
    axarr[1, 1].grid(True)

    plt.legend(loc='upper left', bbox_to_anchor=(0.32, 2.6), borderaxespad=0., fontsize='large')
    f.subplots_adjust(bottom=0.11, top=0.86, left=0.07, right=0.95)

    plt.savefig('./' + 'comparison_dropout_log' + '.png')
    plt.savefig('./' + 'comparison_dropout_log' + '.pdf')
    plt.show()


def normal_loss_acc_plot():
    e_trn_loss, e_trn_acc = csv_log_reader('v7_train_log')

    xx = np.linspace(0, len(e_trn_acc) - 1, len(e_trn_acc))

    fig, axarr = plt.subplots(2, 1, figsize=(8, 8))

    axarr[0].plot(xx, e_trn_loss, '-', label='Train loss', linewidth=3)
    axarr[0].set_xlabel('Epoch #', fontsize='large')
    axarr[0].set_ylabel('Train loss', fontsize='large')
    axarr[0].grid(True)
    # axarr[0].legend(loc='upper left', bbox_to_anchor=(0.60, -0.12), borderaxespad=0., fontsize='large')

    axarr[1].plot(xx, e_trn_acc, 'r--', label='Train accuracy', linewidth=3)
    axarr[1].set_xlabel('Epoch #', fontsize='large')
    axarr[1].set_ylabel('Train accuracy', fontsize='large')
    axarr[1].grid(True)
    # plt.legend(loc='upper left', bbox_to_anchor=(0.60, -0.12), borderaxespad=0., fontsize='large')
    plt.tight_layout(h_pad=1.0)
    fig.subplots_adjust(bottom=0.10, top=0.95, left=0.10, right=0.95)

    # plt.savefig('./' + 'train_loss_acc_log' + '.png')
    plt.savefig('./' + 'train_loss_acc_log' + '.pdf')
    plt.show()


def twin_axis_plot():
    e_trn_loss, e_trn_acc = csv_log_reader('v7_train_log')

    xx = np.linspace(0, len(e_trn_acc) - 1, len(e_trn_acc))

    fig, axarr = plt.subplots(figsize=(6, 4))

    axarr.plot(xx, e_trn_loss, '-', label='Train loss', linewidth=3)
    axarr.set_xlabel('Epoch #', fontsize='large')
    axarr.set_ylabel('Train loss', fontsize='large')
    axarr.set_ylim([0, 9])
    axarr.grid(True)
    axarr.legend(loc='upper left', bbox_to_anchor=(0.55, 0.95), borderaxespad=0., fontsize='large')

    ax2 = axarr.twinx()
    ax2.plot(xx, e_trn_acc, 'r--', label='Train accuracy', linewidth=3)
    # ax2.set_xlabel('Epoch #', fontsize='large')
    ax2.set_ylabel('Train accuracy', fontsize='large')
    ax2.set_ylim([0, 1])
    # ax2.grid(True)
    ax2.legend(loc='upper left', bbox_to_anchor=(0.55, 0.85), borderaxespad=0., fontsize='large')
    fig.subplots_adjust(bottom=0.15, top=0.95, left=0.10, right=0.85)

    # plt.savefig('./' + 'train_loss_acc_log' + '.png')
    plt.savefig('./' + 'train_loss_acc_log' + '.pdf')
    plt.show()


def dir_vr_plot():
    pca_vr = [22.93, 30.20, 28.97, 40.21, 39.57, 40.97, 43.79, 45.43, 48.39, 48.38]
    pca_dir = [2.47, 3.31, 3.48, 3.51, 3.50, 3.50, 4.12, 4.17, 4.25, 4.37]
    lda_vr = [22.49, 31.98, 34.99, 40.54, 40.90, 42.57, 45.89, 46.90, 47.45, 48.46]
    lda_dir = [3.63, 4.08, 4.09, 4.73, 4.36, 4.83, 4.65, 5.18, 4.46, 4.39]

    xx = np.linspace(10, 10 * (len(pca_vr)), len(pca_vr))
    fig, ax = plt.subplots(figsize=(8, 5))

    plt.plot(xx, lda_vr, '-.', label='LDA - VR', linewidth=2)
    plt.plot(xx, pca_vr, '--', label='PCA - VR', linewidth=2)
    ax.legend(loc='upper left', bbox_to_anchor=(0.45, 0.2), borderaxespad=0., fontsize='large')
    ax.set_ylabel('Verification rate (%)', fontsize='large')
    ax.set_ylim(20, 60)
    ax2 = ax.twinx()

    ax2.plot(xx, lda_dir, ':', label='LDA - DIR', linewidth=2)
    ax2.plot(xx, pca_dir, '-', label='PCA - DIR', linewidth=2)
    ax2.legend(loc='upper left', bbox_to_anchor=(0.70, 0.2), borderaxespad=0., fontsize='large')
    ax2.set_ylabel('Open-set identification (%)', fontsize='large')
    ax2.set_ylim(0, 5.5)

    ax.set_xlabel('epochs', fontsize='large')

    plt.grid(True)

    plt.savefig('./' + 'cnn_model7_vr_dir' + '.pdf')
    plt.show()


def smoothing(data_list, smv):
    list_length = len(data_list)
    smooth_value = data_list[0]
    return_list = [smooth_value]
    for i in range(1, list_length):
        smooth_value = smooth_value * smv + data_list[i] * (1 - smv)
        return_list.append(smooth_value)
    # print(data_list)
    # print(return_list)
    return return_list


if __name__ == '__main__':
    # dp_rate_result_plot()
    # normal_loss_acc_plot()
    dir_vr_plot()
    # twin_axis_plot()
