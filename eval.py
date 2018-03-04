import csv
import os
import pprint

from ais import estimate_log_prob

os.environ['GLOG_minloglevel'] = '2'

import pickle
import scipy.io
from tqdm import tqdm

import caffe
import numpy as np
import random
from sklearn.metrics import average_precision_score, normalized_mutual_info_score
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import shutil
from tsne import tsne


caffe.set_mode_cpu()

datasets = {
    'mnist_binary': [
        scipy.io.loadmat('datasets/mnist_binary/trainingX.mat')['trainingX'],
        scipy.io.loadmat('datasets/mnist_binary/trainingY.mat')['trainingY'],
        scipy.io.loadmat('datasets/mnist_binary/validationX.mat')['validationX'],
        scipy.io.loadmat('datasets/mnist_binary/validationY.mat')['validationY'],
        scipy.io.loadmat('datasets/mnist_binary/testX.mat')['testX'],
        scipy.io.loadmat('datasets/mnist_binary/testY.mat')['testY']
    ],
    'omniglot': pickle.load(open('datasets/omniglot/data.pkl', 'rb')),
    '20Newsgroup': [
        scipy.io.loadmat('datasets/20Newsgroup/trainingX.mat')['trainingX'],
        scipy.io.loadmat('datasets/20Newsgroup/trainingY.mat')['trainingY'],
        scipy.io.loadmat('datasets/20Newsgroup/validationX.mat')['validationX'],
        scipy.io.loadmat('datasets/20Newsgroup/validationY.mat')['validationY'],
        scipy.io.loadmat('datasets/20Newsgroup/testX.mat')['testX'],
        scipy.io.loadmat('datasets/20Newsgroup/testY.mat')['testY']
    ],

}

dataset_filter = ['20Newsgroup']
save_filter_tsne = [
    # 'experiments/rbm_cd/mnist_binary/0.001_0_1e-05_10_500',
    # 'experiments/rbm_cd/omniglot/0.001_0_1e-05_10_500',
    # 'experiments/rbm_pm1_geo/mnist_binary/0.01_0.9_1e-05_10_500',
    # 'experiments/rbm_pm1_geo/mnist_binary/0.001_0_1e-05_10_500',
    # 'experiments/rbm_pm1_geo/omniglot/0.001_0_1e-05_10_500',
    # 'experiments/rbm_pm1_cd/mnist_binary/0.001_0_0_10_500',
    # 'experiments/rbm_pm1_cd/mnist_binary/0.01_0.9_0_10_500',
    # 'experiments/rbm_pm1_cd/omniglot/0.01_0_1e-05_10_500',
    # 'experiments/rbm_pm1_cd/omniglot/0.001_0_0_10_500',

    # 'experiments/rbm_pm1_geo/20Newsgroup/0.001_0.9_0_10_500',
    # 'experiments/rbm_pm1_cd/20Newsgroup/0.001_0.9_0.0001_5_500',
    # 'experiments/rbm_pm1_geo/20Newsgroup/0.001_0_1e-05_1_500',
    # 'experiments/rbm_pm1_geo/20Newsgroup/0.001_0.9_0.0001_1_500',
    # 'experiments/rbm_pm1_cd/20Newsgroup/0.01_0_0.001_5_500',
    # 'experiments/rbm_cd/20Newsgroup/0.01_0_0.001_10_500',
    # 'experiments/rbm_pm1_cd/20Newsgroup/0.01_0_0.001_1_500',
    # 'experiments/rbm_pm1_cd/20Newsgroup/0.001_0_0.001_1_500',
    # 'experiments/rbm_cd/20Newsgroup/0.01_0_0.001_5_500',
    # 'experiments/rbm_pm1_cd/20Newsgroup/0.001_0.9_0.0001_10_500',
    # 'experiments/rbm_pm1_geo/20Newsgroup/0.01_0_0.0001_10_500',
    # 'experiments/rbm_cd/20Newsgroup/0.01_0_0.001_1_500',
    # 'experiments/rbm_pm1_cd/20Newsgroup/0.001_0.9_0.001_10_500',
    # 'experiments/rbm_pm1_cd/20Newsgroup/0.01_0_0.001_10_500',
    # 'experiments/rbm_pm1_geo/20Newsgroup/0.01_0_1e-05_5_500',
    # 'experiments/rbm_cd/20Newsgroup/0.001_0.9_0.001_5_500',
    # 'experiments/rbm_pm1_geo/20Newsgroup/0.001_0.9_0.0001_5_500',

    'experiments/rbm_cd/20Newsgroup/0.01_0_0.0001_1_50',
    'experiments/rbm_cd/20Newsgroup/0.001_0.9_1e-05_5_50',
    'experiments/rbm_pm1_geo/20Newsgroup/0.001_0.9_0.0001_10_50',
    'experiments/rbm_cd/20Newsgroup/0.01_0_0.001_1_50',
    'experiments/rbm_pm1_cd/20Newsgroup/0.001_0.9_0.0001_10_50',
    'experiments/rbm_pm1_cd/20Newsgroup/0.01_0_0.0001_10_50',
    'experiments/rbm_pm1_geo/20Newsgroup/0.01_0_0_1_50',
    'experiments/rbm_pm1_geo/20Newsgroup/0.001_0.9_1e-05_10_50',
    'experiments/rbm_cd/20Newsgroup/0.1_0.9_0.001_10_50',
    'experiments/rbm_cd/20Newsgroup/0.001_0.9_0.001_5_50',
    'experiments/rbm_pm1_cd/20Newsgroup/0.01_0_0_10_50',
    'experiments/rbm_cd/20Newsgroup/0.01_0.9_0.001_5_50',
    'experiments/rbm_pm1_cd/20Newsgroup/0.1_0.9_1e-05_5_50',
    'experiments/rbm_pm1_cd/20Newsgroup/0.01_0_0.001_1_50',
    'experiments/rbm_pm1_cd/20Newsgroup/0.01_0_0_5_50',
    'experiments/rbm_pm1_geo/20Newsgroup/0.01_0_0.0001_5_50',
    'experiments/rbm_cd/20Newsgroup/0.01_0_0_10_50',
    'experiments/rbm_pm1_geo/20Newsgroup/0.001_0.9_0.0001_1_50',
    'experiments/rbm_pm1_geo/20Newsgroup/0.001_0.9_0_1_50',
]

steps = [1, 5, 10]
eps = 1e-8
model_dirs = ['experiments/rbm_cd', 'experiments/rbm_pm1_cd', 'experiments/rbm_pm1_geo']

ALGS = [
    # 'mnist_binary_experiments/rbm_cd_1',
    # 'mnist_binary_experiments/rbm_pm1_cd_1',
    # 'mnist_binary_experiments/rbm_pm1_geo_1',
    #
    # 'mnist_binary_experiments/rbm_cd_5',
    # 'mnist_binary_experiments/rbm_pm1_cd_5',
    # 'mnist_binary_experiments/rbm_pm1_geo_5',
    #
    # 'mnist_binary_experiments/rbm_cd_10',
    # 'mnist_binary_experiments/rbm_pm1_cd_10',
    # 'mnist_binary_experiments/rbm_pm1_geo_10',
    #
    # 'omniglot_experiments/rbm_cd_1',
    # 'omniglot_experiments/rbm_pm1_cd_1',
    # 'omniglot_experiments/rbm_pm1_geo_1',
    #
    # 'omniglot_experiments/rbm_cd_5',
    # 'omniglot_experiments/rbm_pm1_cd_5',
    # 'omniglot_experiments/rbm_pm1_geo_5',
    #
    # 'omniglot_experiments/rbm_cd_10',
    # 'omniglot_experiments/rbm_pm1_cd_10',
    # 'omniglot_experiments/rbm_pm1_geo_10',

    '20Newsgroup_experiments/rbm_cd_1',
    '20Newsgroup_experiments/rbm_pm1_cd_1',
    '20Newsgroup_experiments/rbm_pm1_geo_1',

    '20Newsgroup_experiments/rbm_cd_5',
    '20Newsgroup_experiments/rbm_pm1_cd_5',
    '20Newsgroup_experiments/rbm_pm1_geo_5',

    '20Newsgroup_experiments/rbm_cd_10',
    '20Newsgroup_experiments/rbm_pm1_cd_10',
    '20Newsgroup_experiments/rbm_pm1_geo_10',
]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def create_rec():
    random.shuffle(model_dirs)
    for model_dir in model_dirs:
        dataset_dirs = os.listdir(model_dir)
        random.shuffle(dataset_dirs)
        for dataset_dir in dataset_dirs:
            exp_dirs = os.listdir(os.path.join(model_dir, dataset_dir))
            random.shuffle(exp_dirs)

            for exp_dir in tqdm(exp_dirs):
                for run_dir in os.listdir(os.path.join(model_dir, dataset_dir, exp_dir)):
                    print os.path.join(model_dir, dataset_dir, exp_dir, run_dir)

                    recs = []

                    caffe_files = os.listdir(os.path.join(model_dir, dataset_dir, exp_dir, run_dir))

                    if os.path.exists(os.path.join(model_dir, dataset_dir, exp_dir, run_dir, exp_dir + '.rec')):
                        continue

                    for caffe_file in caffe_files:
                        if not caffe_file.endswith('caffemodel'):
                            continue

                        net = caffe.Net(os.path.join(model_dir, dataset_dir, exp_dir, run_dir, exp_dir + '.prototxt'),
                                        os.path.join(model_dir, dataset_dir, exp_dir, run_dir, caffe_file),
                                        caffe.TEST)
                        w = net.params['rbm1'][0].data.astype(np.float64)
                        b = net.params['rbm1'][1].data.astype(np.float64)
                        c = net.params['rbm1'][2].data.astype(np.float64)

                        x = datasets[dataset_dir][2].astype(np.float64)
                        h = sigmoid(np.matmul(x, np.transpose(w)) + np.transpose(c))
                        x_ = sigmoid(np.matmul(h, w) + np.transpose(b))
                        x_ = np.clip(x_, eps, 1 - eps)

                        error = -np.mean(np.sum(x * np.log(x_) + (1.0 - x) * np.log((1.0 - x_)), axis=1), axis=0)

                        recs.append((caffe_file, error))

                    recs = sorted(recs, key=lambda k: k[1])

                    if len(recs) < 1000:
                        continue

                    pickle.dump(recs,
                                open(os.path.join(model_dir, dataset_dir, exp_dir, run_dir, exp_dir + '.rec'), 'wb'),
                                pickle.HIGHEST_PROTOCOL)


def knn_eval(train_features, train_labels, test_features, test_labels):
    n_obs = np.shape(test_features)[0]
    distances = cdist(test_features, train_features)
    k_nerest = np.argsort(distances, axis=1)
    prec = 0.0
    acc = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    knn_labels = np.zeros(shape=[n_obs, len(acc)], dtype=np.int32)

    for n in range(n_obs):
        class_values = train_labels[k_nerest[n, :]]
        y_true = test_labels[n] == class_values
        y_scores = np.arange(y_true.shape[0], 0, -1)
        ap = average_precision_score(y_true, y_scores)
        prec = prec + ap
        for k_neigh in range(len(acc)):
            a = class_values[0:(k_neigh + 1)]
            counts = np.bincount(a)
            b = np.where(counts == np.max(counts))[0]

            knn_labels[n][k_neigh] = b[0]

            if test_labels[n] in b:
                acc[k_neigh] = acc[k_neigh] + (1.0 / float(len(b)))

    prec /= float(np.shape(test_features)[0])
    acc = [x / float(np.shape(test_features)[0]) for x in acc]
    nmi = [normalized_mutual_info_score(test_labels, knn_labels[:, i]) for i in range(len(acc))]
    return prec, acc, nmi


def knn_classify():
    random.shuffle(model_dirs)
    for model_dir in model_dirs:
        dataset_dirs = os.listdir(model_dir)
        random.shuffle(dataset_dirs)
        for dataset_dir in dataset_dirs:
            exp_dirs = os.listdir(os.path.join(model_dir, dataset_dir))
            random.shuffle(exp_dirs)

            for exp_dir in exp_dirs:
                for run_dir in os.listdir(os.path.join(model_dir, dataset_dir, exp_dir)):
                    print os.path.join(model_dir, dataset_dir, exp_dir, run_dir)

                    if not os.path.exists(os.path.join(model_dir, dataset_dir, exp_dir, run_dir, exp_dir + '.rec')):
                        continue

                    if os.path.exists(os.path.join(model_dir, dataset_dir, exp_dir, run_dir, exp_dir + '.classify')):
                        continue

                    recs = pickle.load(
                        open(os.path.join(model_dir, dataset_dir, exp_dir, run_dir, exp_dir + '.rec'), 'rb'))
                    recs = recs[:5]

                    results = []

                    for rec in recs:

                        net = caffe.Net(os.path.join(model_dir, dataset_dir, exp_dir, run_dir, exp_dir + '.prototxt'),
                                        os.path.join(model_dir, dataset_dir, exp_dir, run_dir, rec[0]),
                                        caffe.TEST)
                        w = net.params['rbm1'][0].data
                        c = net.params['rbm1'][2].data

                        h_train = sigmoid(np.matmul(datasets[dataset_dir][0], np.transpose(w)) + np.transpose(c))
                        h_test = sigmoid(np.matmul(datasets[dataset_dir][4], np.transpose(w)) + np.transpose(c))

                        if datasets[dataset_dir][1].shape[1] > 1:
                            train_labels = np.argmax(datasets[dataset_dir][1], axis=1)
                        else:
                            train_labels = np.squeeze(datasets[dataset_dir][1])

                        if datasets[dataset_dir][5].shape[1] > 1:
                            test_labels = np.argmax(datasets[dataset_dir][5], axis=1)
                        else:
                            test_labels = np.squeeze(datasets[dataset_dir][5])

                        prec, acc, nmi = knn_eval(h_train, train_labels, h_test, test_labels)

                        results.append((rec[0], prec, acc, nmi))

                    pickle.dump(results,
                                open(os.path.join(model_dir, dataset_dir, exp_dir, run_dir, exp_dir + '.classify'),
                                     'wb'), pickle.HIGHEST_PROTOCOL)


def result_analysis():
    data = {}

    for model_dir in model_dirs:
        for dataset_dir in os.listdir(model_dir):
            exp_dirs = os.listdir(os.path.join(model_dir, dataset_dir))
            random.shuffle(exp_dirs)

            for exp_dir in exp_dirs:
                partial_results = []

                steps = exp_dir.split('_')[3]
                exp_name = dataset_dir + '_' + model_dir + '_' + str(steps)

                for run_dir in os.listdir(os.path.join(model_dir, dataset_dir, exp_dir)):
                    if not os.path.exists(
                            os.path.join(model_dir, dataset_dir, exp_dir, run_dir, exp_dir + '.classify')):
                        continue

                    print os.path.join(model_dir, exp_dir)

                    results = pickle.load(
                        open(os.path.join(model_dir, dataset_dir, exp_dir, run_dir, exp_dir + '.classify'), 'rb'))

                    this_run_results = []
                    for name, prec, acc, nmi in results:
                        print('{} {} {} {}'.format(name, prec, np.max(acc), np.max(nmi)))

                        this_run_results.append((prec, np.max(acc), np.max(nmi)))

                    partial_results.append([np.max(i) for i in zip(*this_run_results)])

                if len(partial_results) > 0:
                    if exp_name not in data:
                        data[exp_name] = []
                    t_append = [np.mean(r) for r in zip(*partial_results)], [np.std(r) for r in zip(*partial_results)]
                    data[exp_name].append((t_append, os.path.join(model_dir, dataset_dir, exp_dir)))
                    pass

    good_setups = set()

    with open('classify.csv', 'wt') as csv_file:
        writer = csv.writer(csv_file)

        for alg in ALGS:
            if alg in data:
                p = [(l[0][0][0], l[0][1][0], l[1]) for l in data[alg]]
                a = [(l[0][0][1], l[0][1][1], l[1]) for l in data[alg]]
                n = [(l[0][0][2], l[0][1][2], l[1]) for l in data[alg]]

                p_id = np.argmax([l[0] for l in p])
                p_mean, p_std, p_name = p[p_id]

                a_id = np.argmax([l[0] for l in a])
                a_mean, a_std, a_name = a[a_id]

                n_id = np.argmax([l[0] for l in n])
                n_mean, n_std, n_name = n[n_id]

                print('{} {} {} {} {} {} {}'.format(alg, p_mean, p_std, a_mean, a_std, n_mean, n_std))

                writer.writerow([alg, p_mean, a_mean, n_mean])

                good_setups.add(p_name)
                good_setups.add(a_name)
                good_setups.add(n_name)

                copy_figures(p_name)
                copy_figures(a_name)
                copy_figures(n_name)


                pass
            else:
                print('{}'.format(alg))

    for setup in good_setups:
        print(setup)


def copy_figures(name):
    name_list = name.split('/')

    dst_dir = '/'.join(name_list[1:3])
    if not os.path.exists(os.path.join('figures', dst_dir)):
        os.makedirs(os.path.join('figures', dst_dir))
    for file_name in os.listdir(os.path.join(name, '1')):
        if 'test' in file_name and '0' in file_name and file_name.endswith('pdf'):
            shutil.copy(os.path.join(name, '1', file_name), os.path.join('figures', dst_dir, name_list[3]+'_'+file_name))


def move_data_subdirs():
    for model_dir in model_dirs:
        for dataset_dir in os.listdir(model_dir):
            for exp_dir in os.listdir(os.path.join(model_dir, dataset_dir)):
                caffe_files = os.listdir(os.path.join(model_dir, dataset_dir, exp_dir))

                os.mkdir(os.path.join(model_dir, dataset_dir, exp_dir, '1'))

                for caffe_file in caffe_files:
                    os.rename(os.path.join(model_dir, dataset_dir, exp_dir, caffe_file),
                              os.path.join(model_dir, dataset_dir, exp_dir, '1', caffe_file))
                    pass


def extract_pca():
    random.shuffle(model_dirs)
    for model_dir in model_dirs:
        dataset_dirs = os.listdir(model_dir)
        random.shuffle(dataset_dirs)
        for dataset_dir in dataset_dirs:

            if len(dataset_filter) > 0 and dataset_dir not in dataset_filter:
                continue

            exp_dirs = os.listdir(os.path.join(model_dir, dataset_dir))
            random.shuffle(exp_dirs)

            for exp_dir in exp_dirs:
                for run_dir in os.listdir(os.path.join(model_dir, dataset_dir, exp_dir)):
                    print os.path.join(model_dir, dataset_dir, exp_dir, run_dir)

                    if not os.path.exists(os.path.join(model_dir, dataset_dir, exp_dir, run_dir, exp_dir + '.rec')):
                        continue

                    recs = pickle.load(
                        open(os.path.join(model_dir, dataset_dir, exp_dir, run_dir, exp_dir + '.rec'), 'rb'))
                    recs = recs[:5]

                    for num, rec in enumerate(recs):

                        if num > 0:
                            continue

                        if os.path.exists(
                                os.path.join(model_dir, dataset_dir, exp_dir, run_dir, str(num) + '_train.pdf')) and \
                                os.path.exists(
                                    os.path.join(model_dir, dataset_dir, exp_dir, run_dir, str(num) + '_test.pdf')):
                            continue

                        net = caffe.Net(os.path.join(model_dir, dataset_dir, exp_dir, run_dir, exp_dir + '.prototxt'),
                                        os.path.join(model_dir, dataset_dir, exp_dir, run_dir, rec[0]),
                                        caffe.TEST)
                        w = net.params['rbm1'][0].data
                        c = net.params['rbm1'][2].data

                        h_train = sigmoid(np.matmul(datasets[dataset_dir][0], np.transpose(w)) + np.transpose(c))
                        h_test = sigmoid(np.matmul(datasets[dataset_dir][4], np.transpose(w)) + np.transpose(c))

                        y_train = datasets[dataset_dir][1]
                        if y_train.shape[1] != 1:
                            y_train = np.argmax(y_train, axis=1)

                        y_test = datasets[dataset_dir][5]
                        if y_test.shape[1] != 1:
                            y_test = np.argmax(y_test, axis=1)

                        pca = PCA(n_components=2)
                        pca.fit(h_train)

                        h_train_2 = pca.transform(h_train)
                        h_test_2 = pca.transform(h_test)

                        fit, ax = plt.subplots()
                        ax.margins(0.05)

                        for label in range(y_train.max()+1):
                            label_h = h_train_2[(y_train == label).flatten(), :]

                            ax.plot([a[0] for a in label_h], [a[1] for a in label_h], marker='o', linestyle='', ms=1,
                                    label=str(label))

                        plt.legend(('0', '1', '2', '3', '4', '5', '6', '7', '8', '9'), fontsize=8)

                        plt.savefig(os.path.join(model_dir, dataset_dir, exp_dir, run_dir, str(num) + '_train.pdf'))
                        plt.close()

                        fit, ax = plt.subplots()
                        ax.margins(0.05)

                        for label in range(y_test.max()+1):
                            label_h = h_test_2[(y_test == label).flatten(), :]

                            ax.plot([a[0] for a in label_h], [a[1] for a in label_h], marker='o', linestyle='', ms=1,
                                    label=str(label))

                        plt.legend(('0', '1', '2', '3', '4', '5', '6', '7', '8', '9'), fontsize=8)

                        plt.savefig(os.path.join(model_dir, dataset_dir, exp_dir, run_dir, str(num) + '_test.pdf'))
                        plt.close()


def extract_tsne():
    random.shuffle(model_dirs)
    for model_dir in model_dirs:
        dataset_dirs = os.listdir(model_dir)
        random.shuffle(dataset_dirs)
        for dataset_dir in dataset_dirs:

            if len(dataset_filter) > 0 and dataset_dir not in dataset_filter:
                continue

            exp_dirs = os.listdir(os.path.join(model_dir, dataset_dir))
            random.shuffle(exp_dirs)

            for exp_dir in tqdm(exp_dirs):

                steps = exp_dir.split('_')[3]
                exp_name = dataset_dir + '_' + model_dir + '_' + str(steps)

                for run_dir in os.listdir(os.path.join(model_dir, dataset_dir, exp_dir)):
                    if run_dir != '1':
                        continue

                    print os.path.join(model_dir, dataset_dir, exp_dir, run_dir)

                    if not os.path.exists(os.path.join(model_dir, dataset_dir, exp_dir, run_dir, exp_dir + '.rec')):
                        continue

                    recs = pickle.load(
                        open(os.path.join(model_dir, dataset_dir, exp_dir, run_dir, exp_dir + '.rec'), 'rb'))
                    recs = recs[:5]

                    for num, rec in enumerate(recs):
                        if num > 0:
                            continue

                        if os.path.exists(os.path.join(model_dir, dataset_dir,
                                                       exp_dir, run_dir,
                                                       str(num) + '_tsne_test.pdf')):
                            continue

                        if len(save_filter_tsne) > 0 and os.path.join(
                                model_dir, dataset_dir, exp_dir) not in save_filter_tsne:
                            continue

                        net = caffe.Net(os.path.join(model_dir, dataset_dir,
                                                     exp_dir, run_dir, exp_dir + '.prototxt'),
                                        os.path.join(model_dir, dataset_dir,
                                                     exp_dir, run_dir, rec[0]),
                                        caffe.TEST)
                        w = net.params['rbm1'][0].data.astype(np.float64)
                        c = net.params['rbm1'][2].data.astype(np.float64)

                        h_test = sigmoid(np.matmul(
                            datasets[dataset_dir][4].astype(np.float64),
                            np.transpose(w)) + np.transpose(c))

                        y_test = datasets[dataset_dir][5]
                        if y_test.shape[1] != 1:
                            y_test = np.argmax(y_test, axis=1)

                        h_test_2 = tsne(h_test, 2, 50, 20.0)

                        fit, ax = plt.subplots()
                        ax.margins(0.05)

                        for label in range(y_test.max()+1):
                            label_h = h_test_2[(y_test == label).flatten(), :]

                            ax.plot([a[0] for a in label_h], [a[1] for a in label_h], marker='o', linestyle='', ms=1,
                                    label=str(label))

                        plt.savefig(os.path.join(model_dir, dataset_dir, exp_dir, run_dir, str(num) + '_tsne_test.pdf'))
                        plt.close()


def count_all():
    counts = {
        'rec': 0,
        'big_ais': 0,
        'small_ais': 0,
        'classify': 0,
        'pdf': 0,
        'prototxt': 0,
    }

    for model_dir in model_dirs:

        dataset_dirs = os.listdir(model_dir)
        for dataset_dir in dataset_dirs:

            exp_dirs = os.listdir(os.path.join(model_dir, dataset_dir))
            for exp_dir in exp_dirs:

                run_dirs = os.listdir(os.path.join(model_dir, dataset_dir, exp_dir))
                for run_dir in run_dirs:

                    caffe_files = os.listdir(os.path.join(model_dir, dataset_dir, exp_dir, run_dir))
                    for caffe_file in caffe_files:

                        for extension in counts.keys():
                            if caffe_file.endswith(extension):
                                counts[extension] += 1

    counts['rec'] /= 1.0
    counts['classify'] /= 1.0
    counts['pdf'] /= 10.0
    counts['prototxt'] /= 2.0
    counts['big_ais'] /= 1.0
    counts['small_ais'] /= 1.0

    print counts


def check_singles():
    for model_dir in model_dirs:

        dataset_dirs = os.listdir(model_dir)
        for dataset_dir in dataset_dirs:

            exp_dirs = os.listdir(os.path.join(model_dir, dataset_dir))
            for exp_dir in exp_dirs:

                run_dirs = os.listdir(os.path.join(model_dir, dataset_dir, exp_dir))

                if len(run_dirs) < 3:
                    print dataset_dir, exp_dir, len(run_dirs)


def create_ais_val(big_betas=True):

    if big_betas:
        extension = '.big_ais'
    else:
        extension = '.small_ais'

    random.shuffle(model_dirs)
    for model_dir in model_dirs:
        dataset_dirs = os.listdir(model_dir)
        random.shuffle(dataset_dirs)
        for dataset_dir in dataset_dirs:

            if 'mnist_binary' in dataset_dir:
                continue

            exp_dirs = os.listdir(os.path.join(model_dir, dataset_dir))
            random.shuffle(exp_dirs)

            for exp_dir in tqdm(exp_dirs):
                for run_dir in os.listdir(os.path.join(model_dir, dataset_dir, exp_dir)):

                    print('')
                    print(os.path.join(model_dir, dataset_dir, exp_dir, run_dir))

                    aises = []

                    caffe_files = filter(lambda x: x.endswith('caffemodel'),
                                         os.listdir(os.path.join(model_dir, dataset_dir, exp_dir, run_dir)))
                    caffe_files = sorted(caffe_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))

                    if os.path.exists(os.path.join(model_dir, dataset_dir, exp_dir, run_dir, exp_dir + extension)):
                        continue

                    best_ais = -np.inf
                    best_without_change = 30

                    for file_num, caffe_file in enumerate(caffe_files):
                        if not caffe_file.endswith('caffemodel'):
                            continue

                        net = caffe.Net(os.path.join(model_dir, dataset_dir, exp_dir, run_dir, exp_dir + '.prototxt'),
                                        os.path.join(model_dir, dataset_dir, exp_dir, run_dir, caffe_file),
                                        caffe.TEST)
                        w = net.params['rbm1'][0].data.astype(np.float64)
                        b = net.params['rbm1'][1].data.astype(np.float64)
                        c = net.params['rbm1'][2].data.astype(np.float64)

                        x_train = datasets[dataset_dir][0].astype(np.float64)
                        x_val = datasets[dataset_dir][2].astype(np.float64)
                        x_test = datasets[dataset_dir][4].astype(np.float64)

                        ais_train, ais_val, ais_test = estimate_log_prob(
                            np.transpose(b), np.transpose(c), np.transpose(w),
                            np.transpose(x_train),
                            np.transpose(x_val),
                            np.transpose(x_test),
                            big_betas)

                        if ais_val > best_ais:
                            best_ais = ais_val
                            best_without_change = 30
                        else:
                            best_without_change -= 1

                        aises.append((
                            caffe_file,
                            (ais_train, ais_val, ais_test),
                            best_without_change))

                        if best_without_change == 0:
                            break

                        print(caffe_file, [ais_train, ais_val, ais_test],
                              file_num, best_without_change)

                    with open(os.path.join(model_dir, dataset_dir, exp_dir, run_dir, exp_dir + extension), 'wb') as file_:
                        pickle.dump(aises, file_, pickle.HIGHEST_PROTOCOL)


def create_ais_test():
    extension = '.ais'

    random.shuffle(model_dirs)
    for model_dir in model_dirs:
        dataset_dirs = os.listdir(model_dir)
        random.shuffle(dataset_dirs)
        for dataset_dir in dataset_dirs:

            if 'mnist_binary' in dataset_dir:
                continue

            exp_dirs = os.listdir(os.path.join(model_dir, dataset_dir))
            random.shuffle(exp_dirs)

            for exp_dir in tqdm(exp_dirs):
                for run_dir in os.listdir(os.path.join(model_dir, dataset_dir, exp_dir)):

                    print('')
                    print(os.path.join(model_dir, dataset_dir, exp_dir, run_dir))

                    if os.path.exists(os.path.join(model_dir, dataset_dir, exp_dir, run_dir, exp_dir + extension)):
                        continue

                    small_ais_path = os.path.join(model_dir, dataset_dir, exp_dir, run_dir, exp_dir + '.small_ais')

                    if not os.path.exists(small_ais_path):
                        continue

                    with open(small_ais_path, 'rb') as file_:
                        small_ais = pickle.load(file_)

                    model_to_check = next(x[0] for x in reversed(small_ais) if x[2] == 30)


                    net = caffe.Net(os.path.join(model_dir, dataset_dir, exp_dir, run_dir, exp_dir + '.prototxt'),
                                    os.path.join(model_dir, dataset_dir, exp_dir, run_dir, model_to_check),
                                    caffe.TEST)

                    w = net.params['rbm1'][0].data.astype(np.float64)
                    b = net.params['rbm1'][1].data.astype(np.float64)
                    c = net.params['rbm1'][2].data.astype(np.float64)

                    x_train = datasets[dataset_dir][0].astype(np.float64)
                    x_val = datasets[dataset_dir][2].astype(np.float64)
                    x_test = datasets[dataset_dir][4].astype(np.float64)

                    ais_train, ais_val, ais_test = estimate_log_prob(
                        np.transpose(b), np.transpose(c), np.transpose(w),
                        np.transpose(x_train),
                        np.transpose(x_val),
                        np.transpose(x_test),
                        True)

                    with open(os.path.join(model_dir, dataset_dir, exp_dir, run_dir, exp_dir + extension), 'wb') as file_:
                        pickle.dump((ais_train, ais_val, ais_test), file_, pickle.HIGHEST_PROTOCOL)


def ais_report():
    total_results = dict()

    random.shuffle(model_dirs)
    for model_dir in model_dirs:
        dataset_dirs = os.listdir(model_dir)
        random.shuffle(dataset_dirs)
        for dataset_dir in dataset_dirs:

            if 'mnist_binary' in dataset_dir:
                continue

            exp_dirs = os.listdir(os.path.join(model_dir, dataset_dir))
            random.shuffle(exp_dirs)

            for exp_dir in tqdm(exp_dirs):

                run_results = list()

                for run_dir in os.listdir(os.path.join(model_dir, dataset_dir, exp_dir)):
                    ais_path = os.path.join(model_dir, dataset_dir, exp_dir, run_dir, exp_dir + '.ais')

                    if not os.path.exists(ais_path):
                        continue

                    with open(ais_path, 'rb') as file_:
                        ais = pickle.load(file_)

                    run_results.append(ais)

                steps = exp_dir.split('_')[3]
                exp_name = dataset_dir + '_' + model_dir + '_' + str(steps)
                run_results = np.array(run_results)
                run_results = (np.mean(run_results, axis=0), np.std(run_results, axis=0))

                if exp_name not in total_results:
                    total_results[exp_name] = run_results
                elif total_results[exp_name][0][1] < run_results[0][1]:
                    total_results[exp_name] = run_results

    pprint.pprint(total_results)

    with open('ais.csv', 'wt') as csv_file:
        writer = csv.writer(csv_file)

        for result_name in total_results.keys():
            means = total_results[result_name][0].tolist()
            stds = total_results[result_name][1].tolist()
            writer.writerow([result_name, means[0], stds[0], means[1], stds[1], means[2], stds[2]])
            pass



if __name__ == '__main__':


    # diagnostics
    # count_all()

    # data movement
    # check_singles()
    # move_data_subdirs()

    # actual processing
    # create_rec()
    # create_ais_val(False)
    # create_ais_test()
    # knn_classify()

    # extract_tsne()
    # extract_pca()

    result_analysis()
    # ais_report()

    pass
