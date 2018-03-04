import numpy as np
from scipy.misc import logsumexp


def _calculate_log_prob(vishid, hidbiases, visbiases, logZ, testX):
    numcases = testX.shape[0]
    pd = np.matmul(testX, np.transpose(visbiases)) + \
         np.sum(
             np.log(1.0+np.exp(
                 np.matmul(np.ones(shape=(numcases, 1)), hidbiases) +
                 np.matmul(testX, vishid))),
             1, keepdims=True)
    logprob = np.sum(pd) / numcases - logZ
    return logprob


def _base_rate(imgTrain, num_runs):
    eps = 0.00001
    # 784, 50000 - 60000
    [numdims, numcases] = imgTrain.shape

    p_int = np.mean(imgTrain, 1, keepdims=True)
    p_int[p_int == 0] = eps
    p_int[p_int == 1] = 1.0 - eps

    log_base_rate = np.log(p_int) - np.log(1.0 - p_int)

    return np.transpose(log_base_rate)


def _estimate_log_z(vishid, hidbiases, visbiases, numruns, beta, imgTrain=None):
    """
    :param vishid: a matrix of RBM weights[numvis, numhid]
    :param hidbiases: a row vector of hidden  biases[1 numhid]
    :param visbiases: a row vector of visible biases[1 numvis]
    :param numruns: number of AIS runs
    :param beta: a row vector containing beta's
    :param imgTrain: the data that is divided into batches (numcases numdims numbatches)
    :return:
    """

    [numdims, numhids] = vishid.shape

    if imgTrain is not None:
        visbiases_base = _base_rate(imgTrain, numruns)
    else:
        visbiases_base = 0.0 * visbiases


    numcases = numruns

    visbias_base = np.tile(visbiases_base, (numcases, 1))
    hidbias = np.tile(hidbiases, (numcases, 1))
    visbias = np.tile(visbiases, (numcases, 1))

    logww = np.zeros(shape=(numcases, 1))
    negdata = np.tile(1.0 / (1.0 + np.exp(-visbiases_base)), (numcases, 1))
    negdata = negdata > np.random.rand(numcases, numdims)
    # negdata = negdata > 0.5

    logww = logww - (np.matmul(negdata, np.transpose(visbiases_base)) + numhids * np.log(2.0))

    Wh = np.matmul(negdata, vishid) + hidbias
    Bv_base = np.matmul(negdata, np.transpose(visbiases_base))
    Bv = np.matmul(negdata, np.transpose(visbiases))

    for bb in beta[1:]:
        expWh = np.exp(bb * Wh)
        logww = logww + (1.0 - bb) * Bv_base + bb * Bv + np.sum(np.log(1.0 + expWh), 1, keepdims=True)

        poshidprobs = np.divide(expWh, (1.0 + expWh))
        poshidstates = poshidprobs > np.random.rand(numcases, numhids)
        # poshidstates = poshidprobs > 0.5

        negdata = np.divide(
            1.0,
            1.0 + np.exp(
                - (1.0 - bb) * visbias_base
                - bb * (np.matmul(poshidstates, np.transpose(vishid)) + visbias)))
        negdata = negdata > np.random.rand(numcases, numdims)
        # negdata = negdata > 0.5

        Wh = np.matmul(negdata, vishid) + hidbias
        Bv_base = np.matmul(negdata, np.transpose(visbiases_base))
        Bv = np.matmul(negdata, np.transpose(visbiases))

        expWh = np.exp(bb * Wh)
        logww = logww - ((1.0 - bb) * Bv_base + bb * Bv + np.sum(np.log(1.0 + expWh), 1, keepdims=True))

    expWh = np.exp(Wh)
    logww = logww + np.matmul(negdata, np.transpose(visbiases)) + np.sum(np.log(1.0+expWh), 1, keepdims=True)

    r_AIS = logsumexp(logww.flatten()) - np.log(numcases)

    logZZ_base = np.sum(np.log(1.0 + np.exp(visbiases_base))) + numhids * np.log(2.0)
    logZZ_est = r_AIS + logZZ_base

    return logZZ_est


def estimate_log_prob(b, c, w, x_train, x_val, x_test, big_betas):
    # from scipy.io import savemat
    # m_dict = {'b': b, 'c': c, 'w': w, 'x_train': x_train, 'x_val': x_val, 'x_test': x_test}
    # savemat('test_data.mat', m_dict)

    num_runs = 100

    if big_betas:
        beta = np.concatenate(
            [
                np.arange(0.0, 0.5, 1.0/1000.0, np.float64),
                np.arange(0.5, 0.9, 1.0/10000.0, np.float64),
                np.arange(0.9, 1.0, 1.0/100000.0, np.float64),
            ], axis=0)
    else:
        beta = np.concatenate(
            [
                np.arange(0.0, 0.5, 1.0/100.0, np.float64),
                np.arange(0.5, 0.9, 1.0/500.0, np.float64),
                np.arange(0.9, 1.0, 1.0/1000.0, np.float64),
            ], axis=0)

    log_z = _estimate_log_z(w, c, b, num_runs, beta, x_train)
    log_prob_train = _calculate_log_prob(w, c, b, log_z, np.transpose(x_train))
    log_prob_val = _calculate_log_prob(w, c, b, log_z, np.transpose(x_val))
    log_prob_test = _calculate_log_prob(w, c, b, log_z, np.transpose(x_test))
    return log_prob_train, log_prob_val, log_prob_test


