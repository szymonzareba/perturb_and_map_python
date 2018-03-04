import numpy as np
import math
import os


def init_rbm(m, d):
    w = np.random.uniform(low=-1.0 / math.sqrt(float(max(m, d))),
                          high=1.0 / math.sqrt(float(max(m, d))),
                          size=(d, m))

    b = np.random.normal(loc=0, scale=1, size=(d, 1))

    c = np.zeros(shape=(m,1)) - 1.0

    return b, c, w


def save_caffe(m, d, dataset):
    b, c, w = init_rbm(m, d)

    with open(os.path.join('inits', dataset, str(m), 'b.txt'), 'w') as f:
        for bb in b.flatten().tolist():
            f.write('{} '.format(bb))

    with open(os.path.join('inits', dataset, str(m), 'c.txt'), 'w') as f:
        for bb in c.flatten().tolist():
            f.write('{} '.format(bb))

    with open(os.path.join('inits', dataset, str(m), 'w.txt'), 'w') as f:
        for bb in w.flatten().tolist():
            f.write('{} '.format(bb))

save_caffe(50, 100, '20Newsgroup50')
