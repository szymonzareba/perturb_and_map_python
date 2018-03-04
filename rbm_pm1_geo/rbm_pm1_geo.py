import os
import caffe
from experiment_factory import create_protos
from itertools import product
from tqdm import tqdm


def run(dataset, gpu, learning_rates, momentums, weight_decays, steps, num_hiddens, runs):
    DEVICE = 'GPU'
    EXP_PATH = '/home/szymon.zareba/dev/phd_pycaffe/experiments/rbm_pm1_geo/'

    if DEVICE == 'CPU':
        caffe.set_mode_cpu()
    elif DEVICE == 'GPU':
        caffe.set_device(gpu)
        caffe.set_mode_gpu()
    else:
        raise ValueError(DEVICE)

    EXP_PATH = os.path.join(EXP_PATH, dataset)

    all_params = [learning_rates, momentums, weight_decays, steps, num_hiddens]

    for setup in tqdm(list(product(*all_params))):
        for run in range(1, runs + 1):
            run = str(run)
            print 'setup: {}'.format(setup)
            learning_rate, momentum, weight_decay, step, num_hidden = setup

            name = '_'.join([str(s) for s in setup])
            print 'name: pm_geo_{}'.format(name)

            if os.path.exists(os.path.join(EXP_PATH, name, run)):
                print 'experiment exists'
            else:
                model_proto, solver_proto = create_protos(EXP_PATH, name, DEVICE, learning_rate, momentum, weight_decay, step, num_hidden, dataset, run)
                solver = caffe.get_solver(solver_proto)
                solver.solve()
