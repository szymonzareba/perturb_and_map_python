
import os

from ..factory_commons import get_snapshots, get_batch_size


def get_model_proto(cd_steps, num_hidden, dataset):
    lines = []
    lines.append('name: "{}_rbm_cd"'.format(dataset))

    lines.append('layer')
    lines.append('{')
    lines.append('name: "data"')
    lines.append('type: "HDF5Data"')
    lines.append('top: "data"')
    lines.append('hdf5_data_param {')
    lines.append('source: "/home/szymon.zareba/dev/phd_pycaffe/datasets/{}/train.txt"'.format(dataset))
    lines.append('batch_size: {}'.format(get_batch_size(dataset)))
    lines.append('shuffle: true')
    lines.append('}')
    lines.append('}')

    lines.append('layer')
    lines.append('{')
    lines.append('name: "bin1"')
    lines.append('type: "Binarization"')
    lines.append('bottom: "data"')
    lines.append('top: "bin"')
    lines.append('bin_param{')
    lines.append('work: true')
    lines.append('}')
    lines.append('}')

    lines.append('layer')
    lines.append('{')
    lines.append('name: "rbm1"')
    lines.append('type: "RBMPM1"')
    lines.append('bottom: "bin"')
    lines.append('top: "rbm1HOut"')
    lines.append('top: "rbm1HIn"')
    lines.append('top: "rbm1loss"')
    lines.append('loss_weight: 0')
    lines.append('loss_weight: 0')
    lines.append('loss_weight: 1')
    lines.append('param {')
    lines.append('lr_mult: 1')
    lines.append('decay_mult: 0')
    lines.append('}')
    lines.append('param {')
    lines.append('lr_mult: 1')
    lines.append('decay_mult: 0')
    lines.append('}')
    lines.append('param {')
    lines.append('lr_mult: 1')
    lines.append('decay_mult: 0')
    lines.append('}')
    lines.append('rbm_param{')
    lines.append('num_hidden: {}'.format(num_hidden))
    lines.append('llAprox: REC')
    lines.append('w_filler {')
    lines.append('type: "file"')
    lines.append('path: "/home/szymon.zareba/dev/phd_pycaffe/inits/{}/{}/w.txt"'.format(dataset, num_hidden))
    lines.append('}')
    lines.append('b_filler {')
    lines.append('type: "file"')
    lines.append('path: "/home/szymon.zareba/dev/phd_pycaffe/inits/{}/{}/b.txt"'.format(dataset, num_hidden))
    lines.append('}')
    lines.append('c_filler {')
    lines.append('type: "file"')
    lines.append('path: "/home/szymon.zareba/dev/phd_pycaffe/inits/{}/{}/c.txt"'.format(dataset, num_hidden))
    lines.append('}')

    lines.append('rbm_pm_param')
    lines.append('{')
    lines.append('map_method : GreedyEnergyOptimization')
    lines.append('batch_repeats: 1')
    lines.append('geo_param {')
    lines.append('steps: {}'.format(cd_steps))
    lines.append('}')
    lines.append('}')

    lines.append('}')
    lines.append('}')

    lines.append('layer')
    lines.append('{')
    lines.append('name: "rbmreturn"')
    lines.append('type: "RBMReturn"')
    lines.append('bottom: "rbm1HOut"')
    lines.append('bottom: "rbm1HIn"')
    lines.append('}')

    return lines


def get_solver_proto(exp_path, name, device, learning_rate, momentum, weight_decay, dataset, run):
    lines = []
    lines.append('net: "{}.prototxt"'.format(os.path.join(exp_path, name, run, name)))
    lines.append('base_lr:{}'.format(learning_rate))
    lines.append('lr_policy: "fixed"')

    get_snapshots(lines, dataset)

    lines.append('weight_decay:{}'.format(weight_decay))
    lines.append('snapshot_prefix: "{}"'.format(os.path.join(exp_path, name, run, name)))
    lines.append('solver_type: SGD')
    lines.append('momentum:{}'.format(momentum))
    lines.append('solver_mode: {}'.format(device))
    return lines


def create_protos(exp_path, name, device, learning_rate, momentum, weight_decay, cd_steps, num_hidden, dataset, run):
    os.makedirs(os.path.join(exp_path, name, run))

    with open(os.path.join(exp_path, name, run, name+ '.prototxt'), 'w') as f:
        f.writelines([line+'\n' for line in get_model_proto(cd_steps, num_hidden, dataset)])

    with open(os.path.join(exp_path, name, run, name+ '_solver.prototxt'), 'w') as f:
        f.writelines([line+'\n' for line in get_solver_proto(exp_path, name, device, learning_rate, momentum, weight_decay, dataset, run)])

    return os.path.join(exp_path, name, run, name + '.prototxt'), os.path.join(exp_path, name, run, name + '_solver.prototxt')