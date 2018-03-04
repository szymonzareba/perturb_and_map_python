def get_snapshots(lines, dataset):
    if dataset == 'mnist_binary':
        lines.append('max_iter: 500000')
        lines.append('snapshot: 500')
        lines.append('display: 100')
    elif dataset == 'omniglot':
        lines.append('max_iter: 163000')
        lines.append('snapshot: 163')
        lines.append('display: 100')
    elif dataset == '20Newsgroup':
        lines.append('max_iter: 85000')
        lines.append('snapshot: 85')
        lines.append('display: 8500')
    elif dataset == 'toy0.1':
        lines.append('max_iter: 500000')
        lines.append('snapshot: 1000')
        lines.append('display: 1000')
    elif dataset == 'toy0.01':
        lines.append('max_iter: 500000')
        lines.append('snapshot: 1000')
        lines.append('display: 1000')
    else:
        raise ValueError(dataset)


def get_batch_size(dataset):
    if dataset == 'mnist_binary':
        return 100
    elif dataset == 'omniglot':
        return 100
    elif dataset == '20Newsgroup':
        return 100
    elif dataset == '20Newsgroup50':
        return 100
    elif dataset == 'toy0.1':
        return 10
    elif dataset == 'toy0.01':
        return 10
    else:
        raise ValueError(dataset)
