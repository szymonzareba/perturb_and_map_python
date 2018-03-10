from .rbm_cd.rbm_cd import run as run_cd
from .rbm_pm1_cd.rbm_pm1_cd import run as run_pm_cd
from .rbm_pm1_geo.rbm_pm1_geo import run as run_pm_geo

# dataset = 'mnist_binary'
# gpu = 0
# learning_rates = [1e-2]
# momentums = [0.9]
# weight_decays = [0]
# steps = [1]
# num_hiddens = [500]
# runs = 3

dataset = '20Newsgroup'
gpu = 0
learning_rates = [1e-1, 1e-2, 1e-3]
momentums = [0, 0.9]
weight_decays = [0, 1e-3, 1e-4, 1e-5]
steps = [1, 5, 10]
num_hiddens = [50]
runs = 3

# dataset = 'toy0.1'
# dataset = 'toy0.01'
# gpu = 1
# learning_rates = [1e-1, 1e-2, 1e-3]
# momentums = [0, 0.9]
# weight_decays = [0, 1e-3, 1e-4, 1e-5]
# steps = [1, 5, 10]
# num_hiddens = [10]
# runs = 5

# run_cd(dataset, gpu, learning_rates, momentums, weight_decays, steps, num_hiddens, runs)
# run_pm_cd(dataset, gpu, learning_rates, momentums, weight_decays, steps, num_hiddens, runs)
run_pm_geo(dataset, gpu, learning_rates, momentums, weight_decays, steps, num_hiddens, runs)
