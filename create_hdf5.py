import os
from scipy.io import loadmat
import h5py

# path = 'datasets/20Newsgroup/'
path = 'datasets/mnist_binary/'

test_x = loadmat(os.path.join(path, 'testX.mat'))['testX']
test_y = loadmat(os.path.join(path, 'testY.mat'))['testY']

training_x = loadmat(os.path.join(path, 'trainingX.mat'))['trainingX']
training_y = loadmat(os.path.join(path, 'trainingY.mat'))['trainingY']

validation_x = loadmat(os.path.join(path, 'validationX.mat'))['validationX']
validation_y = loadmat(os.path.join(path, 'validationY.mat'))['validationY']

if os.path.exists(os.path.join(path, 'test.h5')):
    os.remove(os.path.join(path, 'test.h5'))

if os.path.exists(os.path.join(path, 'train.h5')):
    os.remove(os.path.join(path, 'train.h5'))

if os.path.exists(os.path.join(path, 'validation.h5')):
    os.remove(os.path.join(path, 'validation.h5'))

f = h5py.File(os.path.join(path, 'train.h5'), 'w')
f.create_dataset('data', data=training_x, dtype='f8')
f.create_dataset('label', data=training_y.T, dtype='f8')
f.close()

f = h5py.File(os.path.join(path, 'validation.h5'), 'w')
f.create_dataset('data', data=validation_x, dtype='f8')
f.create_dataset('label', data=validation_y.T, dtype='f8')
f.close()

f = h5py.File(os.path.join(path, 'test.h5'), 'w')
f.create_dataset('data', data=test_x, dtype='f8')
f.create_dataset('label', data=test_y.T, dtype='f8')
f.close()
