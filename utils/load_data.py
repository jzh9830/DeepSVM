import numpy as np
import scipy.io as scio
import torch
from torch.utils.data import Dataset
from torchvision.datasets import MNIST, CIFAR10
from torchvision import transforms
from PIL import Image

YALE = 'Yale'
UMIST = 'UMIST'

def load_data(name):
    path = '../data/{}.mat'.format(name)
    data = scio.loadmat(path)
    labels = data['Y']
    labels = np.reshape(labels, (labels.shape[0],))
    labels = process_y(labels, num_classes=max(labels)-min(labels)+1)
    X = process_x(np.transpose(data['X']))
    return X, labels

def process_y(Labels, num_classes):
    '''
    function: translate the (numbers, 1) into (types, numbers)
    :param Labels: the data label
    :return:
    Y: the data label (types, numbers)
    '''
    c = num_classes
    n = len(Labels)
    if np.min(Labels) > 0:
        Labels = Labels - np.min(Labels)
    Y = np.ones(shape=[c, n]) * -1
    for i in range(n):
        Y[Labels[i], i] = 1
    return Y

def process_x(Data):
    '''
    function: normalize the data
    :param Data: input Data
    :return:
    X: normalized data
    '''
    Min = np.min(Data)
    Max = np.max(Data)
    X = (Data - Min) / (Max - Min)
    return X

def half_posi_def(M):
    '''
    function: test the Matrix is half-positive definite or not
    :param M: input Matrix
    :return:
    1: yes or 0: no
    '''
    B = np.linalg.eigvals(M)
    if np.all(B >= 0):
        return 1
    else:
        return 0

def one_hot(x, class_count):
    '''
    :param x:
    :param class_count:
    :return:
    '''
    return torch.eye(class_count)[x, :]

class MatDataset(Dataset):
    def __init__(self, filename, transform):
        self.x, self.y = self.load_data(filename)
        if filename == './data/COIL20.mat':
            self.y = self.y - 1
        elif filename == './data/USPS.mat':
            self.y = self.y - 1
        elif filename == './data/Yale.mat':
            self.y = self.y - 1
        elif filename == './data/Palm.mat':
            self.y = self.y - 1
        self.transform = transform
        self.y = (self.y).astype('int64')

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x = np.array(self.x[idx,:,:]).astype(np.uint8)
        x = Image.fromarray(x)
        y = torch.from_numpy(np.array(self.y[idx]))
        if self.transform:
            x = self.transform(x)
        return x, y


    def load_data(self, filename):
        data = scio.loadmat(filename)
        X = data['X']  # n * d
        Y = data['Y'].reshape(-1)  # n * 1
        idx = np.argsort(Y)
        Y = Y[idx]
        X = X[idx, :]
        dim = int(np.sqrt(X.shape[1]))
        X = X.reshape(-1, dim, dim)
        return X, Y

def one_hot(Labels, num_classes):
    c = num_classes
    n = len(Labels)
    Y = torch.ones(size=[c, n]) * -1
    for i in range(n):
        Y[Labels[i].item(), i] = 1
    return Y.cuda()

if __name__ == '__main__':
    load_data(YALE)