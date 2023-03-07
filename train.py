import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import sys

from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import DataLoader
from utils.load_data import *
from utils.parser import parameter_parser
from utils.noise import Uniform_noise
from model.VGG import *
from model.SVM import SVMclassifier
from utils.KD import DistillKL

import warnings
warnings.filterwarnings("ignore")

torch.cuda.set_device('cuda:{}'.format(1))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def intro_data(dataname, datefile, noise_rate=0.1):
    if dataname == 'MNIST':
        train_transform = transforms.Compose([Uniform_noise(low=0, high=1, p=noise_rate),
                                        transforms.Resize(32),
                                        transforms.ToTensor()
                                        ])
        test_transform = transforms.Compose([transforms.Resize(32),
                                        transforms.ToTensor()
                                        ])
        train_data = torchvision.datasets.MNIST(root=datefile,
                                                train=True,
                                                transform=train_transform,
                                                download=False)
        test_data = torchvision.datasets.MNIST(root=datefile,
                                               train=False,
                                               transform=test_transform)
        c = 10
        channel = 1
    elif dataname == 'FashionMnist':
        train_transform = transforms.Compose([Uniform_noise(low=0, high=1, p=noise_rate),
                                              transforms.Resize(32),
                                              transforms.ToTensor()
                                              ])
        test_transform = transforms.Compose([transforms.Resize(32),
                                             transforms.ToTensor()
                                             ])
        train_data = torchvision.datasets.FashionMNIST(root=datefile,
                                                       train=True,
                                                       transform=train_transform,
                                                       download=True)
        test_data = torchvision.datasets.FashionMNIST(root=datefile,
                                                      train=False,
                                                      transform=test_transform)
        c = 10
        channel = 1
    elif dataname == 'cifar10':
        cifar10_mean = (0.49, 0.48, 0.45)
        cifar10_std = (0.25, 0.24, 0.26)
        train_transform = transforms.Compose([Uniform_noise(low=0, high=1, p=noise_rate),
                                              transforms.Resize(32),
                                              transforms.ToTensor(),
                                              transforms.Normalize(cifar10_mean, cifar10_std)
                                              ])
        test_transform = transforms.Compose([transforms.Resize(32),
                                             transforms.ToTensor(),
                                             transforms.Normalize(cifar10_mean, cifar10_std)
                                             ])
        train_data = torchvision.datasets.CIFAR10(root=datefile,
                                                     train=True,
                                                     transform=train_transform,
                                                     download=True)
        test_data = torchvision.datasets.CIFAR10(root=datefile,
                                                    train=False,
                                                    transform=test_transform)
        c = 10
        channel = 3
    elif dataname == 'stl10':
        stl10_mean = (0.4467106, 0.43980986, 0.40664646)
        stl10_std = (0.22414584, 0.22148906, 0.22389975)
        train_transform = transforms.Compose([Uniform_noise(low=0, high=1, p=noise_rate),
                                              transforms.Resize(32),
                                              transforms.ToTensor(),
                                              transforms.Normalize(stl10_mean, stl10_std)
                                              ])
        test_transform = transforms.Compose([transforms.Resize(32),
                                             transforms.ToTensor(),
                                             transforms.Normalize(stl10_mean, stl10_std)
                                             ])
        train_data = torchvision.datasets.STL10(datefile, split="train", transform=train_transform, download=False)
        test_data = torchvision.datasets.STL10(datefile, split="test", download=False, transform=test_transform)
        c = 10
        channel = 3
    else:
        train_transform = transforms.Compose([Uniform_noise(low=0, high=1, p=noise_rate),
                                              transforms.Resize(32),
                                              transforms.ToTensor()
                                              ])
        test_transform = transforms.Compose([transforms.Resize(32),
                                             transforms.ToTensor()
                                             ])
        transform = transforms.Compose([transforms.Resize(32),
                                        transforms.ToTensor()
                                        # gray -> GRB 3 channel (lambda function)
                                        # transforms.Normalize(mean=[0.0, 0.0, 0.0],
                                        #                      std=[1.0, 1.0, 1.0])
                                        ])
        dataset = MatDataset(filename=datefile+'/{}.mat'.format(dataname), transform=transform)
        train_size = int(0.7 * len(dataset.y))
        test_size = len(dataset.y) - train_size
        train_data, test_data = torch.utils.data.random_split(dataset, [train_size, test_size])
        train_data.dataset.transform = train_transform
        test_data.dataset.transform = test_transform
        c = int((max(dataset.y) - min(dataset.y) + 1))
        channel = 1
    return train_data, test_data, c, channel

def train(train_loader, model_e, model_svm, criterion_cls, criterion_kd, optimizer, epoch, gamma):
    model_e.train()
    running_loss = 0.0
    losses = AverageMeter()
    ACC = AverageMeter()
    F1 = AverageMeter()

    for i, (x, target) in enumerate(train_loader):
        x = x.float()
        target = target.reshape(-1)
        if torch.cuda.is_available():
            x = x.cuda()
            target = target.cuda()
        # ===================forward=====================
        z, logit = model_e(x)
        logit_t = model_svm(z.T.detach()).T

        # cls + kd
        loss_cls = criterion_cls(logit, target)
        loss_kd = criterion_kd(logit, logit_t)

        loss = loss_cls + gamma * loss_kd

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # optimize svm
        # Y = one_hot(target, num_classes=model_svm.c)
        # model_svm.optimize(z.T.detach(), Y)

        # metric
        pred = torch.argmax(logit, dim=1)
        acc = accuracy_score(target.cpu().numpy(), pred.detach().cpu().numpy())
        f1 = f1_score(target.cpu().numpy(), pred.detach().cpu().numpy(), average='macro')

        losses.update(loss.item(), x.size(0))
        ACC.update(acc, x.size(0))
        F1.update(f1, x.size(0))

        if i % 50 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'F1_score@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), loss=losses, top1=ACC, top5=F1))
            sys.stdout.flush()
    print(' * ACC@1 {top1.avg:.3f} F1@5 {top5.avg:.3f}'
          .format(top1=ACC, top5=F1))
    return losses.avg, ACC.avg, F1.avg

def optimize_svm(train_loader, model_e, model_svm, epoch):
    model_e.eval()
    with torch.no_grad():
        emb = []
        gt = []
        print('Generate embedding')
        for i, (x, target) in enumerate(train_loader):
            x = x.float()
            target = target.reshape(-1)
            if torch.cuda.is_available():
                x = x.cuda()
                target = target.cuda()

            # compute output
            z, logit = model_e(x)
            emb.append(z)
            gt.append(target)
        print('Optimize SVM teacher')
        emb = torch.cat(emb, dim=0)
        gt = torch.cat(gt, dim=0)
        Y = one_hot(gt, num_classes=model_svm.c)
        model_svm.optimize(emb.T.detach(), Y)

def test(test_loader, model_e, model_svm, criterion_cls, criterion_kd, epoch, gamma):
    model_e.eval()
    losses = AverageMeter()
    ACC = AverageMeter()
    F1 = AverageMeter()
    with torch.no_grad():
        for i, (x, target) in enumerate(test_loader):
            x = x.float()
            target = target.reshape(-1)
            if torch.cuda.is_available():
                x = x.cuda()
                target = target.cuda()
            z, logit = model_e(x)
            logit_t = model_svm(z.T.detach()).T

            # cls + kd
            loss_cls = criterion_cls(logit, target)
            loss_kd = criterion_kd(logit, logit_t)

            loss = loss_cls + gamma * loss_kd
            # metric
            pred = torch.argmax(logit, dim=1)
            acc = accuracy_score(target.cpu().numpy(), pred.detach().cpu().numpy())
            f1 = f1_score(target.cpu().numpy(), pred.detach().cpu().numpy(), average='macro')

            losses.update(loss.item(), x.size(0))
            ACC.update(acc, x.size(0))
            F1.update(f1, x.size(0))
            if i % 50 == 0:
                print('Test: [{0}/{1}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accuracy@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'F1_score@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(test_loader), loss=losses, top1=ACC, top5=F1))
                sys.stdout.flush()
    print(' Test ACC@1 {top1.avg:.3f} F1@5 {top5.avg:.3f}'
          .format(top1=ACC, top5=F1))
    return losses.avg, ACC.avg, F1.avg

if __name__ == '__main__':
    args = parameter_parser()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # load data
    data_name = 'MNIST'
    data_file = './data'
    noise_ratio = 0.1
    # train_data, test_data, c, channel = intro_data(dataname=data_name, datefile=data_file, noise_rate=noise_ratio)
    batch_size = 100
    T = 1
    learning_rate = 0.05
    Epoch = 200
    gamma = 1
    r = 0.05
    train_data, test_data, c, channel = intro_data(dataname=data_name, datefile=data_file, noise_rate=r)
    train_loader = DataLoader(dataset=train_data,
                              batch_size=batch_size,
                              shuffle=True)
    test_loader = DataLoader(dataset=test_data,
                             batch_size=batch_size,
                             shuffle=True)
    # model
    model_e = VGG16(in_channel=channel, n_classes=c)
    model_svm = SVMclassifier(W=[], b=[], d=256, c=c, cuda=True, k=1 - r)

    criterion_cls = nn.CrossEntropyLoss(reduction='mean')
    criterion_kd = DistillKL(T=1)

    # optimizer
    optimizer = optim.SGD(model_e.parameters(),
                          lr=learning_rate,
                          momentum=0.9,
                          weight_decay=5e-4)

    if torch.cuda.is_available():
        model_e.cuda()
        criterion_cls.cuda()
        criterion_kd.cuda()

    for epoch in range(1, Epoch):
        print("==> training...")
        train_loss, train_acc, train_f1 = train(train_loader, model_e, model_svm, criterion_cls, criterion_kd,
                                                optimizer, epoch, gamma)
        print('epoch {}, loss {:.2f}, acc {:.4f}, f1 {:.4f}'.format(epoch, train_loss, train_acc, train_f1))
        # optimize svm
        optimize_svm(train_loader, model_e, model_svm, epoch)
        test_loss, test_acc, test_f1 = test(test_loader, model_e, model_svm, criterion_cls, criterion_kd, epoch,
                                            gamma)
        print('Test, loss {:.2f}, acc {:.4f}, f1 {:.4f}'.format(test_loss, test_acc, test_f1))