import os, sys
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)
sys.path.append("..\\")

import gc
import math
from scipy import sparse
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
import torch.nn.functional as F
import utils.load_data as loader
from utils.promat import *

class SVMclassifier(nn.Module):
    def __init__(self, W, b, d, c, gama=0.1, k=0.9, cuda=True):
        super(SVMclassifier, self).__init__()
        self.d = d
        self.c = c
        if W==[] or b==[]:
            W, b = self.init_param(d, c)
        self.W = W.clone()
        self.oldW = W.clone()
        del W
        self.b = b.reshape(-1,1)
        self.gama = gama
        self.k = k
        self.cuda = cuda
        if cuda==True:
            self.W = self.W.cuda()
            self.oldW = self.oldW.cuda()
            self.b = self.b.cuda()

    def forward(self, X):
        predict = torch.mm(torch.t(self.W), X) + self.b
        return predict

    def svmLoss(self, X, Y):
        M = self.cal_M(X, Y, self.W, self.b)
        alphas, _ = self.cal_alphas(X, Y, M, self.W, self.b)
        M = M.detach()
        alphas = alphas.detach()
        Loss = self.cal_obj(X, Y, self.gama, self.W, self.b, alphas, M)
        # loss = torch.mm(self.W.t(), X) + self.b - Y
        # loss = torch.pow(loss, 2)
        # loss = torch.sum(loss)
        # Loss = loss + self.gama * (torch.norm(self.W))**2
        return Loss

    def init_param(self, d, c):
        # alphas = np.ones(shape=[n, 1]) / n
        # M = np.random.randn(c, n)
        # M = M * 10
        # oldW = 0
        W = torch.FloatTensor(d, c)
        b = torch.FloatTensor(c)
        stdv = 1. / math.sqrt(c)
        W.data.uniform_(-stdv, stdv)
        b.data.uniform_(-stdv, stdv)
        return W, b

    def cal_M(self, X, Y, W, b):
        '''M = Y .* (X * W) + Y .* b - 1'''
        M = Y * (torch.mm(W.T, X)) + Y * b - 1
        M = torch.clamp(M, min=0)
        return M

    def cal_alphas(self, X, Y, M, W, b):
        k = round(self.k * Y.shape[1])-1

        f = torch.mm(W.t(), X) + b - Y - Y * M
        f = torch.sum(f ** 2, axis=0)  # 按列相加
        f = f.t()
        f_sorted = torch.sort(f)[0]

        top_k = f_sorted[k]
        size = Y.shape[1]
        top_k = torch.t(top_k.repeat(size, 1)) + 10 ** -10
        sum_top_k = torch.sum(f_sorted[:k])
        sum_top_k = torch.t(sum_top_k.repeat(size, 1))
        T = top_k - f
        T = F.relu(T)
        alphas = torch.div(T, k * top_k - sum_top_k)


        lower_bound = 0.5 * (k * f_sorted[k] - torch.sum(f_sorted[:k]))
        upper_bound = 0.5 * (k * f_sorted[k + 1] - torch.sum(f_sorted[:k]))
        lamda = (lower_bound + upper_bound) / 2
        lamda = k / 2 * f_sorted[k + 1] + 0.5 * torch.sum(f_sorted[:k])
        # t = torch.sum(f_sorted[:k]) + 2 * lamda
        # t = t / (2 * k * lamda)
        # alphas = t - f / (2 * lamda)
        # alphas = torch.clamp(alphas, min=0.0)
        # alphas = alphas.reshape(-1, 1)
        return alphas.reshape(-1, 1), lamda

    def cal_W(self, X, Y, alphas, M):
        '''if n>1000, need to transfer to '''
        n = X.shape[1]
        d = self.d
        c = self.c
        gama = self.gama
        # temp = alphas.flatten()
        # '''solve the centralized method1'''
        # D = torch.diag(temp)
        # H = D - (torch.mm(alphas, alphas.T)) / torch.sum(alphas)
        # if self.cuda == False:
        #     S = torch.mm(torch.mm(X, H), X.t()) + gama * torch.eye(d)
        # else:
        #     S = torch.mm(torch.mm(X, H), X.t()) + gama * torch.eye(d).cuda()
        # m1 = torch.ge(S, S.t()).float()
        # m2 = torch.gt(S.t(), S).float()
        # S = S * m1 + S.t() * m2
        '''solve the centralized method2'''
        weigt_u = alphas / torch.sum(alphas)
        X_weight = torch.mul(X, weigt_u.reshape(1, -1))
        X_weight_mean = torch.sum(X_weight, dim=1)
        XH = X - X_weight_mean.reshape(-1, 1)
        S = 0
        for i in range(XH.size(1)):
            temp = torch.mm(XH[:, i].reshape(-1, 1), XH[:, i].t().reshape(1,-1))
            S = S + alphas[i]*temp
        if self.cuda == False:
            S = S + gama * torch.eye(d)
        else:
            S = S + gama * torch.eye(d).cuda()
        m1 = torch.ge(S, S.t()).float()
        m2 = torch.gt(S.t(), S).float()
        S = S * m1 + S.t() * m2
        Z = Y + torch.mul(Y, M)
        if self.cuda == False:
            S_12 = tensor_1_2_inv(S)
        else:
            S_12 = tensor_1_2_inv(S.cpu())
            S_12 = S_12.cuda()
        # S_12 = mat_1_2_inv(S.cpu().numpy())
        '''solve the centralized method1 with obtain the H = H D H^T'''
        # B = torch.mm(torch.mm(torch.mm(S_12, X), H), Z.t())
        '''solve the centralized method2, B=S^-1 X H D H^T Z^T= S^-1 (HX) D (HZ)^T'''
        Z_weight = torch.mul(Z, weigt_u.reshape(1, -1))
        Z_weight_mean = torch.sum(Z_weight, dim=1)
        HZ = Z - Z_weight_mean.reshape(-1, 1)
        XHDHT = 0
        for i in range(XH.size(1)):
            temp = torch.mm(XH[:, i].reshape(-1, 1), HZ[:, i].t().reshape(1,-1))
            XHDHT = XHDHT + alphas[i]*temp
        B = torch.mm(S_12, XHDHT)
        ''''''
        U, sigma, V = torch.svd(B) # torch.svd: return (U, sigma, V), numpy.linalg.svd: return (U, sigma, VT)
        # if d < c:
        #     T = torch.cat((torch.eye(d), torch.zeros(size=(c-d, d))), dim=0)
        # else:
        #     T = torch.cat((torch.eye(c), torch.zeros(size=(c, d-c))), dim=1)
        Q = torch.mm(U, V.t())
        W = torch.mm(S_12, Q)
        b = torch.mm((Z - torch.mm(W.t(), X)), alphas) / torch.sum(alphas)
        b = b.reshape(-1,1)
        return W, b

        # B = np.dot(np.dot(np.dot(S_12, X), H), Z.T)
        # U, sigma, VT = np.linalg.svd(B)
        # if d < c:
        #     T = np.vstack((np.eye(d), np.zeros(shape=[c - d, d])))
        # else:
        #     T = np.hstack((np.eye(c), np.zeros(shape=[c, d - c])))
        # Q = np.dot(np.dot(U, T.T), VT)
        # W = np.dot(S_12, Q)
        # b = np.dot((Z - np.dot(W.T, X)), alphas) / sum(alphas)
        # return W, b
        pass

    def cal_obj(self, X, Y, gama, W, b, alphas, M):
        T = torch.mm(W.t(), X) + b - Y - torch.mul(Y, M)
        t2 = torch.pow(T, 2)
        t = torch.sum(t2, dim=0)  # sum according by row
        # alphas = alphas.reshape((n, 1))
        #t = t.reshape((1, n))
        # a = torch.mul(t, alphas)
        a = t.reshape(-1,1) * alphas.reshape(-1,1)
        obj = torch.sum(a) + gama * torch.norm(W)
        return obj

    def optimize(self, X, Y, Iteration=5):
        W = self.W.clone()
        b = self.b.clone()
        groundtruth = torch.argmax(Y, dim=0)
        n = X.shape[1]
        err = 1
        iter = 1
        obj = []
        oldW = self.oldW
        while err>10**-2 and iter<Iteration:
            # update M
            M = self.cal_M(X, Y, W, b)
            # update alpha
            alphas, lamda = self.cal_alphas(X, Y, M, W, b)
            # update W, b
            W, b = self.cal_W(X, Y, alphas, M)

            diff = torch.norm(input=(W-oldW))
            obj.append(self.cal_obj(X, Y, self.gama, W, b, alphas, M).item())
            predict = torch.mm(W.t(), X) + b
            predY = torch.argmax(predict, dim=0)
            acc = torch.sum((groundtruth==predY).float()) / n
            print('diff: %.5f, obj: %.5f, acc: %.5f'%(diff.item(), obj[iter-1], acc.item()))
            if iter > 1:
                err = abs(obj[iter-1] - obj[iter-2])
            iter = iter + 1
            oldW = W
        predict = torch.mm(W.t(), X) + b
        predY = torch.argmax(predict, dim=0)
        acc = torch.sum((groundtruth == predY).float()) / n
        self.W = W
        self.b = b
        self.oldW = W
        gc.collect()