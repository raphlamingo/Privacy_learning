import math
import os
import pickle
import torch
import numpy as np
import scipy as sc
from torch import nn, optim
torch.set_default_dtype(torch.float64)

"""
List of functions:
random_mini_batches 
Read_data               (to read the data from real datasets)
Log_Reg                 (VFL)
ESA                     (LS, Clamped_LS, RCC1, RCC2, CLS)
"""

"""Randomly choosing mini-batched of data"""
# https://github.com/mrtzvrst/Inference-attack-in-vertical-federated-learning/blob/main/code_repository.py
def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    m = X.shape[0]
    mini_batches = []
    np.random.seed(seed)

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :]
    shuffled_Y = Y[permutation]
    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(
        m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


"""Reading the data and splitting it into test and training (return numpy array)"""
# https://machinelearningmastery.com/model-based-outlier-detection-and-removal-in-python/
def Read_data(filename):
    if filename == 'datasets/bank-additional-full.csv':
        with open(filename) as f:
            lines = f.readlines()
        X, Y = [], []
        for line in lines:
            temp = list(line.replace("\"", "").strip().split(';'))
            X.append(temp)

        X = X[1:]
        for i in range(len(X)):
            X[i][20] = 0 if X[i][20] == 'no' else 1

        for i in range(20):
            if X[0][i].replace('.', '', 1).isdigit():
                for j in range(len(X)):
                    X[j][i] = float(X[j][i])
            else:
                t0 = set()
                for j in range(len(X)):
                    t0.add(X[j][i])

                Dict0 = {}
                for j in t0:
                    for l in range(len(X)):
                        if X[l][i] == j and j not in Dict0:
                            Dict0[j] = [X[l][20], 1]
                        elif X[l][i] == j:
                            Dict0[j] = [Dict0[j][0] + X[l][20], Dict0[j][1] + 1]
                    Dict0[j][0] /= Dict0[j][1]
                print(Dict0)

                for j in range(len(X)):
                    X[j][i] = Dict0[X[j][i]][0]
        X = np.array(X)
        return X[:, np.delete(np.arange(20), 11)], X[:, 20]

    elif filename == 'datasets/sensor_readings_24_data.txt':
        with open(filename) as f:
            lines = f.readlines()
        f.close()
        X, Y = [], []
        for line in lines:
            temp = line.strip().split(',')
            X += [list(map(float, temp[0:24]))]
            Y += [temp[24]]

        SET, Dict, ind = set(Y), {}, 0
        for i in SET:
            Dict[i] = ind
            ind += 1

        for i in range(len(Y)):
            Y[i] = Dict[Y[i]]
        return np.array(X), np.array(Y)

    elif filename == 'datasets/Sat_train.txt':
        Xtr, Ytr = [], []
        Xts, Yts = [], []
        with open(filename) as file:
            lines = file.readlines()
            for line in lines:
                temp = list(map(float, line.split()))
                Xtr += [temp[0:36]]
                if temp[36] == 7:
                    Ytr += [int(temp[36] - 2)]
                else:
                    Ytr += [int(temp[36] - 1)]

        with open('datasets/Sat_test.txt') as file:
            lines = file.readlines()
            for line in lines:
                temp = list(map(float, line.split()))
                Xts += [temp[0:36]]
                if temp[36] == 7:
                    Yts += [int(temp[36] - 2)]
                else:
                    Yts += [int(temp[36] - 1)]

        return np.array(Xtr), np.array(Xts), np.array(Ytr), np.array(Yts)


class Log_Reg(nn.Module):#lR model
    def __init__(self, input_dim, output_dim):
        super(Log_Reg, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


"""ESA attack"""


def ESA(Weights, Biases, X, Y, k0_stp, i_stp, Num_of_Predictions, filename, STR, Truncate, Version):
    Num_of_Features = Weights.shape[1]
    Num_of_classes = Weights.shape[0]
    Num_of_samples = X.shape[0]

    t0 = int(0.9 * Num_of_Features)
    MSE = []
    k0 = 0
    while k0 < Num_of_Features:
        i = 0
        while i < t0:

            missing_features = np.mod(np.arange(k0, i + 1 + k0),
                                      Num_of_Features)
            ind, MSE_temp = 0, 0
            while ind < Num_of_Predictions:

                index = np.random.randint(0, Num_of_samples)

                z = np.matmul(Weights, X[index]) + Biases
                v = sc.special.softmax(z, axis=0)

                Wpas = Weights[:, missing_features]
                Wact = Weights[:, [j for j in range(Num_of_Features) if j not in missing_features]]
                X_act = X[index][[j for j in range(Num_of_Features) if j not in missing_features]]

                if 0 <= i < Num_of_classes - 1:
                    W, B = np.concatenate((Wpas, Wact), axis=1), Biases[0:-1] - Biases[1:]
                    W, A = W[0:-1, :] - W[1:, :], np.log(v[0:-1]) - np.log(v[1:])
                    X_pas = np.matmul(np.linalg.inv(W[0:i + 1, 0:i + 1]),
                                      (A[0:i + 1] - B[0:i + 1]) - np.matmul(W[0:i + 1, i + 1:], X_act))

                else:
                    A = np.log(v) - np.matmul(Wact, X_act) - Biases
                    Wpas, A = Wpas[0:-1, :] - Wpas[1:, :], A[0:-1] - A[1:]
                    if Version == 'Extended':
                        m0 = np.matmul(np.linalg.pinv(Wpas), A)[:, None]
                        m1 = np.identity(i + 1) - np.matmul(np.linalg.pinv(Wpas), Wpas)
                        m2 = np.matmul(np.matmul(m1, np.linalg.pinv(m1)), np.ones((i + 1, 1)) - 1 / 2 - m0)
                        X_pas = (m0 + m2).flatten()
                    else:
                        X_pas = np.matmul(np.linalg.pinv(Wpas), A)

                """Truncation"""
                if Truncate == 'yes':
                    X_pas[X_pas < 0], X_pas[X_pas > 1] = 0, 1

                MSE_temp += np.sum((X_pas - X[index][missing_features]) ** 2) / (i + 1)
                ind += 1
            MSE += [[k0, i, np.round(MSE_temp / Num_of_Predictions, decimals=5)]]
            print(k0, i, np.round(MSE_temp / Num_of_Predictions, decimals=5))
            i += i_stp
        k0 += k0_stp

    f = open(filename, 'wb')
    pickle.dump(MSE, f)
    f.close()

    return MSE
