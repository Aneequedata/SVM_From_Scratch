import os
import gzip
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from cvxopt import matrix, solvers
import time
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import recall_score as rec
from sklearn.metrics import precision_score as prec
from sklearn.utils import shuffle


def load_mnist(path, kind='train'):

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels


X_all_labels, y_all_labels = load_mnist('./', kind='train')

"""
We are only interested in the items with label 1, 5 in problem1.
Only a subset of 1000 samples per class will be used.
"""

indexLabel1 = np.where((y_all_labels==1))
xLabel1 =  X_all_labels[indexLabel1][:1000,:].astype('float64')
yLabel1 = y_all_labels[indexLabel1][:1000].astype('float64')

indexLabel5 = np.where((y_all_labels==5))
xLabel5 =  X_all_labels[indexLabel5][:1000,:].astype('float64')
yLabel5 = y_all_labels[indexLabel5][:1000].astype('float64')
yLabel5 = yLabel5 - 6.0

yLabel1[:] = 1
yLabel5[:] = -1


#concate the data

label1 = pd.DataFrame(xLabel1)
label5 = pd.DataFrame(xLabel5)
x = np.concatenate((label1,label5))
y = np.concatenate((yLabel1,yLabel5))

#data processing
scalar = MinMaxScaler()
x = scalar.fit_transform(x)
x_scaled = scalar.transform((x))

# split data into training,test data
x, y = shuffle(x_scaled, y, random_state=2010940) #shuffle the data
total_len = len(y)
train_len = int(total_len * 0.8)
x_train = x[:train_len,:]
y_train = y[:train_len]
x_test = x[train_len:,:]
y_test = y[train_len:]

"""
To train a SVM in case of binary classification we have to convert the labels of the two classes of interest into '+1' and '-1'.
"""







# Hyper-parameters
gamma = 3
C = 1
q = 16

threshold = 1e-4



# Kernel

def kernel(x, z):
    return (np.matmul(x, z.T) + 1) ** (gamma)


def predictor(alfa, x, sv, sv_y, b):
    return np.sign(np.dot((alfa * sv_y), kernel(sv, x)) + b)


def working_set(alfa, x, y, ker):
    y = y.flatten()
    condition_R = np.logical_or(np.logical_and(alfa <= C - threshold*C, y == 1),
                                np.logical_and(alfa >= threshold, y == -1))
    R = np.where(condition_R)
    R = R[0]

    condition_S = np.logical_or(np.logical_and(alfa <= C - threshold*C, y == -1),
                                np.logical_and(alfa >= threshold, y == 1))
    S = np.where(condition_S)
    S = S[0]

    # Gradient
    grad = -(y * (np.dot((np.outer(y, y) * ker), alfa) - 1))

    # Optimality
    m = np.max(grad[R])
    M = np.min(grad[S])

    data_R = np.zeros(len(R), dtype = {'names':('grad','index'),'formats':('float','int')})
    data_R['grad'] = grad[R]
    data_R['index'] = R
    data_R = np.sort(data_R,order = 'grad')
    R_index = data_R['index']


    data_S = np.zeros(len(S), dtype = {'names':('grad','index'),'formats':('float','int')})
    data_S['grad'] = grad[S]
    data_S['index'] = S
    data_S = np.sort(data_S,order = 'grad')
    S_index = data_S['index']

    # I
    I = R_index
    I = I[-q // 2:]

    # J
    J = S_index
    J = J[:q // 2]

    # Working Set
    Ws = np.sort(np.concatenate((I,J)))
    Ws_compl = list(set(np.arange(x.shape[0])) - set(Ws))
    flag = False
    if M - m > -threshold:
        flag = True
    difference = M-m

    return flag, Ws, Ws_compl, difference

pd = np.zeros(x_train.shape[0])
pm = np.zeros(x_train.shape[0])

def solver(x, y, alfa ,alfa_prev ):
    max_it = 1000

    # Computing the starting time
    start = time.time()

    # Reshaping the array y
    y = y.reshape(-1, 1)

    ker = kernel(x, x)
    iteration = 0
    for i in range(max_it):
        # Computing theflag and the parameters W, W_, Q with the function get_WS
        flag, Ws, Ws_compl, difference = working_set(alfa, x, y, ker)

        if flag:
            break

        # print('alpha_prev',alfa_prev)
        # print('alfa', alfa)
        # print('alfasum',np.sum(alfa))
        # print('alfa_prevsum', np.sum(alfa_prev))

        # computing alpha
        P = matrix((np.outer(y, y) * ker)[np.ix_(Ws, Ws)])
        q = matrix(np.dot(alfa_prev[Ws_compl].T, (np.outer(y, y) * ker)[np.ix_(Ws_compl, Ws)]) - 1)
        G = matrix(np.vstack((-np.eye(len(Ws)), np.eye(len(Ws)))))
        h = matrix(np.hstack((np.zeros(len(Ws)), np.ones(len(Ws)) * C)))
        A = matrix(y[Ws].reshape(1, -1))
        b = matrix(np.dot(- y[Ws_compl].T, alfa_prev[Ws_compl]))

        solvers.options['show_progress'] = False

        res = solvers.qp(P, q, G, h, A, b)
        alfa[Ws] = np.array(res['x']).T
        iteration += res["iterations"]
        alfa_prev = alfa

    # Computing the time elapsed
    time_elapsed = time.time() - start
    # computing b
    # K = self.kernel_gauss(self.X, self.X);
    # y = self.y.reshape(-1, 1)

    alfa = alfa.flatten()
    idx = np.where((alfa > threshold) & (alfa < C - threshold*C))
    indexes = idx[0]

    wx = np.dot((y * alfa.reshape(-1, 1)).T, ker[:, indexes])
    b = y[indexes] - wx.T
    b_opt = np.mean(b)
    flag, Ws, Ws_compl, difference = working_set(alfa, x, y, ker)
    # H = Q

    fun_val = (np.dot(np.dot(0.5 * alfa, (np.outer(y, y) * ker)), alfa.T)) - (np.sum(alfa))

    alfa_opt = alfa[indexes]
    sv = x_train[indexes]
    sv_y = y_train[indexes]


    return iteration, time_elapsed, fun_val, difference, alfa_opt, b_opt, sv, sv_y, res["status"]


iteration, time_elapsed, fun_val, difference, alfa_opt, b_opt, sv, sv_y, status = solver(x_train,y_train, pd, pm)


print("\nSetting values of the hyperparameters : ")

print("Optimal Gamma:", 3)
print("Optimal C:", 1)
print("q value : ", 16)
print('Status of the QP:', status)
print('Time:', time_elapsed)
print('Number of Iterations:', iteration)
print("Value of the Initial Objective Function:", 0)
print("Value of the Final Objective Function:", fun_val)
y_pred_train = predictor(alfa_opt, x_train, sv, sv_y, b_opt)

accuracy_train = acc(y_train, y_pred_train.flatten())
recall_train = rec(y_train, y_pred_train.flatten())
precision_train = prec(y_train, y_pred_train.flatten())
print('Performance train: ')
print('Accuracy: ', accuracy_train*100)
print('Recall: ', recall_train*100)
print('Precision: ', precision_train*100)
print('Difference of m(alpha) & M(alpha) :', difference )

y_pred_test = predictor(alfa_opt, x_test, sv, sv_y, b_opt)
accuracy_test = acc(y_test, y_pred_test)
recall_test = rec(y_test, y_pred_test)
precision_test = prec(y_test, y_pred_test)
print('Performance test: ')
print('Accuracy: ', accuracy_test*100)
print('Recall: ', recall_test*100)
print('Precision: ', precision_test*100)


