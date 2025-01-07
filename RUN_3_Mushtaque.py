# Libraries
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

# Data extraction
def load_mnist(path, kind='train'):
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte.gz' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte.gz' % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)

    return images, labels

cwd = os.getcwd()
X_all_labels, y_all_labels = load_mnist(cwd, kind='train')

indexLabel1 = np.where((y_all_labels == 1))
xLabel1 = X_all_labels[indexLabel1][:1000, :].astype('float64')
yLabel1 = y_all_labels[indexLabel1][:1000].astype('float64')

indexLabel5 = np.where((y_all_labels == 5))
xLabel5 = X_all_labels[indexLabel5][:1000, :].astype('float64')
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
X_train = x[:train_len,:]
Y_train = y[:train_len]
X_test = x[train_len:,:]
Y_test = y[train_len:]




# Hyper-parameters
gamma = 3
C = 1
q = 2
threshold = 1e-3

# Kernel
def kernel(x, z):
    return (np.matmul(x, z.T) + 1) ** (gamma)


def predictor(alfa, x, sv, sv_y, b):
    # Make sure all input arrays have the same number of rows as the x array
    if x.shape[0] != sv.shape[0]:
        sv = sv[:x.shape[0],:]
        sv_y = sv_y[:x.shape[0]]
    # Make sure the alfa array has the same number of elements as the sv and sv_y arrays
    if alfa.shape[0] != sv.shape[0]:
        alfa = alfa[:sv.shape[0]]
    return np.sign(np.dot((alfa * sv_y), kernel(sv, x)) + b)




def working_set(alfa, x, y, ker):
    y = y.flatten()
    condition_R = np.logical_or(np.logical_and(alfa <= C - threshold*C, y == 1),
                                np.logical_and(alfa >= threshold, y == -1))
    R = np.where(condition_R)
    R = R[0]

    condition_S = np.logical_or(np.logical_and(alfa > threshold, y == 1),
                                np.logical_and(alfa <= C - threshold*C, y == -1))
    S = np.where(condition_S)
    S = S[0]

    K_R = ker[R, :][:, R]
    K_S = ker[S, :][:, S]
    K_RS = ker[R, :][:, S]

    return R, S, K_R, K_S, K_RS

def MVP(alfa, X, y, b, ker, R, S, K_R, K_S, K_RS, epsilon=1e-3):
    y = y.flatten()
    R_size = len(R)
    S_size = len(S)

    for i in range(R_size):
        a_i = alfa[R[i]]
        y_i = y[R[i]]
        e_i = b - y_i
        for j in range(S_size):
            a_j = alfa[S[j]]
            y_j = y[S[j]]
            e_j = b - y_j
            if abs(y_i - y_j) > epsilon:
                L = max(0, a_j - a_i)
                H = min(C, C + a_j - a_i)
            else:
                L = max(0, a_i + a_j - C)
                H = min(C, a_i + a_j)
            if L == H:
                continue
            eta = 2 * K_RS[i][j] - K_R[i][i] - K_S[j][j]
            if eta >= 0:
                continue
            a_j_new = a_j - y_j * (e_i - e_j) / eta
            if a_j_new > H:
                a_j_new = H
            elif a_j_new < L:
                a_j_new = L
            if abs(a_j - a_j_new) < epsilon:
                continue
            a_i_new = a_i + y_i * y_j * (a_j - a_j_new)
            b_new = b - e_i - y_i * (a_i_new - a_i) * K_R[i][i] - y_j * (a_j_new - a_j) * K_RS[i][j]
            b_new_S = b - e_j - y_i * (a_i_new - a_i) * K_RS[i][j] - y_j * (a_j_new - a_j) * K_S[j][j]

            if a_i_new > threshold and a_i_new < C - threshold:
                b = b_new
            elif a_j_new > threshold and a_j_new < C - threshold:
                b = b_new_S
            else:
                b = (b_new + b_new_S) / 2

            alfa[R[i]] = a_i_new
            alfa[S[j]] = a_j_new

        return alfa, b

def SMO(x, y, ker, C=1, threshold=1e-3, max_iter=1e4):
    alfa = np.zeros(len(x))
    b = 0
    iteration = 0
    while iteration < max_iter:
        alfa_prev = np.copy(alfa)
        b_prev = b

        R, S, K_R, K_S, K_RS = working_set(alfa, x, y, ker)
        alfa, b = MVP(alfa, x, y, b, ker, R, S, K_R, K_S, K_RS)

        alfa_diff = alfa - alfa_prev
        b_diff = b - b_prev
        cond = np.linalg.norm(alfa_diff) + abs(b_diff)

        if cond < threshold:
            break

        iteration += 1

    sv = x[alfa > threshold]
    sv_y = y[alfa > threshold]
    alfa_sv = alfa[alfa > threshold]

    return alfa, b, sv, sv_y, max_iter
for c in [1.0]:
    n = len(Y_train)

    # Set up the input matrices for the solvers.qp function
    P = matrix(np.outer(Y_train, Y_train) * kernel(X_train, X_train))  # Quadratic part of the objective function
    P = np.asarray(P)  # Convert the P matrix to a NumPy array
    P.shape = (n, n)  # Set the dimensions of the P matrix to (n, n)
    P = matrix(P)

    q = matrix(-np.ones(n))  # Linear part of the objective function
    G = matrix(-np.eye(n))  # Constraints (inequality constraints)
    h = matrix(np.zeros(n))  # Right-hand side of the constraints
    A = matrix(Y_train.reshape(1, -1))  # Linear part of the objective function
    A = np.asarray(A)  # Convert the A matrix to a NumPy array
    A.reshape(1, n)  # Reshape the A matrix to have n columns
    A = matrix(A)  # Convert the A matrix back to a cvxopt.base.matrix
    b = matrix(np.zeros(1))  # Right-hand side of the constraints



    # Solve the dual problem using the solvers.qp function
    sol = solvers.qp(P, q, G, h, A, b)

    # Extract the final value of the objective function from the solvers.qp function's output
    print(f'C = {C}: final objective value = {sol["x"]}')


# Train
start = time.time()

N_train = len(X_train)
K = np.zeros((N_train, N_train))

for i in range(N_train):
    for j in range(N_train):
        K[i][j] = kernel(X_train[i], X_train[j])

alfa, b, sv, sv_y ,iterations = SMO(X_train, Y_train, K, C=C, threshold=threshold,max_iter=1e3)

end = time.time()


# Test
Y_pred = predictor(alfa, X_test, sv, sv_y, b)
Y_pred_train = predictor(alfa, X_train, sv, sv_y, b)

print("Value of the Initial Objective Function:", 0)
print(f'C = {C}: final objective value = {sol["x"]}')
print("\nSetting values of the hyperparameters : ")

print("Optimal Gamma:", 3)
print("Optimal C:", 1)
print("Execution time:", end - start)
print("iterations:",  iterations)

print('Performance train: ')
print("Train accuracy:", acc(Y_train, Y_pred_train)*100)
print("Train recall:", rec(Y_train, Y_pred_train , average= None)*100)
print("Train precision:", prec(Y_train, Y_pred_train, average= None)*100)

print('Performance test: ')
print("Test accuracy:", acc(Y_test, Y_pred)*100)
print("Test recall:", rec(Y_test, Y_pred, average= None)*100)
print("Test precision:", prec(Y_test, Y_pred, average= None)*100)


