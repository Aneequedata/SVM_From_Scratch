import os
import gzip
import cvxopt
from cvxopt import solvers
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
import time



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


#concate the data
x = np.concatenate((xLabel1,xLabel5))
y = np.concatenate((yLabel1,yLabel5))

#data processing
scalar = MinMaxScaler()
x = scalar.fit_transform(x)

# split data into training,test data
x, y = shuffle(x, y, random_state=2010940) #shuffle the data
total_len = len(y)
train_len = int(total_len * 0.8)
x_train = x[:train_len,:]
y_train = y[:train_len]
x_test = x[train_len:,:]
y_test = y[train_len:]

"""
To train a SVM in case of binary classification we have to convert the labels of the two classes of interest into '+1' and '-1'.
"""


def polynomial_kernel(x, y, gamma=3.0):
    return (1 + np.dot(x, y)) ** gamma

class SVM(object):

    def __init__(self, kernel=polynomial_kernel, C =3.0, gamma=3.0):
        self.kernel = kernel
        self.C = float(C)
        self.gamma = float(gamma)
        self.dual_obj = 0.0

    def fit(self, x, y,iters=100):
        n_samples, n_features = x.shape

        # Kernel matrix
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i,j] = self.kernel(x[i], x[j], gamma=self.gamma)

        P = cvxopt.matrix(np.outer(y,y) * K)
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y.reshape(1,-1))
        b = cvxopt.matrix(np.zeros(1))
        G = cvxopt.matrix(np.vstack((np.eye(n_samples) * -1, np.eye(n_samples))))
        h = cvxopt.matrix(np.hstack((np.zeros(n_samples), np.ones(n_samples) * self.C)))

        # solve QP problem
        solvers.options['maxiters'] = iters
        solvers.options['show_progress'] = False
        solution = solvers.qp(P, q, G, h, A, b)

        # Lagrange multipliers
        alpha = np.ravel(solution['x'])

        # The final value of objective function of dual problem
        self.dual_obj = solution['primal objective']

        # Support vectors have non zero lagrange multipliers
        sv = alpha > 1e-5
        ind = np.arange(len(alpha))[sv]
        self.alpha = alpha[sv]
        self.sv = x[sv]
        self.sv_y = y[sv]

        self.num_KTT_viloation = len(alpha) - len(self.alpha)

        # Intercept
        self.b = 0
        for n in range(len(self.alpha)):
            self.b += self.sv_y[n]-np.sum(self.alpha * self.sv_y * K[ind[n],sv])
        if len(self.alpha)>0:
            self.b /= len(self.alpha)
        else:
            raise ValueError("No Solution!")

    def predict(self, x):
        y_predict = np.zeros(len(x))
        for i in range(len(x)):
            s = 0
            for a, sv_y, sv in zip(self.alpha, self.sv_y, self.sv):
                s += a * sv_y * self.kernel(x[i], sv)
            y_predict[i] = s
        return np.sign(y_predict + self.b)

    def score(self, x, y):

        y_pred = self.predict(x)
        acc = np.sum(y_pred == y) / len(y)
        return acc

def Kfold_cv(K,x,y,C,gamma,iters):
    strtfdKFold = StratifiedKFold(n_splits=K)
    kfold = strtfdKFold.split(x, y)
    scores = []
    #using k fold cross validation
    for k, (train, val) in enumerate(kfold):
        clf = SVM(polynomial_kernel, C=C, gamma=gamma)
        clf.fit(x[train, :], y[train], iters=iters)
        score = clf.score(x[val, :], y[val])
        scores.append(score)
    #Cross-Validation accuracy
    cross_val_acc = np.mean(scores)
    return cross_val_acc

def Grid_Search_SVM(params_dict,x,y,iters,K):
    C_list, gamma_list = params_dict["C"], params_dict["gamma"]
    best_cross_val_acc = 0.0
    best_C , best_gamma = 0.0, 0.0
    K = K
    for C in C_list:
        for gamma in gamma_list:
            cross_val_acc = Kfold_cv(K,x,y,C,gamma,iters)
            #identify the best parameters
            if best_cross_val_acc < cross_val_acc:
                best_cross_val_acc = cross_val_acc
                best_C = C
                best_gamma  = gamma

    return best_C, best_gamma


# using gridsearch and Kfold_cv method to set the parameter C and gamma
parameters = { 'C':[1.0,2.0,3.0], 'gamma':[1.0,2.0,3.0]}
iterations = 100 # optimization iterations
K = 5 # K fold
print("\nSetting values of the hyperparameters : ")
best_C, best_gamma = Grid_Search_SVM(params_dict = parameters,x=x_train,y=y_train,iters=iterations,K=K)
print("C = {}, gamma = {}".format(best_C, best_gamma))

#initial the SVM model using the selected C and gamma
best_svm = SVM(polynomial_kernel,C= best_C,gamma=best_gamma)

start_time = time.time()
# Start Training on total training data and get the best svm model
best_svm.fit(x_train, y_train,iters=iterations)
# optimization time necessary
optim_time = time.time()-start_time

# training accuracy
train_acc = best_svm.score(x_train,y_train)
print("\nClassification rate on the training set: {:.4}%".format(train_acc*100))

# testing accuracy
test_acc = best_svm.score(x_test,y_test)
print("\nClassification rate on the test set: {:.4}%".format(test_acc*100))

# confuse matrix
y_pred = best_svm.predict(x_test)
confuse_matrix = confusion_matrix(y_test, y_pred)
print("\nThe confusion matrix:")
print(confuse_matrix)

print("\nTime necessary for the optimization: {:.4}s".format(optim_time))

print("\nNumber of optimization iterations:{}".format(int(iterations)))

m_alpha = np.max(best_svm.alpha)
M_alpha = np.min(best_svm.alpha)
diff_alpha = m_alpha - M_alpha
print("\nDifference between m(alpha) and M(alpha): {:.4}".format(diff_alpha))

# record the initial objective of dual problem
init_svm = SVM(polynomial_kernel,C= best_C,gamma=best_gamma)
init_svm.fit(x_train, y_train,iters=1)
dual_obj_init = init_svm.dual_obj
print("\nThe initial objective of dual problem: {:.4}".format(dual_obj_init))

# record the final objective of dual problem
dual_obj_final = best_svm.dual_obj
print("\nThe final objective of dual problem: {:.4}".format(dual_obj_final))

print("\n{} violation of the KKT condition exist".format(best_svm.num_KTT_viloation))





