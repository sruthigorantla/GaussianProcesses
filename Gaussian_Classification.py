
# coding: utf-8

# In[222]:


import numpy as np
import matplotlib.pyplot as plt

def generate_classification_toy_data(n_train=200, mean_a=np.asarray([0, 0]), std_dev_a=1, mean_b=3, std_dev_b=0.5):

    # positive examples are distributed normally
    X1 = (np.random.randn(n_train, 2)*std_dev_a+mean_a).T

    # negative examples have a "ring"-like form
    r = np.random.randn(n_train)*std_dev_b+mean_b
    angle = np.random.randn(n_train)*2*np.pi
    X2 = np.array([r*np.cos(angle)+mean_a[0], r*np.sin(angle)+mean_a[1]])

    # stack positive and negative examples in a single array
    X_train = np.hstack((X1,X2))

    # label positive examples with +1, negative with -1
    y_train = np.zeros(n_train*2)
    y_train[:n_train] = 1
    y_train[n_train:] = -1
    return X_train, y_train


def plot_binary_data(X_train, y_train):
    plt.plot(X_train[0, np.argwhere(y_train == 1)], X_train[1, np.argwhere(y_train == 1)], 'ro')
    plt.plot(X_train[0, np.argwhere(y_train == -1)], X_train[1, np.argwhere(y_train == -1)], 'bo')


# In[228]:


from scipy.linalg import fractional_matrix_power

def kernel(a, b, param = 1):
    sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
    return np.exp(-.5 * (1/param) * sqdist)

def newtons_method(x_init, a):
    return 0
    
def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

## we assume f_prior_cpvariance function to be the kernel function
X_train, y_train = generate_classification_toy_data()

n = 10
Xtest1 = np.linspace(-5, 5, n).reshape(-1,1)
Xtest2 = np.linspace(-5, 5, n).reshape(-1,1)
X1, X2 = np.meshgrid(Xtest1, Xtest2)

X_test = np.hstack([X1.reshape(-1,1), X2.reshape(-1,1)]).T

K = kernel(X_train.T, X_train.T, 1)
K_ss = kernel(X_test.T, X_test.T, 1)
K_s = kernel(X_train.T, X_test.T, 1)

#print K.shape
#print K_ss.shape
#print K_s.shape

plot_binary_data(X_train, y_train)
#plt.plot(X_test[0], X_test[1], 'go')
_=plt.title("2D Toy classification problem")
plt.show()

f_sample_test = np.random.multivariate_normal(np.zeros(n*n), K_ss)
f_sample_prob = sigmoid(f_sample_test)

## Just based on prior covariance and mean
a = (np.round(f_sample_prob))
print(a.shape)
#plot_binary_data(X_train, y_train)
plt.plot(X_test[0,a == 0], X_test[1, a == 0], 'bo')
plt.plot(X_test[0,a == 1], X_test[1, a == 1], 'ro')
_=plt.title("2D Toy classification problem")
plt.show()

## Finding Posterior
def laplace_approximation(K, y_train):
    
    f = np.zeros(K.shape[0]).reshape(-1,1)
    y_train = y_train.reshape(-1,1)
    num_iteration = 20
    I = np.eye(K.shape[0])
    
    i = 0
    while i < num_iteration:
        sig = sigmoid(f)
        dia = sig*(1.0 - sig)
        #print dia.shape
        W = np.diag(dia[:,0])
        W_half = fractional_matrix_power(W, 0.5)
        L = np.linalg.cholesky(I + np.multiply(W_half, np.multiply(K, W_half)))
        b = np.matmul(W, f) + y_train - sig
        d = np.linalg.solve(L, np.matmul(W_half, np.matmul(K, b)))
        a = b - np.matmul(W_half, np.linalg.solve(L.T, d))
        f = np.matmul(K, a)
        
        i += 1
    
    return f

f_hat = laplace_approximation(K, y_train)
#print f_hat

def get_posterior(f_hat, X_train, y_train, X_test):
    
    y_train = y_train.reshape(-1,1)
    I = np.eye(K.shape[0])
    
    sig = sigmoid(f_hat)
    dia = sig*(1.0 - sig)
    W = np.diag(dia[:,0])
    W_half = fractional_matrix_power(W, 0.5)
    L = np.linalg.cholesky(I + np.multiply(W_half, np.multiply(K, W_half)))
    fs_mean = np.matmul(K_s.T, (y_train - sig))
    #print fs_mean.shape
    v = np.linalg.solve(L, np.matmul(W_half, K_s))
    V = K_ss - np.matmul(v.T, v)
    r = (1.0*fs_mean)/(np.sqrt(1.0 + (np.pi * np.diagonal(V))/8.0)).reshape(-1,1)
    pi_s = sigmoid(r)
    return pi_s

pi_s = get_posterior(f_hat, X_train, y_train, X_test)
print(pi_s)

posterior = np.round(np.array(2*pi_s[:, 0]))

plt.plot(X_test[0,posterior == 0], X_test[1, posterior == 0], 'bo', ms = 5)
plt.plot(X_test[0,posterior >= 1], X_test[1, posterior >= 1 ], 'ro', ms = 5)
plt.show()


# In[117]:


f = np.zeros(100).reshape(-1,1)

sig = sigmoid(f)
dia = sig*(1.0 - sig)
print(dia.shape)
np.diag(dia[:,0])

