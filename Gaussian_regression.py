import numpy as np
import matplotlib.pyplot as pl

n = 50
Xtest = np.linspace(-5, 10, n).reshape(-1,1)

def kernel(a, b, param):
    # print(np.sum(a**2,1))
    sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
    return np.exp(-.5 * (1/param) * sqdist)

param = 0.1
K_ss = kernel(Xtest, Xtest, param)

f_prior = np.random.multivariate_normal(np.zeros((n)), K_ss, size=(3)).T
pl.plot(Xtest, f_prior)
pl.axis([-5, 10, -3, 3])
pl.title('Three samples from the GP prior')
pl.show()

Xtrain = np.array([-4, -3, -2, -1,0,0.1,0.2,0.5,1, 2, 3, 4, 5]).reshape(-1,1)
ytrain = np.sin(Xtrain)

K = kernel(Xtrain, Xtrain, param)
K_s = kernel(Xtrain, Xtest, param)
mu = np.matmul(K_s.T,np.dot(np.linalg.inv(K),ytrain)).reshape((n,))

K_post = K_ss-np.matmul(K_s.T,np.matmul(np.linalg.inv(K),K_s))
K_post_diag = K_post.diagonal()
stdv = np.sqrt(K_post_diag)
print(mu,K_post,stdv)
f_post = np.random.multivariate_normal(mu,K_post,size=(3)).T
# print(f_post)

pl.plot(Xtrain, ytrain, 'bs', ms=8)
pl.plot(Xtest, f_post)
pl.gca().fill_between(Xtest.flat, mu-2*stdv, mu+2*stdv, color="#dddddd")
pl.plot(Xtest, mu, 'r--', lw=2)
pl.axis([-5, 10, -3, 3])
pl.title('Three samples from the GP posterior')
pl.show()