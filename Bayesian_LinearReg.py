import csv
import numpy as np
import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt

## Calculating the function values as W . phi(X

def f(X, weights):
	phiX = phi(X, len(weights))
	return np.matmul(phiX, weights)

def phi(X, num_phis):
	
	phiX = np.zeros((X.shape[0], num_phis))
	for i in range(X.shape[0]):
		phi = [phii(i, j, X) for j in range(num_phis)]
		phiX[i,:] = (np.array(phi)).reshape(1,-1)
	
	return phiX

def phii(i,j, X):
	if j==0:
		return X[i]
	elif j==1:
		return 1


epsilon = 0.2 
orig_weights = np.array([0.5, -0.3]).reshape(-1,1)
X = np.random.uniform(-1, 1, 100)

# Varying No. of given points to see the effect estimated distributions

for num_points in range(1, 20, 5):

    epsilon = 0.1 # Noise Variance
    orig_weights = np.array([0.5, -0.3]).reshape(-1,1)
    X = np.random.uniform(-1, 1, 100)

    Xn = X[0:num_points] # Randomly Sampled Points
    Xn = (np.sort(Xn)).reshape(-1,1)
    delta = [np.random.normal(0,epsilon,1) for i in range(len(Xn))]

    t = f(Xn, orig_weights)
    y = [t[i]+delta[i][0] for i in range(t.shape[0])]

    weights_len = 2
    plt.scatter(np.sort(Xn), y)

    prior_mean = 0
    alpha = 2.0

    noise_mean = 0
    beta = (1.0/epsilon)**2
    Xn = Xn.reshape((-1,1))

    des_mat = phi(Xn, weights_len)	
	
    ## Gaussian conditional and marginal formulae
    posterior_cov_inv = alpha*np.eye(weights_len) + beta*np.matmul(des_mat.T, des_mat)
    posterior_cov = np.linalg.inv(posterior_cov_inv)
    posterior_mean = beta * np.matmul(posterior_cov, np.matmul(des_mat.T, t))

    pts_to_plt = np.sort(X)
    mean = posterior_mean
    cov =  posterior_cov
    
    for i in range(0,5):
        
        sample = np.random.multivariate_normal([mean[0,0], mean[1,0]], cov, 1)
        print ("Sample : ",sample)
        sample = np.array(sample).reshape(-1,1)
        plt.plot(pts_to_plt, f(pts_to_plt, sample))

    plt.show()