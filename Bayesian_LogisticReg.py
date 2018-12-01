# from linear_models.bayes_logistic import EBLogisticRegression,VBLogisticRegression
from linear_models.bayes_logistic_2 import EBLogisticRegression
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import cm
# %matplotlib inline

# create data set 
np.random.seed(0)
n_samples  = 500
def f(x):
    if np.sum(x) == 0:
        return -1
    return np.sign(x[0]**2 - 10*x[1] )
    # return np.sign(np.sum(x))


# Some useful variables
d = 2
n_samples = 500
lims = [5,5]
x = np.zeros((n_samples, d))
y = np.zeros((n_samples, 1))

# x1 = np.random.multivariate_normal(mean[0], cov, n_samples)
# x2 = np.random.multivariate_normal(mean[1], cov, n_samples)
# x = np.concatenate((x1,x2))
# y = len(x1)*[1] + len(x2)*[0]
# x = np.asarray(x)
# y = np.asarray(y)
# plt.scatter(x[:,0],x[:,1])
# plt.show()
for i in range(len(x)):
    x[i][0] = np.random.uniform(-lims[0],lims[0])
    x[i][1] = np.random.uniform(-lims[1],lims[1])
    y[i] = f(x[i])
y = y.flatten()
# x = [[0,0],[1,1]]
# y = [-1,1]
x = np.asarray(x)
y = np.asarray(y)
def plot_binary_data(X_train, y_train):
    plt.plot(X_train[0, np.argwhere(y_train == 1)], X_train[1, np.argwhere(y_train == 1)], 'ro')
    plt.plot(X_train[0, np.argwhere(y_train == -1)], X_train[1, np.argwhere(y_train == -1)], 'bo')
    plt.show()
plot_binary_data(x.T,y)


eblr = EBLogisticRegression(tol_solver = 1e-3)
eblr.fit(x,y)
print(eblr.coef_, eblr.sigma_)
# create grid for heatmap
n_grid = 500
max_x      = np.max(x,axis = 0)
min_x      = np.min(x,axis = 0)
X1         = np.linspace(min_x[0],max_x[0],n_grid)
X2         = np.linspace(min_x[1],max_x[1],n_grid)
x1,x2      = np.meshgrid(X1,X2)
Xgrid      = np.zeros([n_grid**2,2])
Xgrid[:,0] = np.reshape(x1,(n_grid**2,))
Xgrid[:,1] = np.reshape(x2,(n_grid**2,))


eblr_grid = eblr.predict_proba(Xgrid)[:,1]
print(eblr_grid)
grids = [eblr_grid]
lev   = np.linspace(0,1,11)  
titles = ['Bayesian Logistic Regression']
for title, grid in zip(titles, grids):
    plt.figure(figsize=(8,6))
    plt.contourf(X1,X2,np.reshape(grid,(n_grid,n_grid)),
                 levels = lev,cmap=cm.coolwarm)
    plt.plot(x[y==-1,0],x[y==-1,1],"bo", markersize = 3)
    plt.plot(x[y==1,0],x[y==1,1],"ro", markersize = 3)
    plt.colorbar()
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    # plt.savefig("./plot.png")
    plt.show()