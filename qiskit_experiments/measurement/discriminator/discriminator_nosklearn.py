import numpy as np
from numpy.linalg import inv
from math import log

    

class LDAnosklearn():
    
    def __init__(self):
        return None
    
    def fit(self, X, y):
        X0 = X[y==0]
        X1 = X[y==1]
        self.X0_mean = np.mean(X0,axis=0).reshape(1,2)
        self.X1_mean = np.mean(X1,axis=0).reshape(1,2)
        N_zeros = len(X0)
        N_ones = len(X1)
        K=2
        sigma_X0 = np.dot((X0-self.X0_mean).T,(X0-self.X0_mean))/(N_zeros-K)
        sigma_X1 = np.dot((X1-self.X1_mean).T,(X1-self.X1_mean))/(N_ones-K)
        sigma = sigma_X0 + sigma_X1
        self.sigma_inv = inv(sigma)
        self.RHS = 0.5*np.matmul(np.matmul((self.X1_mean+self.X0_mean),self.sigma_inv),(self.X1_mean-self.X0_mean).T) - log(N_ones/N_zeros)
        return self
    
    def predict(self, X):
        LHS = np.matmul(np.matmul(X,self.sigma_inv),(self.X1_mean-self.X0_mean).T)
        a = []
        for i in range(len(LHS)):
            if (LHS[i]>self.RHS):
                a.append(1)
            else: a.append(0)
        a = np.array(a)
        return a
    
    def score(self,X,y_true):
        y_pred = self.predict(X)
        count = 0
        for i in range(len(y_true)):
            if (y_true[i]==y_pred[i]):
                count = count + 1
        return count/len(y_true)

            
        


class QDAnosklearn():
    
    def __init__(self):
        return None
    
    def fit(self, X, y):
        X0 = X[y==0]
        X1 = X[y==1]
        self.X0_mean = np.mean(X0,axis=0).reshape(1,2)
        self.X1_mean = np.mean(X1,axis=0).reshape(1,2)
        N_zeros = len(X0)
        N_ones = len(X1)
        K=2
        self.sigma_X0 = np.dot((X0-self.X0_mean).T,(X0-self.X0_mean))/(N_zeros-K)
        self.sigma_X1 = np.dot((X1-self.X1_mean).T,(X1-self.X1_mean))/(N_ones-K)
        sigma = self.sigma_X0 + self.sigma_X1
        self.sigma_inv = inv(sigma)
        self.log_sigma_X0 = np.log(np.linalg.det(self.sigma_X0))
        self.log_sigma_X1 = np.log(np.linalg.det(self.sigma_X1))
        return self
    
    def predict(self, X):
        a = []
        for p in range(X.shape[0]):
            delta_0 = -0.5*self.log_sigma_X0 - 0.5*np.matmul(np.matmul((X[p]-self.X0_mean),inv(self.sigma_X0)),(X[p]-self.X0_mean).T) + 0.5
            delta_1 = -0.5*self.log_sigma_X1 - 0.5*np.matmul(np.matmul((X[p]-self.X1_mean),inv(self.sigma_X1)),(X[p]-self.X1_mean).T) + 0.5
            if (delta_0>delta_1):
                a.append(0)
            else:
                a.append(1)
        return a
    
    def score(self,X,y_true):
        y_pred = self.predict(X)
        count = 0
        for i in range(len(y_true)):
            if (y_true[i]==y_pred[i]):
                count = count + 1
        return count/len(y_true)

       

