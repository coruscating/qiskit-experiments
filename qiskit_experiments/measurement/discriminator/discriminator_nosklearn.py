"""
Linear and Quadratic Discriminant Analysis calculations without sklearn.
"""

import numpy as np
from numpy.linalg import inv
from math import log

    

class LDAnosklearn():
    """Calculation of Linear Discriminant Analysis without sklearn."""
    
    def __init__(self):
        return None
    
    def fit(self, X, y):
        """Calculation for Linear Discriminant Analysis (LDA) model according to the 
        given training data and parameters without sklearn.
        
        The formula for the LDA classification has been taken from Eqn. 4.11 of the book,
        Hastie, Trevor, et al. The Elements of Statistical Learning. 
        Second Edition. Stanford, California. August 2008.
     
        Parameters
        ----------
        
        X : array-like of shape (n_samples,n_features) where n_features=2 (0 and 1)
            Training data.
            
        y : array-like of shape (n_samples,)
            Target values.
        
        """
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
        """Perform classification on an array of test vectors X.
        
        The predicted class C for each sample in X is returned.
        
        Parameters
        ----------
        
        X : array-like of shape (n_samples, n_features) where n_features=2 (0 and 1)
        
        Returns
        -------
        
        C : ndarray of shape (n_samples,)
        
        """
        
        LHS = np.matmul(np.matmul(X,self.sigma_inv),(self.X1_mean-self.X0_mean).T)
        a = []
        for i in range(len(LHS)):
            if (LHS[i]>self.RHS):
                a.append(1)
            else: a.append(0)
        a = np.array(a)
        return a
    
    def score(self,X,y_true):
        """Return the mean accuracy on the given test data and labels.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) where n_features=2 (0 and 1)
            Test samples.
            
        y : array-like of shape (n_samples,)
            True labels for `X`.
            
        Returns
        -------
        score : float
            Mean accuracy of ``self.predict(X)`` wrt. `y`.
        
        """
        y_pred = self.predict(X)
        count = 0
        for i in range(len(y_true)):
            if (y_true[i]==y_pred[i]):
                count = count + 1
        return count/len(y_true)

            
        


class QDAnosklearn():
    """Calculation of Quadratic Discriminant Analysis without sklearn."""
    
    def __init__(self):
        return None
    
    def fit(self, X, y):
        """Calculation for Quadratic Discriminant Analysis (QDA) model according to the 
        given training data and parameters without sklearn.
        
        The formula for the QDA classification has been taken from Eqn. 4.12 of the book,
        Hastie, Trevor, et al. The Elements of Statistical Learning. 
        Second Edition. Stanford, California. August 2008.
        
        Parameters
        ----------
        
        X : array-like of shape (n_samples,n_features) where n_features=2 (0 and 1)
            Training data.
            
        y : array-like of shape (n_samples,)
            Target values.
        
        """
        
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
        """Perform classification on an array of test vectors X.
        
        The predicted class C for each sample in X is returned.
        
        Parameters
        ----------
        
        X : array-like of shape (n_samples, n_features) where n_features=2 (0 and 1)
        
        Returns
        -------
        
        C : ndarray of shape (n_samples,)
        
        """
        
        
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
        """Return the mean accuracy on the given test data and labels.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) where n_features=2 (0 and 1)
            Test samples.
            
        y : array-like of shape (n_samples,)
            True labels for `X`.
            
        Returns
        -------
        score : float
            Mean accuracy of ``self.predict(X)`` wrt. `y`.
        
        """
        y_pred = self.predict(X)
        count = 0
        for i in range(len(y_true)):
            if (y_true[i]==y_pred[i]):
                count = count + 1
        return count/len(y_true)

       

