# -*- coding: utf-8 -*-
import numpy as np
def get_features(X_train, X_test, eigenfaces, y_test_len,k):
    templateFeatures = np.zeros((X_train.shape[1],k))
    
    for i in range(X_train.shape[1]):
        face = X_train[:,i]
        for j in range(k):
           templateFeatures[i,j] = np.dot(eigenfaces[:,j].transpose(), face)         
           
           
    queryFeatures = np.zeros((X_test.shape[1],k))
    
    for i in range(y_test_len):
        face = X_test[:,i]
        for j in range(k):
            queryFeatures[i,j] = np.dot(eigenfaces[:,j].transpose(), face)
            

    return templateFeatures, queryFeatures

def PCA(X):
    
    # Squash it
    X = X.transpose()
    
    # Get the mean face
    
    mean_x = X.mean(axis=1)
    
    # Subtract the mean face - center everybody
    
    for col in range(X.shape[1]):
        X[:,col] =  X[:,col] - mean_x
    
    # Compute the covariance matrix C
    
    C = np.cov(X.transpose())
    
    # Get the eigenfaces from C
    
    evals, evecs = np.linalg.eig(C)
    
    # Show some eigenfaces 
    
    eigenfaces = np.dot(X, evecs)

    return eigenfaces, X, mean_x