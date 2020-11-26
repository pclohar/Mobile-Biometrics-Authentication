# -*- coding: utf-8 -*-
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import get_images

from LBP import LBP
from sklearn.svm import SVC
from evaluate import compute_rates, plot_scoreDist, plot_det, plot_roc
from PCA import PCA, get_features
from sklearn.model_selection import train_test_split
#from sklearn.svm import SVC

def extract_image_vectors(image_directory,i):
    X, y = get_images.get_images(image_directory,i)
    #X, y = get_landmarks.get_landmarks(X, y, 'landmarks/', 5, False)
    X_train, X_test , y_train, y_test = train_test_split(X, y, train_size = 0.70, random_state = 10)
    return X_train, X_test , y_train, y_test

def matching(X_train, X_test, y_train, y_test):

    
    scaler = MinMaxScaler();
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    svm = SVC(decision_function_shape='ovo',kernel = 'rbf', gamma='scale', probability=True)
    svm.fit(X_train,y_train)
    y_predict= svm.predict(X_test)
    
    matching_scores = svm.predict_proba(X_test)  
    
    genuine_scores = []
    impostor_scores = []
    
    for i in range(len(y_test)):
        if y_test[i] == y_predict[i]:
            gen = max(matching_scores[i])
            genuine_scores.append(gen)
            for j in range(len(matching_scores[i])):
                if matching_scores[i][j] != gen:
                    impostor_scores.append(matching_scores[i][j])
        else:
            impostor_scores.extend(matching_scores[i])    
            
    num_correct = 0
    num_incorrect = 0
    
    for i in range(len(y_test)):
        if y_test[i] == y_predict[i]:
            num_correct += 1
        else:
            num_incorrect += 1
            
            
    return num_correct,num_incorrect, genuine_scores, impostor_scores, matching_scores
            
    
dark_directory = "/Users/pallavilohar/OneDrive - University of South Florida/DarkImages"  
occlusion_directory = "/Users/pallavilohar/OneDrive - University of South Florida/Occlusion"  
pose_directory = "/Users/pallavilohar/OneDrive - University of South Florida/FacePOse"

image_directory = [dark_directory, occlusion_directory, pose_directory ]
plot_titles = ['DARK','OCCLUSION','POSE']

for a in range(0,len(image_directory)):

    X_train, X_test , y_train, y_test = extract_image_vectors(image_directory[a],a)
    
    second_dimension_train = X_train.shape[1]*X_train.shape[2]
    X_train = np.reshape(X_train, (X_train.shape[0],second_dimension_train))
    
    second_dimension_test = X_test.shape[1]*X_test.shape[2]
    X_test = np.reshape(X_test, (X_test.shape[0],second_dimension_test))
    
    templateFeatures_X_LBP = LBP(X_train)
    queryFeatures_X_LBP = LBP(X_test)    
    
    x_train_eigenfaces, X_train_PCA, mean_x_train = PCA(X_train)
    
    X_test_PCA = X_test.transpose()
    
    for col in range(X_test_PCA.shape[1]):
        X_test_PCA[:,col] = X_test_PCA[:,col] - mean_x_train

    
    templateFeatures_X_PCA, queryFeatures_X_PCA = get_features(X_train_PCA, X_test_PCA, x_train_eigenfaces, len(y_test),30)
    

    
    templateFeatures = np.concatenate((templateFeatures_X_PCA, templateFeatures_X_LBP), axis=1)
    queryFeatures = np.concatenate((queryFeatures_X_PCA, queryFeatures_X_LBP), axis=1)
    
    gen_scores = []
    imp_scores = []
    num_correct = 0
    num_incorrect = 0
    
    num_correct,num_incorrect, gen_scores, imp_scores, match = matching(templateFeatures, queryFeatures, y_train, y_test)
    print("Num correct = %d, Num incorrect = %d, Accuracy = %0.2f" 
      % (num_correct, num_incorrect, num_correct/(num_correct+num_incorrect)))   

                    
    num_threshold = 500
    
    far, frr, tpr = compute_rates(gen_scores, imp_scores,num_threshold)     
    plot_title = plot_titles[a]
    plot_scoreDist(gen_scores, imp_scores, plot_title)
    plot_roc(far, tpr, plot_title)
    plot_det(far, frr, plot_title)