# -*- coding: utf-8 -*-

import math
import numpy as np
import matplotlib.pyplot as plt

############ functions #############################################################

def dprime(gen_scores, imp_scores):
    x = math.sqrt(2) * abs(np.mean(gen_scores) - np.mean(imp_scores))
    y = math.sqrt(np.var(gen_scores) + np.var(imp_scores))
    return x / y

def plot_scoreDist(gen_scores, imp_scores, plot_title):
    plt.figure()
    plt.hist(gen_scores, color = 'green', lw = 2, histtype = 'step', hatch = '//', label = 'Genuine Scores') 
    plt.hist(imp_scores, color = 'red', lw = 2, histtype = 'step', hatch = '\\', label = 'Impostor Scores')
    plt.xlim([-0.05, 1.05])
    plt.legend(loc='best')
    dp = dprime(gen_scores, imp_scores)
    plt.title(plot_title+ '\nD-prime= %.2f' % dp)
    plt.show()
    return

def get_EER(far, frr):
    eer = 0
    minimum_diff = float('inf')
    for i in range(len(far)):
        d = abs( far[i] - frr[i])
        if d < minimum_diff:
            minimum_diff = d
            eer = (far[i] + frr[i])/2

    return eer

def plot_det(far, frr, plot_title):
    eer = get_EER(far, frr)               
    plt.figure()
    plt.plot(far, frr, lw = 2)
    plt.plot([0,1], [0,1], lw = 1, color = 'black')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('FAR')
    plt.ylabel('FRR')
    plt.title(plot_title+ '\nEER= %.3f' % eer)
    plt.show()
    return

def plot_roc(far, tpr, plot_title):
    plt.figure()
    plt.plot(far, tpr, lw = 2)
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.05,1.05])
    plt.xlabel('FAR')
    plt.ylabel('TAR')
    plt.title(plot_title)
    plt.show()
    return

# Function to compute TPR, FAR, FRR
def compute_rates(gen_scores, imp_scores, num_thresholds):
    thresholds = np.linspace(0.0, 1.0, num=num_thresholds)
    far = []
    frr = []
    tpr = []
    
    for t in thresholds:
        tp, fp, tn, fn = 0, 0, 0, 0

        
        for g_s in gen_scores:
            if g_s >= t:
                tp += 1
            else:
                fn += 1                
        for i_s in imp_scores:
            if i_s >= t:
                fp += 1
            else:
                tn += 1                            
                    
        far.append(fp/(fp+tn))
        frr.append(fn/(fn+tp))
        tpr.append(tp/(tp+fn))
        

    return far, frr, tpr