# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 14:15:53 2020

@author: Haoqi
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import svm
from sklearn.datasets import make_circles,make_moons,make_blobs,make_classification

n_samples = 100

datasets = [
        make_moons(n_samples  = n_samples,noise = 0.2,random_state = 0),
        make_circles(n_samples = n_samples,noise = 0.2,factor = 0.5,random_state = 1),
        make_blobs(n_samples = n_samples,centers = 2,random_state = 5),
        make_classification(n_samples = n_samples,
                            n_features=2,
                            n_informative = 2,
                            n_redundant = 0,
                            random_state = 5)
        ]
Kernels = ["linear","poly","rbf","sigmoid"]
for X,Y in datasets:
    plt.figure(figsize = (5,4))
    plt.scatter(X[:,0],X[:,1],c = Y,s = 50,cmap = 'rainbow')
    
nrows = len(datasets)
ncols = len(Kernels)+1
fig,axes= plt.subplots(nrows,ncols,figsize = (20,16))# 4*4

for ds_cnt,(data_x,data_y) in enumerate(datasets):
    ax = axes[ds_cnt,0]
    if ds_cnt==0:
        ax.set_title('input data')
    ax.scatter(data_x[:,0],data_x[:,1],c = data_y,cmap = plt.cm.Paired,edgecolors = 'k')
    ax.set_xticks([])
    ax.set_yticks([])
    
    for est_idx,kernel in enumerate(Kernels):
        ax = axes[ds_cnt,est_idx+1]
        
        clf = svm.SVC(kernel = kernel,gamma=2).fit(data_x,data_y)
        score = clf.score(data_x,data_y)
        
        # orgin data
        ax.scatter(data_x[:,0],data_x[:,1],c = data_y
        ,zorder= 10
        ,cmap = plt.cm.Paired,edgecolors = 'k')
        
        # plot support vectors
        ax.scatter(clf.support_vectors_[:,0],clf.support_vectors_[:,1],s = 50,
                   facecolor = 'none',
                   zorder = 10,
                   edgecolors= 'k')
        x_min ,x_max = data_x[:,0].min() -0.5,data_x[:,0].max()+0.5
        y_min ,y_max = data_x[:,1].min() -0.5,data_x[:,1].max()+0.5
        
        XX,YY = np.mgrid[x_min:x_max:200j,y_min:y_max:200j]
        
        z = clf.decision_function(np.c_[XX.ravel() ,YY.ravel()]).reshape(XX.shape)
        
        ax.pcolormesh(XX,YY,z>0,cmap = plt.cm.Paired)
        ax.contour(XX,YY,z,colors = ['k','k','k'],
                      linestyles =['--','-','--'],
                      levels = [-1,0,1])
                      
        if ds_cnt == 0:
            ax.set_title(kernel)
        
        ax.text(0.95,0.06,('%.2f'%score).lstrip('0'),
                size = 15,
                bbox = dict(boxstyle = 'round',alpha = 0.8,facecolor = 'white'),
                transform = ax.transAxes,
                horizontalalignment = 'right'
                )
    
plt.tight_layout()
plt.show()
                
        
        
        
    
        
        


















