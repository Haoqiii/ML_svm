# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 11:20:42 2020
@author: Haoqi
"""
from sklearn.datasets import make_blobs
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np

X,y = make_blobs(n_samples = 50,centers =2,random_state = 0,cluster_std = 0.6)
plt.scatter(X[:,0],X[:,1],c = y,cmap = 'rainbow')
# plt.xticks([])
# plt.yticks([])
def plot_svc_decision_function(model,ax = None):
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    x = np.linspace(xlim[0],xlim[1],30)
    y = np.linspace(ylim[0],ylim[1],30)
    
    Y,X = np.meshgrid(y,x)
    xy = np.vstack([X.ravel(),Y.ravel()]).T
    p = model.decision_function(xy).reshape(X.shape)
    
    ax.contour(X,Y,p,colors = 'k',levels = [-1,0,1],alphas = 0.5,linestyles = ['--','-','--'])
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

'''为非线性增加维度绘制3D
'''
r = np.exp(-(X**2).sum(1)) #sum line
rlim = np.linspace(min(r),max(r),30)
from mpl_toolkits import mplot3d 
def plot_3D(elev = 30,azim = 30,X = X,y = y):
    ax = plt.subplot(projection = "3d")
    ax.scatter3D(X[:,0],X[:,1],r,c = y,s = 50,cmap= 'rainbow')
    ax.view_init(elev,azim)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()

from ipywidgets import interact,fixed
interact(plot_3D,elev = [0,30],azip = (-180,180),X = fixed(X),y = fixed(y))
plt.show()










