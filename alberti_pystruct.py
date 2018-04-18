# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 20:57:39 2014

@author: root
"""

import itertools

import numpy as np
from scipy import sparse

from sklearn.metrics import hamming_loss
from sklearn.datasets import fetch_mldata
from sklearn.metrics import mutual_info_score
try:
    from sklearn.utils import minimum_spanning_tree
except ImportError:
    raise ImportError("Please install a recent version of scikit-learn or"
                      "scipy to build minimum spanning trees.")
from pystruct.learners import OneSlackSSVM
from pystruct.models import MultiLabelClf
from pystruct.datasets import load_scene
from readAlberti import loadAlberti, getWkSpectrum
import pdb
def chow_liu_tree(y_):
    # compute mutual information using sklearn
    n_labels = y_.shape[1]
    mi = np.zeros((n_labels, n_labels))
    for i in xrange(n_labels):
        for j in xrange(n_labels):
            mi[i, j] = mutual_info_score(y_[:, i], y_[:, j])
    print mi
    mst = minimum_spanning_tree(sparse.csr_matrix(-mi))
    print mst
    edges = np.vstack(mst.nonzero()).T
    edges.sort(axis=1)
    return edges
    
if __name__=="__main__":
    D=loadAlberti()
    S=[]
    L=[]
    for l,s in D.values():
        S.append(s)        
        L.append(list(np.asarray(np.array(l)>0,np.int)))
    FS=getWkSpectrum(S,k=1,hws=29)
    X=np.hstack(FS)
    X=(X/np.sqrt(np.sum(X**2,axis=0))).T
    Y=np.zeros((X.shape[0],len(L[0])),dtype=np.int)
    P=np.zeros(X.shape[0])
    i0=0
    for i,fs in enumerate(FS):
        i1=i0+fs.shape[1]
        P[i0:i1]=i
        Y[i0:i1]=L[i]
        i0=i1
    prot_ids=np.unique(P)
    n_labels = Y.shape[1]
    full = np.vstack([x for x in itertools.combinations(range(n_labels), 2)])
    #tree = chow_liu_tree(y_train)
    
    full_model = MultiLabelClf( inference_method='qpbo')#edges=full,
    full_ssvm = OneSlackSSVM(full_model, inference_cache=50, C=1.0, tol=0.01)
    Y_pred=np.zeros(Y.shape)
    for p in prot_ids:
        idx=(P!=p)
        X_train=X[idx,:]
        y_train=Y[idx,:]
        X_test=X[~idx,:]
        y_test=Y[~idx,:]
        print p
        if np.mean(y_test)==0 or np.mean(y_test)==4:    
            tree = chow_liu_tree(y_train)
            print("fitting full model...")
            tree_model = MultiLabelClf(edges=tree)
            tree_ssvm = OneSlackSSVM(tree_model, inference_cache=50, C=.1, tol=0.01)
            tree_ssvm.fit(X_train, y_train)
            y_pred=np.vstack(tree_ssvm.predict(X_test))
            Y_pred[~idx,:]=y_pred
            
            print("Training loss full model: %f" % hamming_loss(y_train, np.vstack(tree_ssvm.predict(X_train))))
            print("Test loss full model: %f"  % hamming_loss(y_test, y_pred))
        