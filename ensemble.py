# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 12:05:02 2014

@author: fayyaz
"""
import numpy as np
from baseClassifier import Classifier
class classifierEnsemble(Classifier):
    def __init__(self,classifierTemplate,N=0):        
        if not isinstance(classifierTemplate,self.__class__):
            self.classifiers=[classifierTemplate.__class__(classifierTemplate) for i in range(N)]
            self.classifierTemplate=classifierTemplate
        else:
            self.classifiers=classifierTemplate.classifiers
            self.classifierTemplate=classifierTemplate.classifierTemplate
    def train(self,F,Xn=[]):
        dataEnsemble=False
        if type(Xn)==type([]) and type(Xn[0])==type([]):
            assert len(Xn)==len(self.classifiers)
            dataEnsemble=True        
        for i in range(len(self.classifiers)):
            #print "Training Classifier",i
            
            if dataEnsemble:
                self.classifiers[i].train(F,Xn[i])        
            else:
                self.classifiers[i].train(F,Xn)     
    def testInstances(self,x):        
        s=[]
        for i in range(len(self.classifiers)):            
            s.append(self.classifiers[i].testInstances(x))
        return np.mean(s,axis=0)    
    def toString(self):
        import json
        return json.dumps([c.toString() for c in self.classifiers])
    def fromString(self,S):
        import json
        self.classifiers=[]
        for s in json.loads(S):
            c=self.classifierTemplate.__class__(self.classifierTemplate)
            c.fromString(s)
            self.classifiers.append(c)
                
        