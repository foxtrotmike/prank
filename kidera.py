# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 21:18:24 2014

@author: fayyaz
"""
from Bio.PDB.Polypeptide import three_to_one
import numpy as np
import random
from copy import deepcopy
from matplotlib import pyplot as plt
def readKidera(ifile='kidera.txt'):
    """
    Read the file kdiera.txt containing the values
    """
    D={}
    with open(ifile,'r') as f:
        for l in f:
            ls=l.split()
            D[three_to_one(ls[0])]=[float(i) for i in ls[1:]]
    return D
def kiderify(s,D=readKidera()):
    """
    Convert sequence to kidera numeric values 
    """
    K=[]
    z=[0]*10
    for a in s:
        try:
            K.append(D[a])
        except Exception:
            K.append(z)            
    return np.array(K)

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
def avgdProperties(S,HWS=20):
    F=[]
    D=readKidera(ifile='kidera.txt')
    for s in S:
        K=kiderify(s,D)
        Kr=np.asarray([rolling_window(K[:,i],2*HWS+1) for i in range(K.shape[1])])
        #f=np.vstack((np.mean(Kr,axis=2),np.std(Kr,axis=2)))
        f=np.mean(Kr,axis=2)
        F.append(f)
    return F
def stdProperties(S,HWS=20):
    F=[]
    D=readKidera(ifile='kidera.txt')
    for s in S:
        K=kiderify(s,D)
        Kr=np.asarray([rolling_window(K[:,i],2*HWS+1) for i in range(K.shape[1])])        
        f=np.std(Kr,axis=2)
        F.append(f)
    return F    
def FFT(k,shuffle=False):
    k0=deepcopy(k)
    if shuffle:    
        random.shuffle(k0)
    f=np.fft.fftshift(np.fft.fft(k0))
    return f
if __name__=='__main__':
    s='MMNNNGNQVSNLSNALRQVNIGNRNSNTTTDQSNINFEFSTGVNNNNNNNSSSNNNNVQNNNSGRNGSQNNDNENNIKNTLEQHRQQQQAFSDMSHVEYSRITKFFQEQPLEGYTLFSHRSAPNGFKVAIVLSELGFHYNTIFLDFNLGEHRAPEFVSVNPNARVPALIDHGMDNLSIWESGAILLHLVNKYYKETGNPLLWSDDLADQSQINAWLFFQTSGHAPMIGQALHFRYFHSQKIASAVERYTDEVRRVYGVVEMALAERREALVMELDTENAAAYSAGTTPMSQSRFFDYPVWLVGDKLTIADLAFVPWNNVVDRIGINIKIEFPEVYKWTKHMMRRPAVIKALRGE*'    
    D=readKidera(ifile='kidera.txt')
    HWS=20
    K=kiderify(s,D)
    
    Krolled=[rolling_window(K[:,i],2*HWS+1) for i in range(K.shape[1])]
    """
    k=Krolled[0][100,:]
    F=np.array([FFT(k,shuffle=True) for _ in range(100000)])
    Freal=np.real(F)**2
    Fimag=np.imag(F)**2
    mu_real=np.mean(Freal,axis=0)
    mu_imag=np.mean(Fimag,axis=0)
    std_real=np.std(Freal,axis=0)+1e-10
    std_imag=np.std(Fimag,axis=0)+1e-10
    kF=FFT(k)
    kreal=np.real(kF)**2
    kimag=np.imag(kF)**2
    kalpha=(kreal-mu_real)/std_real
    kbeta=(kimag-mu_imag)/std_imag
    plt.plot(kalpha,'r');plt.plot(kbeta,'b');plt.plot(std_real,'k');
    plt.show()
    """