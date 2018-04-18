# -*- coding: utf-8 -*-
"""
Created on Tue May 14 11:54:37 2013
Saves and loads from compressed pickled files
Also, no need of opening a file first and closing it later
@author: root
"""

import gzip, cPickle,os

def getFileParts(fname):
    "Returns the parts of a file"
    (path, name) = os.path.split(fname)
    n=os.path.splitext(name)[0]
    ext=os.path.splitext(name)[1]
    return (path,n,ext)
def dump(fname, obj,gz=True):
    if gz:
        fhandle=gzip.open(fname, "wb", compresslevel=3)
    else:
        fhandle=open(fname, "wb")
    cPickle.dump(obj=obj, file=fhandle, protocol=-1)
    fhandle.close()
def load(fname):
    try:
        R=cPickle.load(gzip.open(fname, "rb")) 
        return R
    except IOError as e:
        #print e
        #print "Trying regular cPickle"
        return cPickle.load(open(fname, "rb")) 
def mkl2pkl(ifname,ofname=None):    
    if ofname is None:
        of=getFileParts(ifname)
        ofname=os.path.join(of[0],of[1]+'.pkl')
    dump(ofname,load(ifname),gzip=False)
    
