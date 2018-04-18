# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 12:04:23 2014
Functions for data handling
@author: fayyaz
"""
import os
def getFileParts(fname):
    "Returns the parts of a file"
    (path, name) = os.path.split(fname)
    n=os.path.splitext(name)[0]
    ext=os.path.splitext(name)[1]
    return (path,n,ext)
def parsePLAACFile(ifile):    
    #i='LLR';ps=[float(S[k][i]) if not np.isnan(float(S[k][i])) else -1.0 for k in S if '#' in k];ns=[float(S[k][i]) if not np.isnan(float(S[k][i])) else -1.0 for k in S if '#' not in k];import yard;r=yard.ROCCurve(zip(ps+ns,[1]*len(ps)+[-1]*len(ns)));fig=r.get_figure();fig.suptitle(r.auc());r.show();plt.show()
    S={}
    with open(ifile,'r') as fin:
        for i,l in enumerate(fin):
            ls=l.split('\t')
            if i==0:
                hdr=ls
            else:
                S[ls[0]]=dict(zip(hdr[1:],ls[1:]))
    return hdr,S
    
def loadFASTA(ifile):
    """
    Read a fasta file and return a list of tuple (head,seq)
    """
    from Bio import SeqIO    
    for record in SeqIO.parse(open(ifile, "rU"), "fasta"):
        yield (record.description,str(record.seq))
    
def loadAnnotatedPrions(ifile):    
    """
    Given an input fasta file with annotated prion domains this function returns
    a tuple (S,A) where the lists S and A contain the sequence and annotated 
    prion domain for each sequence in the file
    """
    from Bio import SeqIO
    S=[]
    with open(ifile, "rU") as fhandle:
        for record in SeqIO.parse(fhandle, "fasta"):    
            seq=record.seq.tostring()
            try:
                idx=[int(i)-1 for i in record.description.split('#')[1].split('-')]
            except IndexError:
                idx=[0,len(seq)-1]                
            
            S.append((seq,idx))
    return tuple(map(list,[a for a in zip(*S)]))
def writePAPAFile(ofile,S,sort=True):
    """
    Write a papa format file
    S: list 
        S[i][0] is the header
        S[i][1] is a tuple of predictions
            S[i][1][0] is the max. score for a protein
            S[i][1][1] is the position (starting at 0) of the max. score
            S[i][1][2] (optional) is the list of the scores
    The output file is sorted in descending order of the prediction score
    The output file format is:
        line> header,max_score,max_position,[position wise scores]
    """
    if sort:
        S.sort(key=lambda tup: -tup[1][0])
    with open(ofile,'w') as fout:
        for (hdr,vv) in S:            
            if len(vv)>2:
                sstr=','+ str(vv[2])
            else:
                sstr=''
            fout.write(str(hdr) + ',' + str(vv[0]) + ',' + str(vv[1]) +sstr+'\n')   
def readPAPAFile(fname,asTuple=False):
    """
    Reads a papa format file fname
    if asTuple is false (default): only the max_scores are read in
    otherwise the list of tuples S is returned on the same format as in writePAPAFile
    """
    import re
    S=[]        
    with open(fname,'r') as f:
        for l in f:
            if l[0]=='#': #ignore comments
                continue
            mm=re.search(r"\,([-]*[0-9]+\.[0-9]+)\,", l)
            if mm is not None:
                try:
                    v=float(mm.group(1))
                    if asTuple:                        
                        ls=l[mm.start():].split('[')
                        xhdr=ls[0].split(',')
                        hdr=l[:mm.start()]#xhdr[0]
                        pos=int(xhdr[2])
                        vals=[]
                        if len(ls)>1:
                            try:
                                vals=[float(i) for i in ls[1].split(']')[0].split(',')]
                            except ValueError: #if any of them is not float, just give up!
                                pass
                        S.append((hdr,(v,pos,vals)))
                    else:
                        S.append(v)                    
                except Exception as exc:
                    print "ERROR (ignored line):",exc,l                       
                    continue    
    return S        

if __name__=='__main__':
    import sys
    if len(sys.argv)!=2:
        print "Usage: data.py file_name"
        sys.exit(1)
    ifile=sys.argv[1]
    S=readPAPAFile(ifile,asTuple=True)