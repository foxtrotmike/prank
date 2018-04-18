import sys,os
"""""""""
Python modules and matplotlib
"""""""""
#Only for department machines
pymodulespath='/s/chopin/b/grad/minhas/lib64/python/'
pythonpath='/s/chopin/b/grad/minhas/lib/python/'
if os.path.exists(pythonpath):
    sys.path.append(pymodulespath)
    import matplotlib
    matplotlib.use('Agg') #Must be uncommented for running on department machines    
    if "PYTHONPATH" not in os.environ: #required for yard on department machines
        os.environ["PYTHONPATH"]=pythonpath

"""""""""
MSMS, STRIDE, SPINEX
"""""""""
msmspaths=['/s/chopin/c/proj/protfun/arch/x86_64/msms_i86Linux2_2.6.1','/media/sf_Desktop/pairpred/Tools/On_Server/msms_i86Linux2_2.6.1'] #contains stride and msms implementation
for p in msmspaths:
    if os.path.exists(p):
        msmspath=p
        spxpath=msmspath+'/spineXpublic/' #spinex path
        if msmspath not in  os.environ["PATH"]:
            print 'Adding '+msmspath+' to system path'
            os.environ["PATH"]+=os.pathsep+msmspath
        if spxpath not in  os.environ["PATH"]:
            print 'Adding '+spxpath+' to system path'
            os.environ["PATH"]+=os.pathsep+spxpath   
        break
"""""""""
PSAIA
"""""""""
psaiapaths=['/s/chopin/c/proj/protfun/arch/x86_64/psaia/PSAIA-1.0','/home/fayyaz/PSAIA-1.0/']
for p in psaiapaths:
    if os.path.exists(p):
        PSAIA_PATH=p
        if PSAIA_PATH not in  os.environ["PATH"]:
            print 'Adding '+PSAIA_PATH+' to system path'
            os.environ["PATH"]+=os.pathsep+PSAIA_PATH
        break
"""""""""
PSI BLAST
"""""""""
blastpaths=['/s/chopin/c/proj/protfun/arch/x86_64/ncbi-blast-2.2.27+-x64-linux/ncbi-blast-2.2.27+/bin/']    
for p in blastpaths:
    if os.path.exists(p):
        BLAST_DIR=p#'/s/chopin/c/proj/protfun/arch/x86_64/blast+/bin/'
        if BLAST_DIR not in  os.environ["PATH"]:
            print 'Adding '+BLAST_DIR+' to system path'
            os.environ["PATH"]+=os.pathsep+BLAST_DIR    
        break

"""""""""
PROBIS
"""""""""    
probispaths=['/s/chopin/b/grad/minhas/Desktop/probis-2.4.2','/home/fayyaz/Documents/probis-2.4.2']
for p in probispaths:
    if os.path.exists(p):
        PROBIS_PATH=p#'/home/fayyaz/Documents/probis-2.4.2'#
        if PROBIS_PATH not in  os.environ["PATH"]:
            print 'Adding '+PROBIS_PATH+' to system path'
            os.environ["PATH"]+=os.pathsep+PROBIS_PATH    
        break
