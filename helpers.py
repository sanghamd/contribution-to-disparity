##Helper File containing additional functions for main file##
#  Sanghamitra Dutta
# Requires installation of dit package
# There is a variable name change: s: Gender, x: Output, y: Feature

import time
import numpy as np
import random
from matplotlib import rc
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white", palette="muted", color_codes=True, context="talk")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score

import dit
import time


font = {'weight' : 'bold'}

rc('font', **font)

rc('legend', fontsize=28)
#rc('xtick', labelsize=28)


EPS = np.finfo(float).eps


#Function to Print Histograms
def plot_distributions(y, z,fname=None):
    fig, axes = plt.subplots(figsize=(10, 4),sharey=True)
    legend={'gender': ['Z=0','Z=1']}
    attr='gender'
    for attr_val in [0, 1]:
        ax = sns.distplot(y[z == attr_val], hist=False,kde_kws={'shade': True,},label='{}'.format(legend[attr][attr_val]),ax=axes)
    ax.set_xlim(0,1)
    ax.set_ylim(0,8)
    ax.set_yticks([])
    fig.tight_layout()
    plt.savefig(fname, bbox_inches='tight')
    return fig


# Computing Unique Information using dit (solves an optimization)
def create_dist(a,b,c,D_b,D_c):
    S=a
    X=b
    Y=c
    s_len=2  #Gender
    
    x_len=len(D_b)+1  #FoM quantized
    y_len=len(D_c)+1  #Feature quantized
    
    mydict={}
    for s in range(s_len):
        for x in range(x_len):
            for y in range(y_len):
                s1=chr(65+s)
                x1=chr(65+x)
                y1=chr(65+y)
                mydict["{}{}{}".format(s1,x1,y1)]=0

    n=len(a)
    for i in range(n):
        s=chr(65+a[i])
        
        
        if x_len==4:
        
            if b[i]<D_b[0]:
                p=0
            elif b[i]<D_b[1]:
                p=1
            elif b[i]<D_b[2]:
                p=2
            else:
                p=3
            x=chr(65+p)
    
        if x_len==2:
            if b[i]<D_b[0]:
                p=0
            else:
                p=1
            x=chr(65+p)
        
        
        if c[i]<D_c[0]:
            p=0
        elif c[i]<D_c[1]:
            p=1
        elif c[i]<D_c[2]:
            p=2
        else:
            p=3
        
        y=chr(65+p)
        mydict["{}{}{}".format(s,x,y)]=mydict["{}{}{}".format(s,x,y)]+1
    
    
    
    A=[]
    B=[]
    for key, value in mydict.items():
        if value>0:
            A.append(key)
            B.append(value)


    B=B/np.sum(B)

    d=dit.Distribution(A,B)
    return d



def PID(a,b,c,D_b,D_c):
    d=create_dist(a,b,c,D_b,D_c)
    d.set_rv_names('SXY')
    dit_pid = dit.pid.PID_BROJA(d, ['X', 'Y'], 'S')
    Uni = dit_pid.get_partial((('X', ), ))
    Red = dit_pid.get_partial((('X', ), ('Y', )))



    MI = dit.shannon.mutual_information(d, 'S', 'X')
    CMI= dit.shannon.mutual_information(d, 'S', 'XY')-dit.shannon.mutual_information(d, 'S', 'Y')
    return Uni,Red,MI,CMI



# Computing Unique Information using dit (solves an optimization)
def create_dist1(a,b,c):
    S=a
    X=b
    Y=c
    
    
    s_unique=np.unique(S)
    x_unique=np.unique(X)
    y_unique=np.unique(Y)
    
    s_len=len(s_unique)  #Gender Z
    x_len=len(x_unique)  #Output Y
    y_len=len(y_unique)  #Feature (Notice the change in variable names)
    
    mydict={}
    for i in range(s_len):
        for j in range(x_len):
            for k in range(y_len):
                s1=chr(65+i)
                x1=chr(65+j)
                y1=chr(65+k)
                mydict["{}{}{}".format(s1,x1,y1)]=0

    n=len(a)
    for i in range(n):
        s=chr(65+np.where(s_unique==S[i])[0][0])
        x=chr(65+np.where(x_unique==X[i])[0][0])
        y=chr(65+np.where(y_unique==Y[i])[0][0])
        mydict["{}{}{}".format(s,x,y)]=mydict["{}{}{}".format(s,x,y)]+1

    A=[]
    B=[]
    for key, value in mydict.items():
        if value>0:
            A.append(key)
            B.append(value)

    B=B/np.sum(B)
    d=dit.Distribution(A,B)
    return d

def create_dist2(a,b,c):
    S=a
    X=b
    Y=10*c[:,1]+c[:,0]
    
    
    s_unique=np.unique(S)
    x_unique=np.unique(X)
    y_unique=np.unique(Y)
    
    s_len=len(s_unique)  #Gender Z
    x_len=len(x_unique)  #Output Y
    y_len=len(y_unique)  #Feature (Notice the change in variable names)
    
    mydict={}
    for i in range(s_len):
        for j in range(x_len):
            for k in range(y_len):
                s1=chr(65+i)
                x1=chr(65+j)
                y1=chr(65+k)
                mydict["{}{}{}".format(s1,x1,y1)]=0

    n=len(a)
    for i in range(n):
        s=chr(65+np.where(s_unique==S[i])[0][0])
        x=chr(65+np.where(x_unique==X[i])[0][0])
        y=chr(65+np.where(y_unique==Y[i])[0][0])
        mydict["{}{}{}".format(s,x,y)]=mydict["{}{}{}".format(s,x,y)]+1
    
    A=[]
    B=[]
    for key, value in mydict.items():
        if value>0:
            A.append(key)
            B.append(value)

    B=B/np.sum(B)
    d=dit.Distribution(A,B)
    return d


def create_dist3(a,b,c):
    S=a
    X=b
    Y=100*c[:,2]+10*c[:,1]+c[:,0]
    
    
    s_unique=np.unique(S)
    x_unique=np.unique(X)
    y_unique=np.unique(Y)
    
    s_len=len(s_unique)  #Gender Z
    x_len=len(x_unique)  #Output Y
    y_len=len(y_unique)  #Feature (Notice the change in variable names)
    
    mydict={}
    for i in range(s_len):
        for j in range(x_len):
            for k in range(y_len):
                s1=chr(65+i)
                x1=chr(65+j)
                y1=chr(65+k)
                mydict["{}{}{}".format(s1,x1,y1)]=0

    n=len(a)
    for i in range(n):
        s=chr(65+np.where(s_unique==S[i])[0][0])
        x=chr(65+np.where(x_unique==X[i])[0][0])
        y=chr(65+np.where(y_unique==Y[i])[0][0])
        mydict["{}{}{}".format(s,x,y)]=mydict["{}{}{}".format(s,x,y)]+1
    
    A=[]
    B=[]
    for key, value in mydict.items():
        if value>0:
            A.append(key)
            B.append(value)

    B=B/np.sum(B)
    d=dit.Distribution(A,B)
    return d



def PID1(a,b,c):
    d=create_dist1(a,b,c)
    d.set_rv_names('SXY')
    dit_pid = dit.pid.PID_BROJA(d, ['X', 'Y'], 'S')
    Uni = dit_pid.get_partial((('X', ), ))
    Red = dit_pid.get_partial((('X', ), ('Y', )))
    
    
    MI = dit.shannon.mutual_information(d, 'S', 'X')
    return Uni,Red,MI

def PID2(a,b,c):
    d=create_dist2(a,b,c)
    d.set_rv_names('SXY')
    dit_pid = dit.pid.PID_BROJA(d, ['X', 'Y'], 'S')
    Uni = dit_pid.get_partial((('X', ), ))
    Red = dit_pid.get_partial((('X', ), ('Y', )))
    
    
    MI = dit.shannon.mutual_information(d, 'S', 'X')
    return Uni,Red,MI


def PID3(a,b,c):
    d=create_dist3(a,b,c)
    d.set_rv_names('SXY')
    dit_pid = dit.pid.PID_BROJA(d, ['X', 'Y'], 'S')
    Uni = dit_pid.get_partial((('X', ), ))
    Red = dit_pid.get_partial((('X', ), ('Y', )))
    
    
    MI = dit.shannon.mutual_information(d, 'S', 'X')
    return Uni,Red,MI
