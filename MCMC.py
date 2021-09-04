#!/usr/bin/env python


import os
import math
import numpy as np
import random
import traceback

class Func:
    def __call__(self,x,a,b):
        y=a*x+b
        return y

class Likelihood:
    def __init__(self,par1,par2):
        self.a=par1
        self.b=par2
    def __call__(self,x,y,sigma):
        Func_instance=Func()
        Log_Likelihood=-(y-Func_instance(x,self.a,self.b))**2/(2.0*sigma**2)
        return Log_Likelihood

##### Make Sample Data #####
a0,b0=2.0, 2.0 
xdata=np.zeros(100)
ydata=np.zeros(100)
sigmadata=np.zeros(100)
for i in range(100):
    tmp1=random.uniform(0,100)
    Make_Data=Func()
    xdata[i]=tmp1
    ydata[i]=Make_Data(tmp1,a0,b0)+np.random.normal(loc=0,scale=10)
    sigmadata[i]=abs(np.random.normal(loc=0,scale=1))

##### Initial Parameter Setup #####
a,b=0,0
FA = 0.0
cov0=[[0.1,0.0],[0.0,0.1]] #sigma**2 = 0.1
p=1
S0=np.random.multivariate_normal([0.0,0.0], p*cov0,1) #multivariate normal/Gaussian distribution
m0,X0=np.array([a0,b0]),np.array([a0,b0])
Xn=X0 + S0
pprev,mprev,covprev= p,m0,cov0
Pre_Sum_Log_Likelihood = -1e100
FinalLikelihood = 0

Nboot=100000 #Number of trials
para1=np.zeros(Nboot)
para2=np.zeros(Nboot)
OKarray=np.zeros(Nboot)
Larray=np.zeros(Nboot)

ok=0.0
i=1
while i < Nboot:
    mn = mprev+(10.0/(i+100.0))*(Xn-mprev)
    covn = covprev+(10.0/(i+100.0))*((Xn-mprev)*(Xn-mprev).transpose()-covprev)
    pn = pprev+(10.0/(i+100.0))*(FA-0.243)
    S0 = np.random.multivariate_normal([0.0,0.0], pn*covn,1)

    a_tmp=a+S0[0][0]
    b_tmp=b+S0[0][1]

    tmp=Likelihood(a_tmp,b_tmp) #Create Instance of Class "Likelihood"
    All_Log_Likelihood=0.0
    All_Log_Likelihood=[tmp(xdata[j],ydata[j],sigmadata[j]) for j in range(np.size(xdata))]
    Sum_Log_Likelihood=np.sum(All_Log_Likelihood)
    Log_Likelihood_Ratio=Sum_Log_Likelihood-Pre_Sum_Log_Likelihood

######################################
###### Add any conditions ############
######################################
    if Log_Likelihood_Ratio >= 0.0 :
        a=a_tmp
        b=b_tmp
        FA=1.0
        FinalLikelihood=Sum_Log_Likelihood
        Pre_Sum_Log_Likelihood=Sum_Log_Likelihood
    else:
        pp = np.random.rand()
        if Log_Likelihood_Ratio >= math.log(pp):
            a=a_tmp
            b=b_tmp
            FA=1.0
            FinalLikelihood=Sum_Log_Likelihood
            Pre_Sum_Log_Likelihood=Sum_Log_Likelihood
        else:
            ok +=1.0
            FinalLikelihood=Pre_Sum_Log_Likelihood
            FA = 0
    OK=1-ok/i

    pprev = pn
    mprev = mn
    covprev = covn

    Xn = np.array([a,b])
    #print(Log_Likelihood_Ratio)
    print(a,b,FinalLikelihood,math.log(np.random.rand()),sep=",")       
    para1[i]=a 
    para2[i]=b
    OKarray[i]=OK
    Larray[i]=FinalLikelihood 
    i += 1 


outputfile="./output/Likelihood.dat"
label="a,b,OK,LogLikelihood"
line=np.array([para1,para2,OKarray,Larray])
np.savetxt(outputfile, line.T, fmt='%0.8e',delimiter=',', newline='\n', header=label, comments="")
