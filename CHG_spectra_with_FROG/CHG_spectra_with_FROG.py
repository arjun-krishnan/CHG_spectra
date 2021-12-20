# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 21:04:26 2021

@author: arjun
"""

import sys,os

#Changing the working directory to the source directory
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numba import jit
from scipy import special,stats,interpolate
from scipy.constants import *
#from copy import deepcopy
from tqdm import tqdm


class P_array:
    def __init__(self,n=100000,sigmae=0.0007,sigmax=4.69e-4,sigmaxp=4.58e-5,wl_L=800e-9):
        self.n_points= n
        self.sig_e= sigmae
        self.sig_x= sigmax
        self.sig_xp= sigmaxp
        self.lambda_L= wl_L
        N_e= n
        e_E=1.492e9*e   # electron energy in J
        energyspread= 7e-4 
        slicelength=8e-5 #8e-6 
           
        alphaX=8.811383e-01 #1.8348
        alphaY=8.972460e-01 #0.1999
        betaX=13.546
        betaY=13.401
        emitX= 1.6e-8
        emitY= 1.6e-9   
        Dx= 0.0894
        Dxprime= -4.3065e-9 
        CS_inv_x=np.abs(np.random.normal(loc=0,scale=emitX*np.sqrt(2*np.pi),size=N_e))
        CS_inv_y=np.abs(np.random.normal(loc=0,scale=emitY*np.sqrt(2*np.pi),size=N_e))
        phase_x=np.random.rand(N_e)*2*np.pi
        phase_y=np.random.rand(N_e)*2*np.pi
        
        # generate random electron parameters according to beam parameters
        elec0=np.zeros((6,N_e))
        elec0[4]=(np.random.rand(1,N_e)-0.5)*slicelength#/c   
        elec0[5]=np.random.normal(loc=0,scale=energyspread,size=N_e)#/e_m/c**2
        elec0[0]=(np.sqrt(CS_inv_x*betaX)*np.cos(phase_x))+elec0[5,:]*Dx
        elec0[1]=-(np.sqrt(CS_inv_x/betaX)*(alphaX*np.cos(phase_x)+np.sin(phase_x)))+elec0[5,:]*Dxprime
        elec0[2]=(np.sqrt(CS_inv_y*betaY)*np.cos(phase_y))
        elec0[3]=-(np.sqrt(CS_inv_y/betaY)*(alphaY*np.cos(phase_y)+np.sin(phase_y)))
        
        elec=np.asarray([[elec0[0]],[elec0[1]],[elec0[2]],[elec0[3]],[elec0[4]],[elec0[5]]])
        self.p_init= elec.transpose((2,0,1))
        

    def modulate(self,p_array_in,A,wl,w0=1000,FROG=None,phase=0):
        zz = np.array([X[4,0] for X in p_array_in])
        xx = np.array([X[0,0] for X in p_array_in])
        xp = np.array([X[1,0] for X in p_array_in])
        EE = np.array([X[5,0] for X in p_array_in])
        tt=np.linspace(min(zz),max(zz),self.n_points)
        #theta=np.linspace(0,2*np.pi*(max(zz)-min(zz))/wl,self.n_points)
        
        if FROG is None:
            xxx=stats.norm(0,17e-15*c)
            a=xxx.pdf(tt)
            a=a/max(a)
            AA=a*A
            dE=AA*np.sin((2*np.pi*tt/wl)+(phase*tt**2))
            f=interpolate.interp1d(tt,dE)
        else:
            names=['time','Amp','phase','4','5']
            df=pd.read_csv(FROG,delim_whitespace=True,names=names)
            time=np.array(df['time'])*1e-15
            AA= np.array(df['Amp'])*A
            phase=np.array(df['phase'])
            fA=interpolate.interp1d(time, AA)
            fp=interpolate.interp1d(time, phase)
            time=np.linspace(min(time),max(time),100000)
            AA,phase=fA(time),fp(time)
            dE=AA*np.sin((2*np.pi*time*c/wl)+phase)
            f=interpolate.interp1d(time*c, dE)
            
        
        kk=f(zz)*np.exp(-xx**2/w0**2) #f_L_size(xx)
        ze = np.zeros(len(zz))
        MM = np.asarray([[xx],[xp],[ze],[ze],[zz],[EE+kk]])
        p_mod= MM.transpose((2,0,1))
            
        return(p_mod)
    
    def chicane(self,p_array_in,R56):
        RR1=np.copy(RR)
        RR1[4][0]=0
        RR1[4][1]=0
        RR1[4][5]=R56
        p_end= np.matmul(RR1,p_array_in)
        return(p_end)
    
@jit(nopython=True,parallel=True)
def calc_bn(tau0,wls):
    bn=[]
    for wl in wls:
        z=np.sum(np.exp(-1j*2*np.pi*(tau0/wl)))
        bn.append(abs(z)/len(tau0))
    return(np.array(bn))

#RM=pd.read_csv("../../data/TM.txt", usecols=range(1,7))
#RR=np.array(RM)

RR= np.array([[ 9.0952120e-01,  3.1357837e+00,  0.0000000e+00,  0.0000000e+00,
         0.0000000e+00,  7.2400000e-05],
       [-5.5096400e-02,  9.0952210e-01,  0.0000000e+00,  0.0000000e+00,
         0.0000000e+00,  4.4000000e-05],
       [ 0.0000000e+00,  0.0000000e+00,  1.0000000e+00,  3.2707655e+00,
         0.0000000e+00,  0.0000000e+00],
       [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00,
         0.0000000e+00,  0.0000000e+00],
       [-4.4000000e-05, -7.2300000e-05,  0.0000000e+00,  0.0000000e+00,
         1.0000000e+00,  1.5083000e-03],
       [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
         0.0000000e+00,  1.0000000e+00]])

fname=r"C:/Users/arjun/OneDrive - Technische Universität Dortmund/DELTA/CHG_Spectra_measurements/2021-11-19-FROG/27.795/1/2021-11-19_18-06-59_ETemporal.dat"

A=0.0042
R56= np.linspace(0,130e-6,131)
p_in= P_array(n=int(5e4),wl_L=800e-9,sigmae=0.0007)
p_mod1=p_in.modulate(p_in.p_init,A=A,wl=800e-9,FROG=fname)

wls=np.linspace(350e-9,450e-9,1001)
hmap=[]
for R1 in tqdm(R56):
    p_chic1=p_in.chicane(p_mod1,R56=R1)
    tau=np.array([X[4,0] for X in p_chic1])
    bn=calc_bn(tau,wls)
    #bn=bn/max(bn)
    hmap.append(bn)
plt.figure()
plt.contourf(wls,R56,hmap,levels=100)

'''
phi=np.linspace(-5*np.pi,5*np.pi,len(wl))
E=np.zeros(len(tt))
for i in range(len(wl)):
    E+=a[i]*np.sin((2*np.pi*tt/wl[i])+(phi[i]))
plt.plot(tt/c*1e15,E)
'''