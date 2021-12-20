#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 12:58:49 2021

@author: iiap
"""
import numpy as np
import pandas as pd
import scipy.constants as const
import sdds

##### natural constants #####
c=299792458        # speed of light
e_charge=1.6021766208e-19 # electron charge
e_m=9.10938356e-31      # electron mass in eV/c^2
Z0=376.73          # impedance of free space in Ohm
epsilon_0=8.854e-12 # vacuum permittivity
e_E=1.492e9*e_charge   # electron energy in J
e_gamma=e_E/e_m/c**2 # Lorentz factor

def compl(x):
    return(-complex(0,2*np.pi*x))
f=np.vectorize(compl)

def modify_lattice(file,Bu=None,P0=None,w0=None):
    f=open(file,'r')
    txt=f.read()
  #  f.close()
    if Bu != None:
        i=txt.find('Bu=')
        j=txt.find(',',i)
        old=txt[i:j]
        txt=txt.replace(txt[i:j],"Bu="+str(Bu))
        print("changed from "+old+" to "+txt[i:j])
    if P0 != None:
        i=txt.find('laser_peak_power=',0)
        j=txt.find(',',i)
        old=txt[i:j]
        txt=txt.replace(txt[i:j],"laser_peak_power="+str(P0))
        print("changed from "+old+" to "+txt[i:j]) 
    if w0 != None:
        i=txt.find('laser_w0=',0)
        j=txt.find(',',i)
        old=txt[i:j]
        txt=txt.replace(txt[i:j],"laser_w0="+str(w0))
        print("changed from "+old+" to "+txt[i:j])         
    
    f=open(file,"w")
    f.write(txt)
    f.close()


def define_bunch(Test,N=1e4):
    slicelength=50e-6 # length of simulated bunch slice in m
    N_e=int(N) # number of electrons
    
    ##### electron parameter #####
    e_E=1.492e9*e_charge   # electron energy in J
    energyspread= 7e-4 
       
    alphaX=8.811383e-01 #1.8348
    alphaY=8.972460e-01 #0.1999
    betaX=13.546
    betaY=13.401
    emitX= 1.6e-8
    emitY= 1.6e-9   
    Dx= 0.0894
    Dxprime= -4.3065e-9 
    
    if(Test):
        slicelength=2e-6 
        N_e=int(1e3)
        energyspread= 0
        emitX= 0
        emitY= 0  
        Dx= 0
        Dxprime= 0
        
    CS_inv_x=np.abs(np.random.normal(loc=0,scale=emitX*np.sqrt(2*np.pi),size=N_e))
    CS_inv_y=np.abs(np.random.normal(loc=0,scale=emitY*np.sqrt(2*np.pi),size=N_e))
    phase_x=np.random.rand(N_e)*2*np.pi
    phase_y=np.random.rand(N_e)*2*np.pi
    
    # generate random electron parameters according to beam parameters
    elec=np.zeros((6,N_e))
    elec[4]=(np.random.rand(1,N_e)-0.5)*slicelength#/c   
    elec[5]=np.random.normal(loc=e_E,scale=e_E*energyspread,size=N_e)#/e_m/c**2
    elec[0]=(np.sqrt(CS_inv_x*betaX)*np.cos(phase_x))+((elec[5,:]-e_E)/e_E)*Dx
    elec[1]=-(np.sqrt(CS_inv_x/betaX)*(alphaX*np.cos(phase_x)+np.sin(phase_x)))+((elec[5,:]-e_E)/e_E)*Dxprime
    elec[2]=(np.sqrt(CS_inv_y*betaY)*np.cos(phase_y))
    elec[3]=-(np.sqrt(CS_inv_y/betaY)*(alphaY*np.cos(phase_y)+np.sin(phase_y)))
    
    #changing to the units used by Elegant
    elec[5]=elec[5]/e_m/c**2
    elec[4]=elec[4]/c
    
    
    data_in=sdds.SDDS(0)

    #column=['x', 'xp', 'y', 'yp', 't', 'p']
    #for name in column:
    #   data_in.defineSimpleColumn(name,1)
    
    data_in.load("bunch_in.sdds")
    for i in range(6):
        data_in.columnData[i][0]=list(elec[i])
    if(Test):
    	data_in.save("bunch_test.sdds")
    else:
    	data_in.save("bunch_in.sdds")
    
def track_chicane(bunch_in,R56,R51=0,R52=0,isr=True):
    RR=np.array([[ 9.0952120e-01,  3.1357837e+00,  0.0000000e+00,  0.0000000e+00,
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
    RR[4,0],RR[4,1]=R51,R52    
    RR[4,5]=R56
    bunch_end= np.matmul(RR,bunch_in)
    elec2=np.copy(bunch_end.transpose((2,1,0))[0])
    elec2[4]=elec2[4]/c
    if isr:
        L=0.37
        d=0.10      
        alpha=np.sqrt(R56/((4*L/3)+2*d))
        #B=(e_gamma*e_beta*e_m*c*np.sin(alpha))/(L*e_charge)
        rho=L/alpha
        sigE=4*np.sqrt((55*const.alpha*(((const.h/(2*np.pi))*const.c)**2)*(e_gamma**7)*L)/((2*24*(3**0.5))*(rho)**3))
        print(sigE/e_E)
        SR_dE=np.random.normal(loc=0,scale=sigE/e_E,size=len(elec2[5]))
        elec2[5]+=SR_dE
    elec2[5]=(elec2[5]+1)*e_E/e_m/c**2
    return(elec2)


