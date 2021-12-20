# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 17:00:03 2021

@author: NewUser
"""


import sys,os

#Changing the working directory to the source directory
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

import numpy as np
import matplotlib.pyplot as plt
import subprocess
from scipy import special
import scipy.constants as const
import pandas as pd
from time import time
import sdds
from lsrmdltr_setup import *
from numba import jit
from tqdm import tqdm

@jit(parallel=True)
def calc_bn(tau0,wl):
    bn=[]
    for i in range(len(wl)):
        z=np.sum(np.exp(-1j*2*np.pi*(tau0/wl[i])))
        bn.append(abs(z)/len(tau0))
    return(np.array(bn))
        
##### natural constants #####
c=299792458        # speed of light
e_charge=1.6021766208e-19 # electron charge
e_m=9.10938356e-31      # electron mass in eV/c^2
Z0=376.73          # impedance of free space in Ohm
epsilon_0=8.854e-12 # vacuum permittivity

e_E=1.492e9*e_charge   # electron energy in J
e_gamma=e_E/e_m/c**2 # Lorentz factor

define_bunch(Test=True)
define_bunch(Test=False,N=1e5)

#undulator1 parameters
mod1_length= 1.80
mod1_periods= 9
mod1_periodlength= mod1_length/mod1_periods
l1_lambda= 800e-9
mod1_K=np.sqrt(4*l1_lambda*e_gamma**2/mod1_periodlength-2)
mod1_Bmax=2*np.pi*mod1_K*e_m*c/(e_charge*mod1_periodlength)
l1_sigma=1.1e-3
l1_w0=l1_sigma*np.sqrt(2)
l1_E= 2.0e-3
l1_fwhm= 40e-15
l1_P_max=l1_E/(0.94*l1_fwhm)

modify_lattice("lattice.lte",Bu=mod1_Bmax,P0=l1_P_max,w0=l1_w0)
modify_lattice("lattice_test.lte",Bu=mod1_Bmax,P0=l1_P_max,w0=l1_w0)
subprocess.run(["mpiexec","-np","3","Pelegant","run_test.ele"])
dat=sdds.SDDS(0)
dat.load("run_test.out")
dE=(np.array(dat.columnData[5][0])*e_m*c**2/e_E)-1
A11=max(dE)

t1=time()
print("Elegant is Tracking through modulator 1...")
try:
    subprocess.run(["mpiexec","-np","3","Pelegant","run.ele"])
except:
    print("An error occured! Check the elegant output.")
else:    
    print("Finished Tracking!\t Runtime: ",time()-t1)

dat=sdds.SDDS(0)
dat.load("run.out")
dE=(np.array(dat.columnData[5][0])*e_m*c**2/e_E)-1
tau=-(np.array(dat.columnData[4][0])*c)
plt.figure()
plt.plot(tau-np.mean(tau),dE,',b')

MM2 = np.asarray([dat.columnData[0],dat.columnData[1],dat.columnData[2],dat.columnData[3],[list(tau-np.mean(tau))],[list(dE)]])
p_mod1= MM2.transpose((2,0,1))

R56= np.linspace(0,130e-6,131)
wls=np.linspace(350e-9,450e-9,1001)

hmap=[]
for R1 in tqdm(R56):
    p_chic1= track_chicane(p_mod1,R56=R1,isr=False)
    tau=-p_chic1[4]*c
    bn=calc_bn(tau,wls)
    #bn=bn/max(bn)
    hmap.append(bn)
plt.figure()
plt.contourf(wls*1e9,R56*1e6,hmap,levels=100)
plt.xlabel("$\lambda$ (nm)")
plt.ylabel("$R_{56}$ ($\mu m$)")










