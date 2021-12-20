# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 11:44:29 2020

@author: Arjun
"""

import os
#Changing the working directory to the source directory
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
from lsrmod_functions import *
from tqdm import tqdm


##### natural constants #####
c=const.c           # speed of light
e_charge=const.e    # electron charge
m_e=const.m_e       # electron mass in eV/c^2
Z0=376.73           # impedance of free space in Ohm
epsilon_0= const.epsilon_0 # vacuum permittivity

e_E=1.492e9*e_charge    # electron energy in J
e_gamma=e_E/m_e/c**2    # Lorentz factor

        
##### Simulation parameter #####
slicelength=50e-6    # length of simulated bunch slice in m
tstep=5e-12         # timestep in s
N_e=int(1e5)        # number of electrons

bunch_test=define_bunch(Test=True)
bunch_init=define_bunch(Test=False,N=N_e,slicelength=slicelength)
elec=np.copy(bunch_init)

##### defining Laser 1 #####
l1_wl= 800e-9   # wavelength of the laser
l1_sigx= 0.7e-3 # sigma width at the focus
l1_fwhm=40e-15  # pulse length 
l1_E= 2e-3      # pulse energy

##### defining modulator 1 #####
mod1= Modulator(periodlen=0.20,periods=9,laser_wl=l1_wl,e_gamma=e_gamma)
l1= Laser(wl=l1_wl,sigx=2*l1_sigx,sigy=l1_sigx/2,pulse_len=l1_fwhm,pulse_E=l1_E,focus=mod1.len/2,M2=1.0,pulsed=False,phi=0e10)


#### Test Tracking through Modulators
elec_test= lsrmod_track(mod1,l1,bunch_test,tstep=tstep,disp_Progress=False)
z,dE=calc_phasespace(elec_test,plot=False)
A11=(max(dE))

print("A1= ",A11)


#defining laser noise values
l1.pulsed=True
l1.phi=0e10


print("\n\nTracking through Modulator 1...")
elec_M1= lsrmod_track(mod1,l1,elec,tstep=tstep)

R56= np.linspace(0,130e-6,131)
wls=np.linspace(180e-9,220e-9,1001)

print("\n\nCalculating Spectra...")
hmap=[]
for R1 in tqdm(R56):
    elec_C1=chicane_track(elec_M1,R56=R1)
    z,dE=calc_phasespace(elec_C1,plot=False)
    bn=calc_bn(z,wls)
    hmap.append(bn)

plt.figure()
plt.contourf(wls*1e9,R56*1e6,hmap,levels=100)
plt.xlabel("$\lambda$ (nm)")
plt.ylabel("$R_{56}$ ($\mu m$)")
