

import numpy as np 
import matplotlib.pyplot as plt
import math

def basic_ranging(depth,ss,sed_speed,t,plotflag=False):


    n0=0    
    n1=1
    n2=2
    n3=3

    distance=np.linspace(0,20000,2001,endpoint=True) #Used for waterborne arrival calculations
    theta1s=np.linspace(0,3.14/4,150,endpoint=True) #Used for subsurface arrival calculations



    ############Waterborne arrival calculations
    t0=[((2*n0+1)*math.sqrt((d/(2*n0+1))**2+depth**2))/ss for d in distance]
    t1=[((2*n1+1)*math.sqrt((d/(2*n1+1))**2+depth**2))/ss for d in distance]
    t2=[((2*n2+1)*math.sqrt((d/(2*n2+1))**2+depth**2))/ss for d in distance]
    t3=[((2*n3+1)*math.sqrt((d/(2*n3+1))**2+depth**2))/ss for d in distance]


    #########Basement arrival calculations
    if t > 0:
        theta2s=[math.asin(sed_speed*math.sin(theta1)/ss) for theta1 in theta1s]
        d_dbs=[math.sqrt((depth/math.cos(theta1))**2-depth**2)+2*math.sqrt((t/(math.cos(math.asin(sed_speed*math.sin(theta1)/ss))))**2-t**2) for theta1 in theta1s]
        d_mpbs=[3*math.sqrt((depth/math.cos(theta1))**2-depth**2)+2*math.sqrt((t/(math.cos(math.asin(sed_speed*math.sin(theta1)/ss))))**2-t**2) for theta1 in theta1s]
        d_mpb2s=[3*math.sqrt((depth/math.cos(theta1))**2-depth**2)+4*math.sqrt((t/(math.cos(math.asin(sed_speed*math.sin(theta1)/ss))))**2-t**2) for theta1 in theta1s]
        H1=[depth/math.cos(theta1) for theta1 in theta1s]
        H2=[t/math.cos(theta2) for theta2 in theta2s]
        TT1=[math.sqrt(depth**2+d_db**2)/ss for d_db in d_dbs]
        TT2=[math.sqrt(depth**2+d_mpb**2)/ss for d_mpb in d_mpbs]
        TT3=[math.sqrt(depth**2+d_mpb2**2)/ss for d_mpb2 in d_mpb2s]
        inds=np.arange(len(H1))
        TT_db=[H1[j]/ss+(2*H2[j])/sed_speed for j in inds]
        TT_mpb=[(3*H1[j]/ss)+(2*H2[j])/sed_speed for j in inds]
        TT_mpb2=[(3*H1[j]/ss)+(4*H2[j])/sed_speed for j in inds]

    ##########Interpolate basement arrivals to match water column arrivals
        d_interp=np.interp(distance,d_dbs,TT_db)
        mp_interp=np.interp(distance,d_mpbs,TT_mpb)
        mp_interp2=np.interp(distance,d_mpb2s,TT_mpb2)
        mp_basement=np.subtract(mp_interp,t0)
        mp_basement2=np.subtract(mp_interp2,t0)

    else:
        d_interp=[]
        mp_interp=[]
        mp_interp2=[]

    if plotflag:
        plt.scatter(distance,np.subtract(t0,t0))
        plt.scatter(distance,np.subtract(d_interp,t0))
        plt.scatter(distance,np.subtract(t1,t0))
        plt.scatter(distance,np.subtract(mp_interp,t0))
        plt.scatter(distance,np.subtract(mp_interp2,t0))
        plt.scatter(distance,np.subtract(t2,t0))
        plt.scatter(distance,np.subtract(t3,t0))
        plt.show()

    return(distance,t0,t1,t2,t3,d_interp,mp_interp,mp_interp2)