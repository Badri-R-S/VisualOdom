import cv2
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as pl
import numpy as np
import pandas as pd

from helper import *
from Disparity import *
from EstimationMotion import *
from DatasetLoader import *
from Features import *

def VODOM(detector,method,mask):
    #Create empty transformation matrix         
    T_m = np.eye(4)

    estimated = np.zeros((frames, 3, 4))
    estimated[0] = T_m[:3, :]

    
    k_left, r_left, t_left = decomposePmat(Pleft)
    k_right, r_right, t_right = decomposePmat(Pright)
    
    # Iterate through all frames
    for i in range(frames -1):

        img_l = images_left[i]
        img_r = images_right[i]
        img_p1 = images_left[i+1]
    
        depth = calculateDepthDisp(img_l, 
                                   img_r, 
                                   'semi',
                                   Pleft, 
                                   Pright)
            
        if detector=='FAST':
            kp0, des0,_= FAST(img_l, mask)
            kp1, des1,_= FAST(img_p1, mask)
            des1 = des1.astype(np.uint8)
            des0 = des0.astype(np.uint8)
        
        # Get matches between features detected in the two images
            matches,key1,key2,mp=match(kp0,kp1,des0,des1,0.5,2)
        
        elif detector=='ORB':
            kp0, des0= ORB(img_l, mask)
            kp1, des1= ORB(img_p1, mask)
        
        # Get matches between features detected in the two images
            matches,key1,key2,mp=match(kp0,kp1,des0,des1,0.5,2)

        if method==1:
            rot,trans,_,_=motion1(key1,key2,k_left,depth)
        elif method==2:
            rot,trans,_=motion2(key1,key2,k_left,k_right)
        elif method==3:
            rot,trans,_=motion3(key1,key2,k_left,k_right,depth)
        print("PROCESSING NOW :",i)

        # Create blank homogeneous transformation matrix
        Tr = np.eye(4)
       
        Tr[:3, :3] = rot
        Tr[:3, 3] = trans.T
        inv_Tmat = np.linalg.inv(Tr)

        
        T_m = T_m.dot(inv_Tmat)

        estimated[i+1, :, :] = T_m[:3, :]

        
    return estimated

def ViewPlot(V):
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(V[:, :, 3][:, 0], 
            V[:, :, 3][:, 1], 
            V[:, :, 3][:, 2], label='estimated', color='orange')

    ax.plot(groundt[:, :, 3][:, 0], 
            groundt[:, :, 3][:, 1], 
            groundt[:, :, 3][:, 2], label='ground truth')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax.view_init(elev=-20, azim=270)
    plt.show()

Pleft,Pright,_,_,groundt,imgL1,imgR1,imgL2,imgR2,imh1,imw1,images_left,images_right,frames=dataset_loader('02')
Kleft,reft,tleft=decomposePmat(Pleft)
Kright,rright,tright=decomposePmat(Pright)
v=VODOM('ORB',1,ROI)
ViewPlot(v)