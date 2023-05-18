import cv2
import numpy as np
from Features import *
from Disparity import *
from DatasetLoader import *
from helper import *
def motion1(pA, pB, p, depth1, max_depth=3500):
    rot=np.eye(3)
    transvector=np.zeros((3,1))
    #here we can take two approaches 
    #Approach 1 uses the depth and gets the real world 3D points from 2D points as(u,v,depth)
    #This is done by measuring disparity between 2 two views of same with different views. 
    if depth1 is not None:
        dPoints=np.zeros((0,3))
        outlier=[]
        for i ,(x,y) in enumerate(pA):
            m=depth1[int(y),int(x)]
            if m>max_depth:
                outlier.append(i)
                continue
        # Obtaining points from image and reprojecting into 3D world, 
            dPoints = np.vstack([dPoints, np.linalg.inv(p).dot(m*np.array([x, y, 1]))])
        #Now delete the out;iers from image points with best feature
        pA=np.delete(pA,outlier,0)
        # print(len(dPoints))
        pB=np.delete(pB,outlier,0)
        #Get RotMat and TVector from the points projected , and image points are 2D ixel coords in second image
        #PNP takes min no points to solve for translation and takes the solution to check in error margin
        #If the number of points is not above inlier threshold
        _,rot,trans,inliers=cv2.solvePnPRansac(dPoints,pB,p,None)
        #AS per CV2 documentation, the rotation matrix returns axis angle representation,and we need
        #we can use Rodrigues formula
        rmat = cv2.Rodrigues(rot)[0]

    return rmat, trans, pA, pB

#Method2
def motion2(kp1,kp2,kl,kr):

    M_r = np.hstack((R, t))
    M_l = np.hstack((np.eye(3, 3), np.zeros((3, 1))))
    kp1hom = np.hstack([kp1, np.ones(len(kp1)).reshape(-1,1)])
    kp2hom = np.hstack([kp2, np.ones(len(kp2)).reshape(-1,1)])
    E = cv2.findEssentialMat(kp1hom, kp2hom, kl)[0]
    _, R, t, mask = cv2.recoverPose(E, kp1hom, kp2hom, kl)

    P_l = np.dot(kl,  M_l)
    P_r = np.dot(kr,  M_r)
    point_4d_hom = cv2.triangulatePoints(P_l, P_r, np.expand_dims(kp1, axis=1), np.expand_dims(kp2, axis=1))
    point_4d = point_4d_hom / np.tile(point_4d_hom[-1, :], (4, 1))
    point_3d = point_4d[:3, :].T
    return R,t,point_3d
#Method3
def motion3(pA, pB, p1,p2,depth1, max_depth=3500):
    rot=np.eye(3)
    transvector=np.zeros((3,1))
    #here we can take two approaches 
    #Approach 1 uses the depth and gets the real world 3D points from 2D points as(u,v,depth)
    #This is done by measuring disparity between 2 two views of same with different views. 
    if depth1 is not None:
        dPoints=np.zeros((0,3))
        outlier=[]
        for i ,(x,y) in enumerate(pA):
            m=depth1[int(y),int(x)]
            if m>max_depth:
                outlier.append(i)
                continue
        # Obtaining points from image and reprojecting into 3D world, 
            dPoints = np.vstack([dPoints, np.linalg.inv(p1).dot(m*np.array([x, y, 1]))])
        #Now delete the out;iers from image points with best feature
        pA=np.delete(pA,outlier,0)
        # print(len(dPoints))
        pB=np.delete(pB,outlier,0)
        #Get RotMat and TVector from the points projected , and image points are 2D ixel coords in second image
        #PNP takes min no points to solve for translation and takes the solution to check in error margin
        #If the number of points is not above inlier threshold
        rot, trans, _, _ = LOF(pA, pB,p1,p2)
        inliers = np.ones(len(pA), dtype=bool)
        # trans=trans.reshape(3,1)


  
    return rot, trans, pA, pB

# Rot2,trans2,d3=motion2(key1,key2,Kleft,Kright)
# print("ROTATION \n",Rot2)    
# axx=d3[:,0]
# ay=d3[:,1]
# az=d3[:,2]
# fig = plt.figure(figsize=(7,6))
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(axx,ay,az)
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# ax.view_init(elev=-40, azim=270)
#FOR ORB
# depth1=calculateDepthDisp(imgL1,imgR1,'semi',Kleft,tleft,tright)
# rot,trans,_,_=motion1(key1,key2,Kleft,depth1)
# print(rot)
# print(trans)