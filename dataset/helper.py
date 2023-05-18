from sklearn.neighbors import LocalOutlierFactor
import numpy as np
import cv2
import random
from Features import *
from DatasetLoader import Pleft,Pright
import random

def decomposePmat(P):
    k,r,t,_,_,_,_=cv2.decomposeProjectionMatrix(P)
    t=(t/t[3])[:3]
    return k,r,t

Kleft,Rleft,Tleft=decomposePmat(Pleft)
Kright,Rright,Tright=decomposePmat(Pright)
print("LEFT CAMERA K MATRIX : \n",Kleft)
print("LEFT CAMERA R MATRIX : \n",Rleft)
print("LEFT CAMERA T VECTOR : \n",Tleft)
def essentialMatrix(kp1,kp2,K):
    E,_=cv2.findEssentialMat(kp1,kp2,K)
    return E
def fundamental(kp1,kp2):
    # F,_=cv2.findFundamentalMat(kp1,kp2,method=cv2.LMEDS)
    # return F
    M=np.zeros(shape=((len(kp1)),9))
    for i in range(len(kp1)):
        x0,y0=kp1[i][0],kp1[i][1]
        x1,y1=kp2[i][0],kp2[i][1]
        M[i]=np.array([x0*x1,x0*y1,x0,y0*x1,y0*y1,y0,x1,y1,1])
    #Find SVD of M matrix 
    _,_,V=np.linalg.svd(M)
    F=V[-1,:]
    F=F.reshape(3,3) #Now rank is 3
    #To reduce F to a rank of 2, last singular value of F=0
    U,S,v=np.linalg.svd(F)
    #Last singular value=0
    S[-1]=0
    singular=np.zeros((3,3))
    for i in range(3):
        singular[i][i]=S[i] #Diagonal Singular Matrix
    #Now un normalize F
    F=np.dot(U,np.dot(singular,v))
    # print("Shape of F :",np.shape(F))
    return F
def essential(M1,M2,F):
    Ematrix=M2.T.dot(F).dot(M1)
    U,S,V=np.linalg.svd(Ematrix)
    S=[1,1,0] #Forcing last singular value as 0
    Ematrix=np.dot(U,np.dot(np.diag(S),V))
    return Ematrix
def get_camera_pose(E):
    U, _, V = np.linalg.svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    R = [np.dot(U, np.dot(W, V.T)), np.dot(U, np.dot(W, V.T)),
         np.dot(U, np.dot(W.T, V.T)), np.dot(U, np.dot(W.T, V.T))]
    t = [U[:, 2], -U[:, 2], U[:, 2], -U[:, 2]]
    for i in range(4):
        if np.linalg.det(R[i]) < 0:
            R[i] = -R[i]
            t[i] = -t[i]
    return R, t

def getPoints(intrin1, intrin2, inlier1, inlier2, rotation, translation):
    pts = []
    I = np.identity(3)
    Camera1 = np.dot(intrin1, np.hstack((I, np.zeros((3, 1)))))

    for i in range(len(translation)):
        xl1 = np.vstack((inlier1[:, 0], inlier1[:, 1]))  # Shape: (2, num_points)
        xl2 = np.vstack((inlier2[:, 0], inlier2[:, 1]))  # Shape: (2, num_points)

        T = np.hstack((rotation[i], -np.dot(rotation[i], translation[i].reshape(3, 1))))
        Camera2 = np.dot(intrin2, T)

        X = cv2.triangulatePoints(Camera1, Camera2,xl1,xl2)
        X /= X[3]  # Normalize homogeneous coordinates
        pts.append(X[:3])

    return pts


def GetRotTrans(pts, rot_mat, trans_mat):
    max_p = 0
    best_v = 0  # Index of the best pose that passes the chirality test

    for i in range(len(pts)):
        ptsa = pts[i]
        # ptsa = ptsa / ptsa[3, :]

        # Calculate the number of points with positive z coordinate in both cameras
        sum_of_pos_z_Q1 = sum(ptsa[2, :] > 0)
        sum_of_pos_z_Q2 = sum(ptsa[2, :] > 0)
        num = sum_of_pos_z_Q1 + sum_of_pos_z_Q2

        if num > max_p:
            best_v = i
            max_p = num
        else:
            break

    Rot = rot_mat[best_v]
    Trans = trans_mat[best_v]
    X3D = pts[best_v][:3, :].T

    return Rot, Trans, X3D
def checkcheiral(pts, R, T):
    n = 0
    for i in range(pts.shape[1]):
        # Homogeneous coordinates of the 3D point
        X_hom = np.hstack((pts[:,i], 1))

        # Convert homogeneous coordinates to 3D coordinates
        X = X_hom[:3] / X_hom[3]

        # Check if R3(X-T)>0
        if R[2,:].dot(X - T) > 0:
            n += 1

    return n 


def LOF(p1, p2, K1, K2, threshold=0.02, num_iter=5000):
    # Convert the input points to homogeneous coordinates
    p1H = np.hstack((p1, np.ones((p1.shape[0], 1))))
    p2H = np.hstack((p2, np.ones((p2.shape[0], 1))))

    # Initialize the best fundamental matrix and inliers
    best_F = None
    best_inliers = []
    lof = LocalOutlierFactor(n_neighbors=20, contamination='auto')
    lof.fit(p1)
    outlier_scores = -lof.negative_outlier_factor_

    inliers = outlier_scores > threshold
    if np.sum(inliers) > np.sum(best_inliers):
        best_inliers = inliers

    # Get the inlier points
    inlier_p1 = p1[np.where(best_inliers)[0]]
    inlier_p2 = p2[np.where(best_inliers)[0]]

    # Perform LMedS using the inlier points
    best_R = None
    best_t = None
    best_error = float('inf')

    for i in range(num_iter):
        # Select a random sample of 4 points to estimate pose
        sample_size = min(4, inlier_p1.shape[0])
        rand_idx = random.sample(range(inlier_p1.shape[0]), sample_size)

        # Estimate camera pose using the selected points
        E = essential(K1, K2, best_F)
        final_R, final_t = get_camera_pose(E)
        pts = getPoints(K1, K2, inlier_p1[rand_idx], inlier_p2[rand_idx], final_R, final_t)
        R, t, final_pts = GetRotTrans(pts, final_R, final_t)

        # Check chirality
        num_positive_Z = checkcheiral(final_pts.T, R, t)

        # Update the best camera pose if the current one has a lower chirality error
        if num_positive_Z < best_error:
            best_R = R
            best_t = t
            best_error = num_positive_Z

    return best_R, best_t, inlier_p1, inlier_p2