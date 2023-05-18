import pandas as pd  
import numpy as np 
import cv2  
import os

def dataset_loader(number):
    filename=number
    seq_dir = '/home/smit/ros1_ws/src/VisualOdom/dataset/sequences/{}/'.format(number)
    poses_dir = '/home/smit/ros1_ws/src/VisualOdom/dataset/poses/{}.txt'.format(number)
    poses = pd.read_csv(poses_dir, delimiter=' ', header=None)
    
    groundt = np.zeros((len(poses), 3, 4))
    for i in range(len(poses)):
        groundt[i] = np.array(poses.iloc[i]).reshape((3, 4))
    left_imagesfie=os.listdir(seq_dir+'image_0')
    right_imagesfie=os.listdir(seq_dir+'image_1')
    
    num_frames = len(left_imagesfie)

    # Get calibration details for scene
    calib = pd.read_csv(seq_dir + 'calib2.txt', delimiter=' ', header=None, index_col=0)
    P0 = np.array(calib.loc['P0:']).reshape((3, 4))
    P1 = np.array(calib.loc['P1:']).reshape((3, 4))
    P2 = np.array(calib.loc['P2:']).reshape((3, 4))
    P3 = np.array(calib.loc['P3:']).reshape((3, 4))

    images_left = [cv2.imread(seq_dir + 'image_0/' + name_left, 0) for name_left in sorted(left_imagesfie)]
    images_right = [cv2.imread(seq_dir + 'image_1/' + name_right, 0) for name_right in sorted(right_imagesfie)]

    images_left = list(images_left)
    images_right = list(images_right)
    
    left_first = images_left[0]
    left_second= images_left[1]

    right_first=images_right[0]
    right_second=images_right[1]
    print(left_imagesfie[0])
    

    return (P0, P1, P2, P3,groundt,left_first,left_second,
            right_first, images_left, images_right,num_frames)

def decomposePmat(P):
    k,r,t,_,_,_,_=cv2.decomposeProjectionMatrix(P)
    t=(t/t[3])[:3]
    return k,r,t
    
Pleft,Pright,_,_,groundt,imgL1,imgL2,imgR1,images_left,images_right,frames=dataset_loader('02')
Kleft,reft,tleft=decomposePmat(Pleft)
Kright,rright,tright=decomposePmat(Pright)
print(groundt)
# cv2.imshow("LEFT",imgL1)
# cv2.waitKey()
# cv2.destroyAllWindows()