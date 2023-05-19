import cv2
import numpy as np
from DatasetLoader import *
def computedisparity(img1,img2,method,blk=11):
    # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    if(method=='semi'):
        matcher = cv2.StereoSGBM_create(numDisparities=96,
                                            minDisparity=0,
                                            blockSize=11,
                                            P1 = 8 * 3 * 6 ** 2,
                                            P2 = 32 * 3 * 6 ** 2,
                                            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
                                        )
    elif (method=='sbm'):
        matcher = cv2.StereoBM_create(numDisparities=96,blockSize=blk)

    disp_map= matcher.compute(img1,img2).astype(np.float32)/16
    # disp_map = (disp_map / disp_map.max()) * 255

    return disp_map

def computedepth(dmap,kleft,tleft,tright):
    #Focal length of camera fx can be identified using K matrix
    focal=kleft[0][0]
    baseline=tright[0]-tleft[0]
    #to avoid division by zero we take 0 to be as 0.01
    dmap=dmap.astype(np.float32)
    dmap[dmap == 0.0] = 0.1
    dmap[dmap == -1.0] = 0.1
    #Depth = f*b/dmap
    depth=np.ones(dmap.shape)
    depth=focal*baseline/dmap
    return depth

def calculateDepthDisp(img1,img2,method,Pleft,Pright,blk=11):
    dmap=computedisparity(img1,img2,method,blk)
    kleft,rleft,tleft=decomposePmat(Pleft)
    kright,right,tright=decomposePmat(Pright)
    depthinfo=computedepth(dmap,kleft,tleft,tright)
    return depthinfo   

ROI = np.zeros(imgL1.shape[:2], dtype=np.uint8)
yroi = imgL1.shape[0]
xroi= imgL1.shape[1]
cv2.rectangle(ROI, (95,0), (xroi,yroi), (255), thickness = -1)