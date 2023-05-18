import cv2
import numpy as np
import matplotlib.pyplot as plt
#We are implementing two feature tracking techniques. ORB feature tracking is very common for visual odometry and it is hence extended to VSLAM. Now, the second technique we plan to use is 
#the FAST tracking algorithm. This algorithm is quicker in computation time and highly robust as its
# a simple corner detection algorithm that only requires a small number of intensity comparisons.
def ORB(image,mask=None):

    # detect and extract features from the image
    ORB = cv2.ORB_create()
    keyp1, descrip1 = ORB.detectAndCompute(image, mask)
    return keyp1,descrip1


def BFMatch(descrip1,descrip2,thresh,k):
    bf = cv2.BFMatcher_create(cv2.NORM_HAMMING2, crossCheck=False)
    matches1 = bf.knnMatch(descrip1, descrip2, k)
    knn1=[]
    for m,n in matches1:
        if m.distance < n.distance * thresh:
             knn1.append(m)
    return knn1

def match(k1, k2, d1, d2,thresh,k):
    #Now having all keypoints, we will use BF matcher to match descriptor to one image with other using distance calculation
    #All features are compared. We use euclidean distance to match descriptor
    AllMatches =BFMatch(d1,d2,thresh,k)
    matched_pairs=[]
    pA = np.float32([k1[m.queryIdx].pt for m in AllMatches])
    pB = np.float32([k2[m.trainIdx].pt for m in AllMatches])
    matched_pairs.append([pA[0], pA[1], pB[0], pB[1]])
    matched_pairs = np.array(matched_pairs).reshape(-1, 4)
    return AllMatches, pA, pB,matched_pairs

def draw_matches(img1,k1,img2,k2,match_pairs):

    match=cv2.drawMatches(img1,k1,img2,k2,match_pairs,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow("Matched Images",match)
    cv2.waitKey()

    
def draw_matches2(img1, k1, img2, k2, match_pairs):
    keypoints1 = [cv2.KeyPoint(pt[0], pt[1], 1) for pt in match_pairs[:, :2]]
    print(keypoints1)
    keypoints2 = [cv2.KeyPoint(pt[0], pt[1], 1) for pt in match_pairs[:, 2:]]

    matches = [cv2.DMatch(i, i, 0) for i in range(len(match_pairs))]
    matched_img = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # cv2.imshow("images",matched_img)
    # plt.show()
#############CODE FOR FAST##############
def FAST(img):
    fast = cv2.FastFeatureDetector_create()
    fast.setThreshold(25)
    fast.setNonmaxSuppression(True)
    keyp1 = fast.detect(img, None)
    sift=cv2.SIFT_create()
    kp1,des = sift.compute(img, keyp1)
    image=cv2.drawKeypoints(img,keyp1,None,color=(0,255,0),flags=0)
    # plt.imshow(image)
    return keyp1,des,image

imgL1=cv2.imread("C:/Users/achkr/OneDrive/Desktop/ENPM673/VisualOdom/dataset/sequences/02/image_0/000000.png")
imgR1=cv2.imread("C:/Users/achkr/OneDrive/Desktop/ENPM673/VisualOdom/dataset/sequences/02/image_1/000000.png")
imgL2=cv2.imread("C:/Users/achkr/OneDrive/Desktop/ENPM673/VisualOdom/dataset/sequences/02/image_0/000001.png")
imgR2=cv2.imread("C:/Users/achkr/OneDrive/Desktop/ENPM673/VisualOdom/dataset/sequences/02/image_1/000001.png")
k1,d1=ORB(imgL1)
k2,d2=ORB(imgL2)
print(type(d2))
# bm=BFMatch(d1,d2,0.5)
matches,key1,key2,mp=match(k1,k2,d1,d2,0.8,2)
# draw_matches2(imgL1,key1,imgL2,key2,mp)
draw_matches(imgL1,k1,imgL2,k2,matches)
keypf1,des1,img1=FAST(imgL1)
keypf2,des2,img2=FAST(imgL2)
des1 = des1.astype(np.uint8)
des2 = des2.astype(np.uint8)
matchesf,kf1,kf2,mpf=match(keypf1,keypf2,des1,des2,0.8,2)
draw_matches2(imgL1,kf1,imgL2,kf2,mpf)
draw_matches(imgL1,keypf1,imgL2,keypf2,matchesf)

cv2.waitKey()
cv2.destroyAllWindows()
