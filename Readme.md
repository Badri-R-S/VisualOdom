# VISUAL ODOMETRY 
# Dependencies necessary to run the code:
```
pip3 install scikit-learn
pip3 install pandas
pip3 install numpy
import cv2
```
# Instructons to run the code: 
	- git clone the files from the repository.
	- navigate to the dataset folder.
	- Run "python3 VODOM.py" to run the code.
	- Default methods : FAST feature detction and PnPRANSAC (1).
	- To change algorithm:
		- Provide "ORB" in line 99 as first parameter in VODOM.py.
		- give "2" as a second parameter to run camera pose estimation using Essential matrix
		- give "3" as the second parameter to run camera pose estimation using Essential matrix + Local Outlier Factor (LOF).
	-Give dist_threshold value 0.5+ for FAST.
	- Method3 - Results will not converge and will have high error due to its instability - Method 1 is the most stable method
# Depth Estimation 
Using PNPSOLVER as the centric trajectory solver, we first computed depth and feature matching. For depth and disparity, we used stereo SGBM, or semi sliding window, which gave better results compared to SBM. The below image shows our result when performed with stereo SGBM.

![DisparitySBGM](https://github.com/Achuthankrishna/VisualOdom/assets/74654704/23a7db9c-c660-4919-af76-0940ce035a71)

Using that, we computed depth, which gave us a better understanding of the depth information.

![Dpth](https://github.com/Achuthankrishna/VisualOdom/assets/74654704/0d4044f2-c399-4f31-9ab8-9f26f9c249c1)

# Feature Extraction
We approached with two methods. First one was using ORB , a quick method to get matci=hing keypoints and matches. Used a lowe's distance of 0.8 to filter best matches, and we were able to estimate around deccent amount of matches for the given dataset sequence.

![FilteredORB](https://github.com/Achuthankrishna/VisualOdom/assets/74654704/f267632d-12ab-434c-a69e-b707133e6c19)


Second feature extraction, we combined FAST + SIFT , that provided more robust results than ORB and the computation time was slightly higher than ORB.

![ORBAST](https://github.com/Achuthankrishna/VisualOdom/assets/74654704/853c5b94-4144-41ec-918f-d39f71ca65df)

# Motion Estimation
Deployed three approaches 
- Approach 1 : Point-n-Perspective method and depth information to get project 2D image points as 3D homogenous points and estimate Translation Vector and Rotation matrix using SolvePnP. This is the most efficient method 

![FAST+PNP](https://github.com/Achuthankrishna/VisualOdom/assets/74654704/8d3ce938-3bba-45bb-9d20-92ed5a5a4a76)



- Approach 2 : Calculate Camera Pose using Essential Matrix and recover pose using LMedS - Unstable

![FAST-Method2](https://github.com/Achuthankrishna/VisualOdom/assets/74654704/7b638c25-741a-425b-b32e-6242fb57cdd0)

- Approach 3 : Implement Local Outlier Factor for 20 neighbors and calculate essential matrix and get rotation and translation matrices. - Highly Unstable 

![output-2](https://github.com/Achuthankrishna/VisualOdom/assets/74654704/11b09244-66d2-43d1-b961-da793ab6c6a0)

# CONTRIBUTORS 
Badrinaryan - [@[Irdab2000](https://github.com/Irdab200
)]

Smit Dumore - [[@smitdumore](https://github.com/smitdumore)]

Achuthan- [@[Achuthankrishna](https://github.com/Achuthankrishna)]
# WARNING
Please be patient - as it takes atleast 20 mins long for method 2 and 3 - Method 1 results converge in average 15 mins. This is due to dataset size.
 
