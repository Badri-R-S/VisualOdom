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



# WARNING
Please be patient - as it takes atleast 20 mins long for method 2 and 3 - Method 1 results converge in average 15 mins. This is due to dataset size.
 
