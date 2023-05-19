# Dependencies necessary to run the code:
	- scikit learn
	- pandas
	- numpy
	- cv2

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
# Pipeline 
Using PNPSOLVER as the centric trajectory solver, we first computed depth and feature matching. For depth and disparity, we used stereo SGBM, or semi sliding window, which gave better results compared to SBM. 
![DisparitySBGM](https://github.com/Achuthankrishna/VisualOdom/assets/74654704/23a7db9c-c660-4919-af76-0940ce035a71)


# WARNING
Please be patient - as it takes atleast 20 mins long for method 2 and 3 - Method 1 results converge in average 15 mins. This is due to dataset size.
 
