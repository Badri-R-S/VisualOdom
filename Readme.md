Dependencied necessary to run the code:
	- scikit learn
	- pandas
	- numpy
	- cv2

Instructons to run the code: 
	- git clone the files from the repository.
	- navigate to the dataset folder.
	- Run "python3 VODOM.py" to run the code.
	- Default methods : FAST feature detction and PnPRANSAC (1).
	- To change algorithm:
		- Provide "ORB" in line 99 as first parameter in VODOM.py.
		- give "2" as a second parameter to run camera pose estimation using Essential matrix
		- give "3" as the second parameter to run camera pose estimation using Essential matrix + Local Outlier Factor (LOF).
	-Give dist_threshold value 0.5+ for FAST.
#Method3 - Results will not converge and will have high error due to its instability - Method 1 is the most stable method

Please be patient - as it takes atleast 20 mins long for method 2 and 3 - Method 1 results converge in average 15 mins. This is due to dataset size.
 