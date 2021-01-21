## Trust-based Recommender System

#### System Features:  
- Has a Simple GUI.  
- Employs Trust between users to predict items ratings.
- Evaluate the results and compare them with the real data.  

#### Steps to run the code:  

1-	Install latest [python](https://www.python.org/downloads/) version.  
2-	From Windows command line/Mac terminal move to the project directory.   
3-	Type: ```python TrustRS.py```   
4-	If any error appears reagarding missing packages then you may install them by typing:  
```
pip install PySimpleGUI
pip install pandas
pip install numpy
pip install scipy
pip install sklearn
```
5-	Run again.   
6-	The following window should appear:   

<img src="https://github.com/Mhz95/Trust-Based-Recommender-System/blob/master/scrn.png" width="500">
 
 
7-	Browse and select RS_Dataset folder, Set Review_sample, Trust_sample and Similarity sample files.  
8-	Click on Preprocess Data. Holdout is the percentage of the test set. The rest is used as a training set.  
9-	Select an approach (Refer to our paper to find out description of the approaches)  
10-	Compute trust values (usually takes 20-30 min.)  
11-	Predict Ratings.  
12-	Evaluate System, it will show the evaluation metrics results i.e MAE, MSE, RMSE. Also, it will show a comparison of the top 10 recommended items for both real and predicted ratings.  

<img src="https://github.com/Mhz95/Trust-Based-Recommender-System/blob/master/scrn2.png" width="500">


