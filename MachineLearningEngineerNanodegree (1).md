# Machine Learning Engineer Nanodegree

## Capstone Project

M P S Vishal Singh Tomar   
10 September 2019

## I. Definition

### Project Name
**Detecting diabetic retinopathy :**  Detect diabetic retinopathy to stop blindness before it's too late
### Project overview  

Imagine being able to detect blindness before it happened.

Millions of people suffer from diabetic retinopathy, the leading cause of blindness among working aged adults. This project aim to detect and prevent this disease among people living in rural areas where medical screening is difficult to conduct.

Currently, Technicians travel to these rural areas to capture images and then rely on highly trained doctors to review the images and provide diagnosis. Their goal is to scale their efforts through technology; to gain the ability to automatically screen images for disease and provide information on how severe the condition may be.

We will build a machine learning model to speed up disease detection. Working with thousands of images collected in rural areas to help identify diabetic retinopathy automatically. If successful, we will not only help to prevent lifelong blindness, but these models may be used to detect other sorts of diseases in the future, like glaucoma and macular degeneration.
 

### Problem Statement



&nbsp; &nbsp; &nbsp; **Diabetic retinopathy (DR)** is the fastest growing cause of blindness, with nearly **415 million** diabetic patients at risk worldwide. If caught early, the disease can be treated; if not, it can lead to irreversible blindness. 
Lets Look at the following diagram to have a better understanding of the problem. 
![Drag Retina](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRlGGC3Md8NrH-qaNi-aYHGPu6EASL2C_MmzLuiAvKLuwPMjqN_YA)
 + There are at least 5 things to spot on. 
 + These five spots are not easily detectable.
 + infact some images in dataset have very poor lighting which makes it even more harder to spot these spots. These things make tarining a deep Learning model difficult.
 
 The following figure is a screen shot of dataset images.
 

 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  ![Drag Retina](https://www.sookshmas.com/ImgsU/m/491/1568186544566491.png)

It is clear that images have poor lighting and nerves and spots which are to be checked are hardly visible.
We need a way to improve its lighting conditions and aslo highlight nerves,spots and try to remove other not so relevent features from the images so the our model can efficiently perform. 

### Metrics

I will use Log-loss function for performance analysis.  Logarithmic Loss or log Loss, Works by penalising the false classifications. It works well for multi-class classification. When working with Log Loss, the classifier must assign probablity to each class for all the samples . Suppose, there are N samples belonging to M classes, then the Log Loss is calculated as below:
![Log Loss](https://miro.medium.com/max/581/0*i2_eUc_t8A1EJObd.png)

Log Loss has no upper bound and it exists on the range [0, ∞). Log Loss nearer to 0 indicates higher accuracy, whereas if the Log Loss is away from 0 then it indicates lower accuracy.
In general, minimising Log Loss gives greater accuracy for the classifier.

Log Loss is a Micro -average matrix.

Micro-average is preferable if there is a class imbalance problem.

In our sample data we cannot assume to have a balanced set of images for all the classes hence I prefer using Log Loss function.

## II. Analysis

### Data Exploration


&nbsp; &nbsp; This data set is taken from [kaggle competitions](https://www.kaggle.com/competitions) it is publicly available as a part of [APTOS 2019 Blindness Detection](https://www.kaggle.com/c/aptos2019-blindness-detection/) competition.

Dataset includes both images and csv files.

 + csv files named as **train.csv** and **test.csv** contains
     1. id_code
     2. diagnosis
 	 
 

- **id\_code** is same as the image file name i.e., Each row represents a image filename as **id_code** and what category it belongs to as **diagnosis**.

A clinician has rated each image for the severity of diabetic retinopathy on a scale of 0 to 4: 



	0 - No DR

	1 - Mild

	2 - Moderate

	3 - Severe

	4 - Proliferative DR 
	
1. Below  diagram   shows image samples in the dataset to be used.  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  ![Drag Retina](https://www.sookshmas.com/ImgsU/m/491/1568186544566491.png)
 2. These are retinal photographic images.
3. These images indicate damage to eyes by showing blood cloting, darkspots or brusted blood vessels. 
4. These things are the root causes for blindness. Hence instead of tracking Patient's medical record data which will have many irrelevent parameters we can focus on these symptoms.  
5. The dataset we use contains images which are cassified into below 5 classes. 
6. Number of images per classes are:
    * Class 0 ( No DR) : 1805 images (49.2%)
    * Class 1 (Mild) : 370 images (10.1%)
    * Class 2 (Moderate) : 999 images (27.28%)
    * Class 3 (Severe) : 193 images (5.27%)
    * Class 4 (Proliferative DR ) : 295 images (8.05%)
7. Hence we can conclude that dataset is largly imbalanced.
8. I will randomly split data into train,test and validation sets with 60%,20% and 20% respectively


### Exploratory Visualization
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![grahp](https://www.sookshmas.com/ImgsU/m/491/1571740522382491.png)

The above graph shows the number of sample images in each category(0-4)  Train(blue), test(yellow) and valid(green). It is clear that number of sample data is not balanced accross the categories. Hence we have to carefully select the evaluation Metrics.

### Algorithms and Techniques

- I have used Ben Graham's preprocessing method to improve lightning condition in the images.
- Auto image croping method to crop out black background around the retina images.
- VGG16 pre trained model to extract bottleneck features of the images.


### Benchmark Model

This project is still under research hence no solid papres have been published on benchmarks. Hence i would like to use the following [notebook](https://www.kaggle.com/mathormad/aptos-resnet50-baseline) published publicly by [KeepLearning](https://www.kaggle.com/mathormad) as benchmark. 
+ This is a ResNet50 implementation with loss='categorical_crossentropy'. Applied on image size = 300.
 

## III. Methodology

### Data Preprocessing
+ First we will load the image.
+ Resize the images to 224 x 224 x 3 so that it matches the ImageNet format.

+ Then we will use Ben Graham's preprocessing method [2] to improve the lightning condition. As mentioned in the above problem statment image clarity is poor due to dull lighting. 
  +  We will first try to sharpen image using GaussianBlur function. 
  + Then crop the image to remove black space on the side.
  
**Mixup & Data Generator**

We will create a data generator that will perform random transformation to our datasets (flip vertically/horizontally, rotation, zooming). This will help our model generalize better to the data.

### Implementation
I have used vgg16 pre trained model to create Bottleneck features of the given data. Then made a simple model ( summary screenshot shared below ) and Fed the botained bottleneck features as input to this model.
![transfer learning ](https://www.sookshmas.com/ImgsU/m/491/1571741758512491.png)

Below figure demonstrates the implementation of the model

![transfer learning ](https://www.sookshmas.com/ImgsU/m/491/1571742048865491.jpg)


## IV. Results


### Model Evaluation and Validation

The final result obtained for the model on test data is loss=0.528 and accuracy of 79.81%. I used vgg16 model to create bottleneck features and then used it as input to my model.
### Justification
The [work](https://www.kaggle.com/mathormad/aptos-resnet50-baseline) published by [KeepLearning](https://www.kaggle.com/mathormad) on kaggle uses Resnet50 pretrained model. It gives an accuracy of 76.63 and loss = 0.697 on the test data it also took about 512 sec for each epoch on kaggle gpu.
I used vgg16 model instaded of Resnet50 which improved the learning time a lot. It took around 62 sec for each epoch which is very less comapred to resnet50 model. I have done some changes in preprocesing step which significintly improved the result. Obtained model's accuracy is 79.81 and loss= 0.528
## V. Conclusion

### Free-Form Visualization
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ![visual](https://www.sookshmas.com/ImgsU/m/491/221019040507491.png)

In this image we use heatmap and overlay to visualise our predictive model's output.
 - We see that in the fig1 there are some white spots in the middle which is reflected in the heat map and overly as blue or cooler area 
 - In the fig2 there are black spots in the middel which are maked as red area.
 - This indicates that model is correctly able to differenciate between different features.

### Reflection

This project is based on image recognition, Unlike other image recognition projects it deals an unsual set of images. There were many challanges faced during the pre processing of the images. 

![visual](https://www.sookshmas.com/ImgsU/m/491/1571743292642491.png)

As you can see in the above figure
 - Images were very dull and visiblity was poor.
 - Images had black background.
 - Features are very hard to recognize.


I learned a lot about image processing through this project techniques such as Ben Graham's preprocessing method discribles how to improve visiblity by changing image gama value and also auto croping black background. Below figure shows the processed images.

![visual](https://www.sookshmas.com/ImgsU/m/491/1571743766964491.png)


### Improvement

I feel that there is still scope of improvement in the image preprocessing step, As these are retinal images we can try different preprocessing methods to improve the quality. 