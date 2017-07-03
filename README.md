# traffic-sign-classifier
A traffic sign classifier using deep learning and tensorflow. It's the project 2 of the udacity Self-Driving Car Nanodegree. It can only recognize German traffic signs at the moment.

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./images/unique_traffic_signs.png "Unique Traffic Signs"
[image2]: ./images/histogram_train.png "Training Histogram"
[image3]: ./images/histogram_valid.png "Validation Histogram"
[image4]: ./images/histogram_test.png "Test Histogram"
[image5]: ./images/preprocessed.png "Pre-processing"
[image6]: ./images/augmented.png "Data Augmentation"
[image7]: ./new-images/1.jpg "Priority"
[image8]: ./new-images/10.jpg "Roundabout"
[image9]: ./new-images/11.jpg "Yield"
[image10]: ./new-images/12.jpg "Children Crossing"
[image11]: ./new-images/13.jpg "Ahead Only"
[image12]: ./images/softmax_priority.png "Priority"
[image13]: ./images/softmax_roundabout.png "Roundabout"
[image14]: ./images/softmax_yield.png "Yield"
[image15]: ./images/softmax_children_crossing.png "Children Crossing"
[image16]: ./images/softmax_ahead_only.png "Ahead Only"

## Rubric Points

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used numpy to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799.
* The size of the validation set is 4410.
* The size of test set is 12630.
* The shape of a traffic sign image is (32, 32, 3).
* The number of unique classes/labels in the data set is 43.

#### 2. Include an exploratory visualization of the dataset.

Here are all unique traffic signs of the data set.

![alt text][image1]

Below are the bar charts showing the data distributions of the training data set, the validation data set and the test data set:

![alt text][image2]
![alt text][image3]
![alt text][image4]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

My classifier performed the following three pre-processing steps:
1. Normalization
2. Histogram equalization
3. Grayscaling

I normalized the image data so that the optimizer can find a good solution quickly and it did improve the validation accuracy.

I decided to apply the histogram equalization to the images because I found some images in the training set are either too bright or too dark to be recognizable, even by a human.

The grayscaling was inspired by [the paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). It said converting the images to grayscale improved their test accuracy.

Here is an example of a traffic sign image before and after pre-processing.

![alt text][image5]

To further improve the accuracy, I augmented the training data set with the following transformations:

![alt text][image6]

The total number of the final training set after the data augmentation is 253,523. I did not flip every image in the data set because some traffic signs cannot be flipped without changing their meanings.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x48 	|
| ReLU					|												|
| Max pooling	      	| 2x2 stride, same padding, outputs 14x14x48 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x96		|
| ReLU					|												|
| Max pooling	      	| 2x2 stride, same padding, outputs 5x5x96 				|
| Fully connected		| outputs 240        									|
| ReLu
| Dropout		|         									|
| Fully connected		| outputs 120        									|
| ReLu
| Dropout		|         									|
| Fully connected		| outputs 43        									|
| Softmax				|         									|

It's basically a wider LeNet-5.

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the following hyperparameters:

| HyperParameter         		|     Value	        					| 
|:---------------------:|:---------------------------------------------:| 
| Optimizer         		| Adam | 
| Learning rate         		| 0.001 |
| L2 regularization beta  		| 0.0001 |
| Batch size  		| 256 |
| # of epochs  		| 40 |
| dropout probability  		| 0.5 |

It took nearly 9 minutes to run on my GTX 1080 Ti. 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 1.000
* validation set accuracy of 0.995
* test set accuracy of 0.979

I chose the LeNet-5 as my baseline network architecture because it can easily get 0.98 accuracy on the MNIST dataset, which consists of images of handwritten digits, and it can even be trained on a CPU.

With the LeNet-5, I achieved the validation set accuracy of 0.89. Then I normalized the dataset and got 0.917 validation accuracy. I added two dropout layers between the last fully connected layers and it boosted the accuracy to 0.958 immediately. But the training set accuracy decreased from 1.000 to 0.997. It seemed to me that the dropout layers regularized the network too much. To fix this underfitting problem, I increased the network capacity by adding more neurons to the convolution and the fully connected layers and it worked as expected. The training and the validation accuracy increased to 1.000 and 0.972, respectively. The L2 regularization pushed the accuracy to 0.980. To further improve the accuracy, I augmented the dataset and it improved the accuracy to 0.990. By converting images to grayscale and applying the histogram equalization, the validation accuracy reached 0.995.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image7] ![alt text][image8] ![alt text][image9] 
![alt text][image10] ![alt text][image11]

I thought the 2th and 4th image would be quite difficult to classify because one is too blurry and the other's contour is a little bit complex.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Priority      		| Priority   									| 
| Roundabout     			| Roundabout										|
| Yield					| Yield											|
| Children Crossing	      		| Children Crossing					 				|
| Ahead Only			| Ahead Only      							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. I expected that my model should be unable to predict at least one of them due to the model's accuracy on the test set is 0.979. It's probably because I chose the five traffic signs that my model happened to be good at. In the future work, I would like to collect more real world data to validate my model.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

Below are bar charts showing the top 5 softmax probabilities for the five new images:

![alt text][image12]
![alt text][image13]
![alt text][image14]
![alt text][image15]
![alt text][image16]

The model was quite certain about 4 out of the 5 traffic signs, but not so sure on the 4th traffic sign (34% certainty). 

