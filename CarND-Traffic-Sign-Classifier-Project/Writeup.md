# **Traffic Sign Classifier CNN Writeup** 

**Objectives:**
1. Load the data set 
2. Explore, summarize and visualize the data set
3. Design, train and test a model architecture
4. Use the model to make predictions on new images
5. Analyze the softmax probabilities of the new images
---
## Explaination

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic signs data set:

* The size of training set - 34799 images
* The size of the validation set - 4410 images
* The size of test set - 12630 images
* The shape of a traffic sign image - (32, 32, 3)
* The number of unique classes/labels in the data set - 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set showing few images of the dataset:
![data](https://github.com/AllenMendes/Traffic-Sign-Classifier-CNN/blob/master/CarND-Traffic-Sign-Classifier-Project/Downloads/data.png)

Following are bar chart representations of the distribution of traffic signs based on their classes/labels in the training, validation and test datasets:

![train1](https://github.com/AllenMendes/Traffic-Sign-Classifier-CNN/blob/master/CarND-Traffic-Sign-Classifier-Project/Downloads/train1.png)

![valid1](https://github.com/AllenMendes/Traffic-Sign-Classifier-CNN/blob/master/CarND-Traffic-Sign-Classifier-Project/Downloads/valid1.png)

![test1](https://github.com/AllenMendes/Traffic-Sign-Classifier-CNN/blob/master/CarND-Traffic-Sign-Classifier-Project/Downloads/test1.png)


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. 

As the traffic signs were captured in different lighting conditions, I decided to perform a histogram equalization on all the images to even out the spread of light intensity over the image thereby improving the overall contrast of the images.

**Histogram Equalization output:**

![hist](https://github.com/AllenMendes/Traffic-Sign-Classifier-CNN/blob/master/CarND-Traffic-Sign-Classifier-Project/Downloads/hist.png)

Next I converted all the images into their grayscale equivalent as I didn't want the model to learn the color features of the traffic signs (as shapes are more important for traffic signs).

**Grayscale output:**

![gray](https://github.com/AllenMendes/Traffic-Sign-Classifier-CNN/blob/master/CarND-Traffic-Sign-Classifier-Project/Downloads/gray.png)

Then I normalized all the images to a new range of (-1,1). Below is the output for one of the images in the dataset. 

**Normalization output:**

![norm](https://github.com/AllenMendes/Traffic-Sign-Classifier-CNN/blob/master/CarND-Traffic-Sign-Classifier-Project/Downloads/norm.png)

As observed from the distribution charts of the number images per class/label, few classes have far more images than the other classes. This may make the model to lean towards false positives or the classes/labels having more images just because of the non unifrom distribution of images over all classes/labels. Hence to augment the dataset, I performed random scaling, translation, warping and brightness change to the images of the classes/labels which had less than 1000 images so that all classess/labels contain atleast 1000 images each. Following are the outputs of the pre-processing techniques I used to augment the dataset:

**Random Scaling:**

![scale](https://github.com/AllenMendes/Traffic-Sign-Classifier-CNN/blob/master/CarND-Traffic-Sign-Classifier-Project/Downloads/scaled.png)

**Random Translation:**

![trans](https://github.com/AllenMendes/Traffic-Sign-Classifier-CNN/blob/master/CarND-Traffic-Sign-Classifier-Project/Downloads/translate.png)

**Random Warping:**

![warp](https://github.com/AllenMendes/Traffic-Sign-Classifier-CNN/blob/master/CarND-Traffic-Sign-Classifier-Project/Downloads/warp.png)

**Random Brightness change:**

![bright](https://github.com/AllenMendes/Traffic-Sign-Classifier-CNN/blob/master/CarND-Traffic-Sign-Classifier-Project/Downloads/bright.png)

***Dataset size and distribution before data augmentation- 34799 images***

![before](https://github.com/AllenMendes/Traffic-Sign-Classifier-CNN/blob/master/CarND-Traffic-Sign-Classifier-Project/Downloads/before_aug.png)

***Dataset size and distribution after data augmentation- 51690 images***

![after](https://github.com/AllenMendes/Traffic-Sign-Classifier-CNN/blob/master/CarND-Traffic-Sign-Classifier-Project/Downloads/after_aug.png)


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

After considering multiple options for my CNN architecture, I decided on a multi-layer deep inception block (Multimodal) based approach for my final model architecture. At each layer, multiple filters would learn different features about a single image to generalize the model. This helped me achieve an ***accuracy of 96.6%*** over the given dataset.

## CNN architecture

![CNN](https://github.com/AllenMendes/Traffic-Sign-Classifier-CNN/blob/master/CarND-Traffic-Sign-Classifier-Project/Traffic-Classifier-CNN.JPG)

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   							| 
| Convolution 5x5x6     	| 1x1 stride, Valid padding, Output 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  Valid padding, Outputs 14x14x6 				|
| Convolutions 5x5x16, 3x3x16, 1x1x16	    | 1x1 stride, Valid padding, Outputs 10x10x16, 12x12x16, 14x14x16 |
| RELU					|												|
| Average pooling	      	| 2x2 stride,  Valid padding, Outputs 14x14x6 				|
| Convolutions 3x3x100 on each of the 3 layers   | 1x1 stride, Valid padding, Outputs 3x3x100, 4x4x100, 5x5x100 |
| RELU					|												|
| Flatten 3 layers and concat in 1 inception block					|			Output: 900+1600+2500 = 5000									|
| RELU					|												|
| Dropout					|					keep_prob = 50%							|
| Fully connected		| 5000 -> 1000        									|
| RELU					|												|
| Dropout					|					keep_prob = 50%							|
| Fully connected		| 1000 -> 43        									|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an the following hyperparameters:
* Optimizer - Adam
* Batch size - 75
* Epochs - 150
* Learning rate - 0.0009

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* Training set accuracy - 100%
* Validation set accuracy - 99.7% 
* Test set accuracy - 96.6%

I started off by trying out the simple Lenet architecture which gave me an accuracy of 94%. After reading multiple research papers, I understood that the model needed for this kind of traffic classifier project doesn't need not be very ellaborate but should be deep enough to accurately pick the various features in the traffic sign images. I observed that by having multiple filters for a single convolution layer made a significant difference but again you don't want to overdo it. Also I tried a combination of max pooling and average pooling between the hidden layers to increase the overall accuracy of the model. At the end I was playing around with different hyperparameters values of batch size and epochs to finally come down to the values mentioned above for the best validation accuracy of 99.7%. At this point I was convinced that the tuning has reached its limits and now would be the time to test my model on the test dataset and ***Voila !!!*** , my first ever CNN model worked beautifully to give me 96.6% accuracy.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I chose the following 12 German traffic signs that I found on the web:

![testimages](https://github.com/AllenMendes/Traffic-Sign-Classifier-CNN/blob/master/CarND-Traffic-Sign-Classifier-Project/Downloads/test_images.png)

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set.

Here are the results of the prediction:

![result](https://github.com/AllenMendes/Traffic-Sign-Classifier-CNN/blob/master/CarND-Traffic-Sign-Classifier-Project/Downloads/result.png)

I was pretty amazed to see that the model accurately guessed all the 12 test images with 100% accuracy. You can see in the predictions (1st, 2nd and 3rd guesses) section, how the model picked out the its guesses for each of the test images from the validation set.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. 

As seen from the above result, the model predicted all the images with 100% accuracy. I even included a 2nd and 3rd guess result based on the softmax probablity values. If you observe closely the below few predictions, you can see that model is trying to co relate shapes of arrow signs, human shaped figures, circular shaped traffic signs, etc. and other such shape related features found in the test images to make its predictions.

![ob](https://github.com/AllenMendes/Traffic-Sign-Classifier-CNN/blob/master/CarND-Traffic-Sign-Classifier-Project/Downloads/ob.JPG)

The high prediction accuracy of 100% might be because these images could be very similar to the ones in the dataset and hence the model was very certain about all the input images. To conclude, below are the softmax probabilities for all of the test images:

![soft](https://github.com/AllenMendes/Traffic-Sign-Classifier-CNN/blob/master/CarND-Traffic-Sign-Classifier-Project/Downloads/soft_result.png)
