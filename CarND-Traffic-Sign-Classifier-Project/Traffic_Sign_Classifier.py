
# coding: utf-8

# # Self-Driving Car Engineer Nanodegree
# 
# ## Deep Learning
# 
# ## Project: Build a Traffic Sign Recognition Classifier
# 
# In this notebook, a template is provided for you to implement your functionality in stages, which is required to successfully complete this project. If additional code is required that cannot be included in the notebook, be sure that the Python code is successfully imported and included in your submission if necessary. 
# 
# > **Note**: Once you have completed all of the code implementations, you need to finalize your work by exporting the iPython Notebook as an HTML document. Before exporting the notebook to html, all of the code cells need to have been run so that reviewers can see the final implementation and output. You can then export the notebook by using the menu above and navigating to  \n",
#     "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission. 
# 
# In addition to implementing code, there is a writeup to complete. The writeup should be completed in a separate file, which can be either a markdown file or a pdf document. There is a [write up template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) that can be used to guide the writing process. Completing the code template and writeup template will cover all of the [rubric points](https://review.udacity.com/#!/rubrics/481/view) for this project.
# 
# The [rubric](https://review.udacity.com/#!/rubrics/481/view) contains "Stand Out Suggestions" for enhancing the project beyond the minimum requirements. The stand out suggestions are optional. If you decide to pursue the "stand out suggestions", you can include the code in this Ipython notebook and also discuss the results in the writeup file.
# 
# 
# >**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut. In addition, Markdown cells can be edited by typically double-clicking the cell to enter edit mode.

# ---
# ## Step 0: Load The Data

# In[1]:


# Load pickled data
import pickle

training_file = "./traffic-signs-data/train.p"
validation_file= "./traffic-signs-data/valid.p"
testing_file = "./traffic-signs-data/test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

print("X_valid shape:", X_valid.shape)
print("y_valid shape:", y_valid.shape)

print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)


# ---
# 
# ## Step 1: Dataset Summary & Exploration
# 
# The pickled data is a dictionary with 4 key/value pairs:
# 
# - `'features'` is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
# - `'labels'` is a 1D array containing the label/class id of the traffic sign. The file `signnames.csv` contains id -> name mappings for each id.
# - `'sizes'` is a list containing tuples, (width, height) representing the original width and height the image.
# - `'coords'` is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image. **THESE COORDINATES ASSUME THE ORIGINAL IMAGE. THE PICKLED DATA CONTAINS RESIZED VERSIONS (32 by 32) OF THESE IMAGES**
# 
# Complete the basic data summary below. Use python, numpy and/or pandas methods to calculate the data summary rather than hard coding the results. For example, the [pandas shape method](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.shape.html) might be useful for calculating some of the summary results. 

# ### Provide a Basic Summary of the Data Set Using Python, Numpy and/or Pandas

# In[2]:


### Use python, pandas or numpy methods rather than hard coding the results
import numpy as np

# TODO: Number of training examples
n_train = len(X_train)

# TODO: Number of validation examples
n_validation = len(X_valid)

# TODO: Number of testing examples.
n_test = len(X_test)

# TODO: What's the shape of an traffic sign image?
image_shape = X_train[0].shape

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(np.unique(y_train))

print("Number of training examples =", n_train)
print("Number of validation examples =", n_validation) #Valid set should be 20-25% of train set
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)


# ### Include an exploratory visualization of the dataset

# Visualize the German Traffic Signs Dataset using the pickled file(s). This is open ended, suggestions include: plotting traffic sign images, plotting the count of each sign, etc. 
# 
# The [Matplotlib](http://matplotlib.org/) [examples](http://matplotlib.org/examples/index.html) and [gallery](http://matplotlib.org/gallery.html) pages are a great resource for doing visualizations in Python.
# 
# **NOTE:** It's recommended you start with something simple first. If you wish to do more, come back to it after you've completed the rest of the sections. It can be interesting to look at the distribution of classes in the training, validation and test set. Is the distribution the same? Are there more examples of some classes than others?

# In[4]:


### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.
import matplotlib.pyplot as plt
import cv2
get_ipython().run_line_magic('matplotlib', 'inline')

import csv
# Load the id -> name mapping from signnames.csv:
with open('signnames.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    id2name = {int(row['ClassId']): row['SignName'] for row in reader}
    

import random
# randomly plot 10 images with 5 in a row
fig, axes = plt.subplots(2, 5, figsize=(15, 7))
for row in axes:
    for axis in row:
        idx = random.randint(0, n_train - 1)
        img, name = X_train[idx, ...], id2name[y_train[idx]]           
        axis.imshow(img)
        axis.set_title(name)

plt.show()    


# In[5]:


# Plot number of images per class type
unique_train, counts_train = np.unique(y_train, return_counts=True)
plt.bar(unique_train, counts_train)
plt.grid()
plt.title("Train Dataset Sign Counts")
plt.show()

unique_valid, counts_valid = np.unique(y_valid, return_counts=True)
plt.bar(unique_valid, counts_valid)
plt.grid()
plt.title("Valid Dataset Sign Counts")
plt.show()

unique_test, counts_test = np.unique(y_test, return_counts=True)
plt.bar(unique_test, counts_test)
plt.grid()
plt.title("Test Dataset Sign Counts")
plt.show()


# In[6]:


# Find the most and least common image sets
from collections import Counter

c = Counter(y_train)
print('Most common five:')
for cid, cnt in c.most_common(5):
    print('  {:.2f}%, {} images --- {}'.format(cnt / n_train * 100, cnt, id2name[cid]))

print('\nLeast common five:')
for cid, cnt in c.most_common()[-5:]:
    print('  {:.2f}%, {} images --- {}'.format(cnt / n_train * 100, cnt, id2name[cid]))


# ----
# 
# ## Step 2: Design and Test a Model Architecture
# 
# Design and implement a deep learning model that learns to recognize traffic signs. Train and test your model on the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).
# 
# The LeNet-5 implementation shown in the [classroom](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) at the end of the CNN lesson is a solid starting point. You'll have to change the number of classes and possibly the preprocessing, but aside from that it's plug and play! 
# 
# With the LeNet-5 solution from the lecture, you should expect a validation set accuracy of about 0.89. To meet specifications, the validation set accuracy will need to be at least 0.93. It is possible to get an even higher accuracy, but 0.93 is the minimum for a successful project submission. 
# 
# There are various aspects to consider when thinking about this problem:
# 
# - Neural network architecture (is the network over or underfitting?)
# - Play around preprocessing techniques (normalization, rgb to grayscale, etc)
# - Number of examples per label (some have more than others).
# - Generate fake data.
# 
# Here is an example of a [published baseline model on this problem](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). It's not required to be familiar with the approach used in the paper but, it's good practice to try to read papers like these.

# ### Pre-process the Data Set (normalization, grayscale, etc.)

# Minimally, the image data should be normalized so that the data has mean zero and equal variance. For image data, `(pixel - 128)/ 128` is a quick way to approximately normalize the data and can be used in this project. 
# 
# Other pre-processing steps are optional. You can try different techniques to see if it improves performance. 
# 
# Use the code cell (or multiple code cells, if necessary) to implement the first step of your project.

# In[7]:


#Save original
X_train_rgb = X_train
X_test_rgb = X_test
X_valid_rgb = X_valid

X_train_hist = X_train
X_test_hist = X_test
X_valid_hist = X_valid


# In[8]:


#Histogram equalization
def HistogramEqualization(image):
    temp_image = image.copy()
    temp_image[:,:,0] = cv2.equalizeHist(image[:,:,0])
    temp_image[:,:,1] = cv2.equalizeHist(image[:,:,1])
    temp_image[:,:,2] = cv2.equalizeHist(image[:,:,2])
    return temp_image

for i in range(len(X_train)):
    X_train_hist[i] = HistogramEqualization(X_train[i])
    
for i in range(len(X_test)):
    X_test_hist[i] = HistogramEqualization(X_test[i])
    
for i in range(len(X_valid)):
    X_valid_hist[i] = HistogramEqualization(X_valid[i])    

# randomly plot 10 images with 5 in a row
fig, axes = plt.subplots(2, 5, figsize=(15, 7))
for row in axes:
    for axis in row:
        idx = random.randint(0, n_train - 1)
        img, name = X_train_hist[idx, ...], id2name[y_train[idx]]         
        axis.imshow(img)
        axis.set_title(name)

print("Histogram Equalization Output")
plt.show() 


# In[9]:


### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include 
### converting to grayscale, etc.
### Feel free to use as many code cells as needed.

import tensorflow as tf
from tensorflow.contrib.layers import flatten
from math import ceil
from sklearn.utils import shuffle

# Convert to grayscale

X_train_gray = np.sum(X_train_hist/3, axis=3, keepdims=True)

X_test_gray = np.sum(X_test_hist/3, axis=3, keepdims=True)

X_valid_gray = np.sum(X_valid_hist/3, axis=3, keepdims=True)

print("Train RGB",X_train_rgb.shape)
print("Train Gray",X_train_gray.shape)

print("Valid RGB",X_valid_rgb.shape)
print("Valid Gray",X_valid_gray.shape)

print("Test RGB",X_test_rgb.shape)
print("Test Gray",X_test_gray.shape)


# In[10]:


#Apply grayscaling
X_train = X_train_gray
X_test = X_test_gray
X_valid = X_valid_gray
print("Grayscaling done !!!")


# In[11]:


# randomly plot 10 images with 5 in a row
fig, axes = plt.subplots(2, 5, figsize=(15, 7))
for row in axes:
    for axis in row:
        idx = random.randint(0, n_train - 1)
        img, name = X_train[idx, ...].squeeze(), id2name[y_train[idx]]    
        axis.imshow(img, cmap='gray')
        axis.set_title(name)

plt.show()  


# In[12]:


# Find the original mean of datasets
print("Train original mean",np.mean(X_train))
print("Valid original mean",np.mean(X_valid))
print("Test original mean",np.mean(X_test))


# In[13]:


## Normalize the train and test datasets to (-1,1)
X_train_normalized = (X_train - 128)/128 
X_valid_normalized = (X_valid - 128)/128 
X_test_normalized = (X_test - 128)/128

print("Normalization done !!!")

# Find the new mean of datasets after normalization
print("Train normalised mean",np.mean(X_train_normalized))
print("Valid normalised mean",np.mean(X_valid_normalized))
print("Test normalised mean",np.mean(X_test_normalized))

fig, axs = plt.subplots(1,2, figsize=(10, 7))
axs = axs.ravel()
axs[0].set_title('Original')
axs[0].imshow(X_train[10000].squeeze(), cmap='gray')

axs[1].set_title('Normalized')
axs[1].imshow(X_train_normalized[10000].squeeze(), cmap='gray')


# In[14]:


### Pre-processing of image data
### Random Translation of images
import cv2

def randomTranslate(img):
    rows,cols,_ = img.shape
    
    # Translation limits in x and y directions
    px = 2
    dx,dy = np.random.randint(-px,px,2)

    M = np.float32([[1,0,dx],[0,1,dy]])
    dst = cv2.warpAffine(img,M,(cols,rows))
    
    dst = dst[:,:,np.newaxis]
    
    return dst

testImage = X_train_normalized[10000]

randomTranslate_Output = randomTranslate(testImage)

fig, axs = plt.subplots(1,2, figsize=(10, 7))
axs[0].imshow(testImage.squeeze(), cmap='gray')
axs[0].set_title('Original')

axs[1].imshow(randomTranslate_Output.squeeze(), cmap='gray')
axs[1].set_title('Translated')


# In[15]:


### Random Scaling of images
def randomScale(img):   
    rows,cols,_ = img.shape

    # Scaling limits
    px = np.random.randint(-2,2)

    # Destination points
    pts1 = np.float32([[px,px],[rows-px,px],[px,cols-px],[rows-px,cols-px]])

    # Source points
    pts2 = np.float32([[0,0],[rows,0],[0,cols],[rows,cols]])

    M = cv2.getPerspectiveTransform(pts1,pts2)

    dst = cv2.warpPerspective(img,M,(rows,cols))
    
    dst = dst[:,:,np.newaxis]
    
    return dst

randomScale_Output = randomScale(testImage)
    
fig, axs = plt.subplots(1,2, figsize=(10, 7))

axs[0].imshow(testImage.squeeze(), cmap='gray')
axs[0].set_title('Original')

axs[1].imshow(randomScale_Output.squeeze(), cmap='gray')
axs[1].set_title('Scaled')


# In[18]:


### Random Warping of images
def randomWarp(img):
    
    rows,cols,_ = img.shape

    # Warping limits
    rndx = np.random.rand(3) - 0.5
    rndx *= cols * 0.06   
    rndy = np.random.rand(3) - 0.5
    rndy *= rows * 0.06

    # 3 starting points for transform and 1/4 away from all edges
    x1 = cols/4
    x2 = 3*cols/4
    y1 = rows/4
    y2 = 3*rows/4

    pts1 = np.float32([[y1,x1],
                       [y2,x1],
                       [y1,x2]])
    pts2 = np.float32([[y1+rndy[0],x1+rndx[0]],
                       [y2+rndy[1],x1+rndx[1]],
                       [y1+rndy[2],x2+rndx[2]]])

    M = cv2.getAffineTransform(pts1,pts2)

    dst = cv2.warpAffine(img,M,(cols,rows))
    
    dst = dst[:,:,np.newaxis]
    
    return dst

randomWarp_output = randomWarp(testImage)

fig, axs = plt.subplots(1,2, figsize=(10, 7))

axs[0].imshow(testImage.squeeze(), cmap='gray')
axs[0].set_title('Original')

axs[1].imshow(randomWarp_output.squeeze(), cmap='gray')
axs[1].set_title('Warped')


# In[19]:


### Randomly changing brightness images

def randomBrightness(img):
    # Change range to (0,2)
    new_image = img + 1.0   
    img_max_value = max(new_image.flatten())
    max_coef = 2.0/img_max_value
    min_coef = max_coef - 0.1
    coef = np.random.uniform(min_coef, max_coef)
    dst = new_image * coef - 1.0
    return dst

randomBrightness_output = randomBrightness(testImage)

fig, axs = plt.subplots(1,2, figsize=(10, 7))

axs[0].imshow(testImage.squeeze(), cmap='gray')
axs[0].set_title('Original')

axs[1].imshow(randomBrightness_output.squeeze(), cmap='gray')
axs[1].set_title('Brightness adjusted')


# In[20]:


#Before augumentation
unique_train, counts_train = np.unique(y_train, return_counts=True)
plt.bar(unique_train, counts_train)
plt.grid()
plt.title("Train Dataset Sign Counts")
plt.show()

print(np.bincount(y_train))
print("Min samples for any label:", min(np.bincount(y_train)))
print("Max samples for any label:", max(np.bincount(y_train)))

print('X, y shapes Before:', X_train_normalized.shape, y_train.shape)


# In[21]:


# # #DONT RUNNN THISSSS UNLESS YOU WANT TO AUGUMENT NEW DATA
# # AUGUMENTED DATA SAVED IN PICKLE FILES (SEE NEXT SECTION)
# # ############################################
# input_indices = []
# output_indices = []

# for class_n in range(n_classes):
#     print(class_n, ': ', end='')
#     class_indices = np.where(y_train == class_n)
#     n_samples = len(class_indices[0])
#     if n_samples < 1000:
#         for i in range(1000 - n_samples):
#             input_indices.append(class_indices[0][i%n_samples])
#             output_indices.append(X_train_normalized.shape[0])
#             new_img = X_train_normalized[class_indices[0][i % n_samples]]
#             new_img = randomTranslate(randomScale(randomWarp(randomBrightness(new_img))))
#             X_train_normalized = np.concatenate((X_train_normalized, [new_img]), axis=0)
#             y_train = np.concatenate((y_train, [class_n]), axis=0)
#             if i % 50 == 0:
#                 print('|', end='')
#             elif i % 10 == 0:
#                 print('-',end='')
#     print('')
            
# ##################################################
# # print('X, y shapes AFTER:', X_train_normalized.shape, y_train.shape)
# # pickle.dump( X_train_normalized, open( "X_train_normalized.p", "wb" ) )
# # pickle.dump( y_train, open( "y_train.p", "wb" ) )
# # pickle.dump( input_indices, open( "input_indices.p", "wb" ) )
# # pickle.dump( output_indices, open( "output_indices.p", "wb" ) )
# # print("Saved Data")


# In[22]:


### Load saved augumented data (Augument data only once and save the new dataset as a pickle file)
X_train_normalized = pickle.load( open( "X_train_normalized.p", "rb" ) )
y_train = pickle.load( open( "y_train.p", "rb" ) )
input_indices = pickle.load( open( "input_indices.p", "rb" ) )
output_indices = pickle.load( open( "output_indices.p", "rb" ) )
print("Loaded Data")

print('X, y shapes AFTER:', X_train_normalized.shape, y_train.shape)

# Display comparisons of 5 random augmented data points
choices = list(range(len(input_indices)))
picks = []
for i in range(5):
    rnd_index = np.random.randint(low=0,high=len(choices))
    picks.append(choices.pop(rnd_index))
fig, axs = plt.subplots(1,5, figsize=(15, 6))
fig.subplots_adjust(hspace = .2, wspace=.001)
axs = axs.ravel()
for i in range(5):
    image = X_train_normalized[input_indices[picks[i]]].squeeze()
    axs[i].imshow(image, cmap = 'gray')
    axs[i].set_title(y_train[input_indices[picks[i]]])


# In[23]:


#After augumentation
unique_train, counts_train = np.unique(y_train, return_counts=True)
plt.bar(unique_train, counts_train)
plt.grid()
plt.title("Train Dataset Sign Counts")
plt.show()

print(np.bincount(y_train))
print("Min samples for any label:", min(np.bincount(y_train)))
print("Max samples for any label:", max(np.bincount(y_train)))


# In[24]:


## Shuffle the training dataset 

from sklearn.utils import shuffle

X_train_normalized, y_train = shuffle(X_train_normalized, y_train)

print("Training set SHUFFLED !!!")


# In[25]:


## Split validation dataset off from training dataset

from sklearn.model_selection import train_test_split

X_train, X_validation, y_train, y_validation = train_test_split(X_train_normalized, y_train, test_size=0.20, random_state=42)

print("Old X_train/X_train normalized size:",len(X_train_normalized))
print("New X_train size:",len(X_train))
print("X_validation size:",len(X_validation))


# # Model Architecture

# In[26]:


### Define your architecture here.
### Feel free to use as many code cells as needed.


# In[27]:


import tensorflow as tf

EPOCHS = 150
BATCH_SIZE = 75

print('done')


# In[28]:


from tensorflow.contrib.layers import flatten

def TrafficSignClassifier(x):    
    # Hyperparameters
    mu = 0
    sigma = 0.1
    
  
    # TODO: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    W1 = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean = mu, stddev = sigma), name="W1")
    x = tf.nn.conv2d(x, W1, strides=[1, 1, 1, 1], padding='VALID')
    b1 = tf.Variable(tf.zeros(6), name="b1")
    x = tf.nn.bias_add(x, b1)
    print("Layer 1 shape:",x.get_shape())

    # TODO: Activation.
    x = tf.nn.relu(x)
    
    # TODO: Max Pooling. Input = 28x28x6. Output = 14x14x6.
    x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')    
    
    layer1 = x
    
    # Inception Module
    # TODO: Layer 2: Convolutional. Input = 14x14x6. Filters = 5x5x16, 3x3x16, 1x1x16. Output = 10x10x16, 12x12x16, 14x14x16
    W2_1 = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma), name="W2_1")
    x_1 = tf.nn.conv2d(x, W2_1, strides=[1, 1, 1, 1], padding='VALID')
    b2 = tf.Variable(tf.zeros(16), name="b2")
    x_1 = tf.nn.bias_add(x_1, b2)
    
    W2_2 = tf.Variable(tf.truncated_normal(shape=(3, 3, 6, 16), mean = mu, stddev = sigma), name="W2_2")
    x_2 = tf.nn.conv2d(x, W2_2, strides=[1, 1, 1, 1], padding='VALID')
    b2 = tf.Variable(tf.zeros(16), name="b2")
    x_2 = tf.nn.bias_add(x_2, b2)
    
    W2_3 = tf.Variable(tf.truncated_normal(shape=(1, 1, 6, 16), mean = mu, stddev = sigma), name="W2_3")
    x_3 = tf.nn.conv2d(x, W2_3, strides=[1, 1, 1, 1], padding='VALID')
    b2 = tf.Variable(tf.zeros(16), name="b2")
    x_3 = tf.nn.bias_add(x_3, b2)
                     
    # TODO: Activation.
    x_1 = tf.nn.relu(x_1)
    x_2 = tf.nn.relu(x_2)
    x_3 = tf.nn.relu(x_3)

    # TODO: Average Pooling. Input = 10x10x16, 12x12x16, 14x14x16. Output = 5x5x16, 6x6x16, 7x7x16
    x_1 = tf.nn.avg_pool(x_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')     
    x_2 = tf.nn.avg_pool(x_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    x_3 = tf.nn.avg_pool(x_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    layer2_1 = x_1
    layer2_2 = x_2
    layer2_3 = x_3
    
    # Inception Module
    # TODO: Layer 3: Convolutional. Input = 5x5x16, 6x6x16, 7x7x16. Filters = 3x3x100. Output = 3x3x100, 4x4x100, 5x5x100
    W3_1 = tf.Variable(tf.truncated_normal(shape=(3, 3, 16, 100), mean = mu, stddev = sigma), name="W3_1")
    x_1 = tf.nn.conv2d(x_1, W3_1, strides=[1, 1, 1, 1], padding='VALID')
    b3 = tf.Variable(tf.zeros(100), name="b3")
    x_1 = tf.nn.bias_add(x_1, b3)
    
    W3_2 = tf.Variable(tf.truncated_normal(shape=(3, 3, 16, 100), mean = mu, stddev = sigma), name="W3_2")
    x_2 = tf.nn.conv2d(x_2, W3_2, strides=[1, 1, 1, 1], padding='VALID')
    b3 = tf.Variable(tf.zeros(100), name="b3")
    x_2 = tf.nn.bias_add(x_2, b3)
    
    W3_3 = tf.Variable(tf.truncated_normal(shape=(3, 3, 16, 100), mean = mu, stddev = sigma), name="W3_3")
    x_3 = tf.nn.conv2d(x_3, W3_3, strides=[1, 1, 1, 1], padding='VALID')
    b3 = tf.Variable(tf.zeros(100), name="b3")
    x_3 = tf.nn.bias_add(x_3, b3)
                     
    # TODO: Activation.
    x_1 = tf.nn.relu(x_1)  
    x_2 = tf.nn.relu(x_2)
    x_3 = tf.nn.relu(x_3)
    
    layer3_1 = x_1
    layer3_2 = x_2
    layer3_3 = x_3

    # TODO: Flatten. Input = 3x3x100. Output = 900.
    layer2flat_1 = flatten(layer3_1)
    print("layer2flat_1 shape:",layer2flat_1.get_shape())
    
    # TODO: Flatten. Input = 4x4x100. Output = 1600.
    layer2flat_2 = flatten(layer3_2)
    print("layer2flat_2 shape:",layer2flat_2.get_shape())
    
    # TODO: Flatten. Input = 5x5x100. Output = 2500.
    layer2flat_3 = flatten(layer3_3)
    print("layer2flat_3 shape:",layer2flat_3.get_shape())

    
    # Concat layer2flat and x. Input = 900+1600+2500. Output = 5000
    x = tf.concat([layer2flat_1, layer2flat_2, layer2flat_3], 1)
    print("x shape:",x.get_shape())
    
    #TODO: Activation.
    x = tf.nn.relu(x)
    
    # Dropout
    x = tf.nn.dropout(x, keep_prob)
    
    # TODO: Layer 4: Fully Connected. Input = 5000. Output = 1000.
    W4 = tf.Variable(tf.truncated_normal(shape=(5000, 1000), mean = mu, stddev = sigma), name="W4")
    b4 = tf.Variable(tf.zeros(1000), name="b4")    
    x = tf.add(tf.matmul(x, W4), b4)
    
    #TODO: Activation.
    x = tf.nn.relu(x)
    
    # Dropout
    x = tf.nn.dropout(x, keep_prob)

    #TODO: Layer 5: Fully Connected. Input = 1000. Output = 43.
    W5 = tf.Variable(tf.truncated_normal(shape=(1000, 43), mean = mu, stddev = sigma))
    b5 = tf.Variable(tf.zeros(43)) 
    logits = tf.add(tf.matmul(x, W5), b5)
    
    return logits

print('done')


# In[29]:


tf.reset_default_graph() 

x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
keep_prob = tf.placeholder(tf.float32) # probability to keep units
one_hot_y = tf.one_hot(y, 43)

print('done')


# ### Train, Validate and Test the Model

# A validation set can be used to assess how well the model is performing. A low accuracy on the training and validation
# sets imply underfitting. A high accuracy on the training set but low accuracy on the validation set implies overfitting.

# In[30]:


### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected, 
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.


# In[31]:


rate = 0.0009

logits = TrafficSignClassifier(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)


# In[32]:


correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

print('done')


# In[34]:


### DONT RUN THIS UNLESS YOU WANT TO RETRAIN THE MODEL
# ## Train the model on train dataset
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     num_examples = len(X_train)
    
#     print("Training...")
#     print()
#     validation_accuracy_figure = []
#     for i in range(EPOCHS):
#         X_train, y_train = shuffle(X_train, y_train)
#         for offset in range(0, num_examples, BATCH_SIZE):
#             end = offset + BATCH_SIZE
#             batch_x, batch_y = X_train[offset:end], y_train[offset:end]
#             sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})
            
#         validation_accuracy = evaluate(X_validation, y_validation)
#         validation_accuracy_figure.append(validation_accuracy)
#         print("EPOCH {} ...".format(i+1))
#         print("Validation Accuracy = {:.3f}".format(validation_accuracy))
#         print()
        
#     saver.save(sess, './Model/Traffic_Sign_Classifier_Model')
#     print("Model saved")

# # Plot a graph of validation accuracy over 150 epochs
# plt.plot(validation_accuracy_figure)
# plt.title("Validation Accuracy")
# plt.show()


# In[35]:


# Evaluate the accuracy of the model on the test dataset

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver2 = tf.train.import_meta_graph('./Model/Traffic_Sign_Classifier_Model.meta')
    saver2.restore(sess, "./Model/Traffic_Sign_Classifier_Model")
    test_accuracy = evaluate(X_test_normalized, y_test)
    print("Test Set Accuracy = {:.3f}".format(test_accuracy))


# ---
# 
# ## Step 3: Test a Model on New Images
# 
# To give yourself more insight into how your model is working, download at least five pictures of German traffic signs from the web and use your model to predict the traffic sign type.
# 
# You may find `signnames.csv` useful as it contains mappings from the class id (integer) to the actual sign name.

# ### Load and Output the Images

# In[36]:


### Load the images and plot them here.
### Feel free to use as many code cells as needed.


# In[37]:


# Display test images
import glob
import matplotlib.image as mpimg

fig, axs = plt.subplots(2,6, figsize=(10, 7))
fig.subplots_adjust(hspace = .2, wspace=.001)
axs = axs.ravel()

my_images = []

for i, img in enumerate(glob.glob('./Test-images/*.png')):
    image = cv2.imread(img)
    axs[i].axis('off')
    axs[i].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    my_images.append(image)

my_images = np.asarray(my_images)
my_images_gry = np.sum(my_images/3, axis=3, keepdims=True)
my_images_normalized = (my_images_gry - 128)/128 

print(my_images_normalized.shape)


# ### Predict the Sign Type for Each Image

# In[38]:


### Run the predictions here and use the model to output the prediction for each image.
### Make sure to pre-process the images with the same pre-processing pipeline used earlier.
### Feel free to use as many code cells as needed.

softmax_logits = tf.nn.softmax(logits)
top_k = tf.nn.top_k(softmax_logits, k=3)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.import_meta_graph('./Model/Traffic_Sign_Classifier_Model.meta')
    saver.restore(sess, "./Model/Traffic_Sign_Classifier_Model")
    my_softmax_logits = sess.run(softmax_logits, feed_dict={x: my_images_normalized, keep_prob: 1.0})
    my_top_k = sess.run(top_k, feed_dict={x: my_images_normalized, keep_prob: 1.0})

    
    fig, axs = plt.subplots(len(my_images),4, figsize=(16, 20))
    fig.subplots_adjust(hspace = .4, wspace=.2)
    axs = axs.ravel()

    for i, image in enumerate(my_images):
        axs[4*i].axis('off')
        axs[4*i].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axs[4*i].set_title('Input')
        guess1 = my_top_k[1][i][0]
        index1 = np.argwhere(y_validation == guess1)[0]
        axs[4*i+1].axis('off')
        axs[4*i+1].imshow(X_validation[index1].squeeze(), cmap='gray')
        axs[4*i+1].set_title('Top guess: {} ({:.0f}%)'.format(guess1, 100*my_top_k[0][i][0]))
        guess2 = my_top_k[1][i][1]
        index2 = np.argwhere(y_validation == guess2)[0]
        axs[4*i+2].axis('off')
        axs[4*i+2].imshow(X_validation[index2].squeeze(), cmap='gray')
        axs[4*i+2].set_title('2nd guess: {} ({:.0f}%)'.format(guess2, 100*my_top_k[0][i][1]))
        guess3 = my_top_k[1][i][2]
        index3 = np.argwhere(y_validation == guess3)[0]
        axs[4*i+3].axis('off')
        axs[4*i+3].imshow(X_validation[index3].squeeze(), cmap='gray')
        axs[4*i+3].set_title('3rd guess: {} ({:.0f}%)'.format(guess3, 100*my_top_k[0][i][2]))


# ### Analyze Performance

# In[39]:


### Calculate the accuracy for these 5 new images. 
### For example, if the model predicted 1 out of 5 signs correctly, it's 20% accurate on these new images.


# ### Output Top 5 Softmax Probabilities For Each Image Found on the Web

# For each of the new images, print out the model's softmax probabilities to show the **certainty** of the model's predictions (limit the output to the top 5 probabilities for each image). [`tf.nn.top_k`](https://www.tensorflow.org/versions/r0.12/api_docs/python/nn.html#top_k) could prove helpful here. 
# 
# The example below demonstrates how tf.nn.top_k can be used to find the top k predictions for each image.
# 
# `tf.nn.top_k` will return the values and indices (class ids) of the top k predictions. So if k=3, for each sign, it'll return the 3 largest probabilities (out of a possible 43) and the correspoding class ids.
# 
# Take this numpy array as an example. The values in the array represent predictions. The array contains softmax probabilities for five candidate images with six possible classes. `tf.nn.top_k` is used to choose the three classes with the highest probability:
# 
# ```
# # (5, 6) array
# a = np.array([[ 0.24879643,  0.07032244,  0.12641572,  0.34763842,  0.07893497,
#          0.12789202],
#        [ 0.28086119,  0.27569815,  0.08594638,  0.0178669 ,  0.18063401,
#          0.15899337],
#        [ 0.26076848,  0.23664738,  0.08020603,  0.07001922,  0.1134371 ,
#          0.23892179],
#        [ 0.11943333,  0.29198961,  0.02605103,  0.26234032,  0.1351348 ,
#          0.16505091],
#        [ 0.09561176,  0.34396535,  0.0643941 ,  0.16240774,  0.24206137,
#          0.09155967]])
# ```
# 
# Running it through `sess.run(tf.nn.top_k(tf.constant(a), k=3))` produces:
# 
# ```
# TopKV2(values=array([[ 0.34763842,  0.24879643,  0.12789202],
#        [ 0.28086119,  0.27569815,  0.18063401],
#        [ 0.26076848,  0.23892179,  0.23664738],
#        [ 0.29198961,  0.26234032,  0.16505091],
#        [ 0.34396535,  0.24206137,  0.16240774]]), indices=array([[3, 0, 5],
#        [0, 1, 4],
#        [0, 5, 1],
#        [1, 3, 5],
#        [1, 4, 3]], dtype=int32))
# ```
# 
# Looking just at the first row we get `[ 0.34763842,  0.24879643,  0.12789202]`, you can confirm these are the 3 largest probabilities in `a`. You'll also notice `[3, 0, 5]` are the corresponding indices.

# In[40]:


### Print out the top five softmax probabilities for the predictions on the German traffic sign images found on the web. 
### Feel free to use as many code cells as needed.

fig, axs = plt.subplots(12,2, figsize=(9, 19))
axs = axs.ravel()

for i in range(len(my_softmax_logits)*2):
    if i%2 == 0:
        axs[i].axis('off')
        axs[i].imshow(cv2.cvtColor(my_images[i//2], cv2.COLOR_BGR2RGB))
    else:
        axs[i].bar(np.arange(n_classes), my_softmax_logits[(i-1)//2]) 
        axs[i].set_ylabel('Softmax Prob')       

