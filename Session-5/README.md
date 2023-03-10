# Adding Regularization methods and Modularizing the Code
## Problem Statement:-

You are making 3 versions of your 4th assignment's best model (or pick one from best assignments):
1. Network with Group Normalization
2. Network with Layer Normalization
3. Network with L1 + BN


Create these graphs:
* Graph 1: Training Loss for all 3 models together(Late Assignment Part)
* Graph 2: Test/Validation Loss for all 3 models together
* Graph 3: Training Accuracy for all 3 models together(Late Assignment Part)
* Graph 4: Test/Validation Accuracy for all 3 models together

Find 20 misclassified images for each of the 3 models, and show them as a 5x4 image matrix in 3 separately annotated images. 

## Folder Structure
* Experiments
    * Contains all the files that we experimented for finishing the assignment
* logs
  * Contains text files which has logs and summary and misclassified images for each the model used
  * loss and accuracy graphs
* models
  * contains all model designs
* utils
  * contains all utility methods needed for training and validating model
* model.py
  * Main file which loads all the required methods sequentially just like colab notebook
  * model.py has two dictionaries
    * models -
      * has key and list as value
      * value is list which has three values. first value in list is a unique key which is used to get Model instance
        Second value in the list is a boolean which says is l1 required or not. Third value in the list is a boolean 
        which says is l2 required or not
    * Models - returns model instance
  * For this assignment we are looping through the models dict and selecting each model based on the key from Models dict   
  * Future Enhancement -  model can be selected by providing options through command line and based on the user input 
    appropriate model will be selected for training

# Normalization:-

In image processing, normalization is a process that changes the range of pixel intensity values. 

The normalize is quite simple, it looks for the maximum intensity pixel (we will use a grayscale example here) and a minimum intensity and then will determine a factor that scales the min intensity to black and the max intensity to white. This is applied to every pixel in the image which produces the final result. 

The Basic Formulae of implementaion of normalization can be represented in the following experession:-

![image](https://user-images.githubusercontent.com/51078583/121730596-86b5b200-cb0d-11eb-8d06-898729c46467.png)


There are mainly three types of Normalization techniques we will be discussing:-
* Batch Normalization 
* Layer Normalization 
* Group Normalization

## Batch Normalization:-
It can be considered as the rescaling of image with respect to the channels. 

Mathematically, BN layer transforms each input in the current mini-batch by subtracting the input mean in the current mini-batch and dividing it by the standard deviation.
Below given is the Mathematical implication of the Batch Normalization. 

Pictorial representation of Batch Normalization:-

![image](https://user-images.githubusercontent.com/51078583/121733150-bf0abf80-cb10-11eb-8cce-af1dddad1715.png)

For example:-

![image](https://user-images.githubusercontent.com/51078583/121732567-0775ad80-cb10-11eb-8e29-39a4c143834b.png)


## Layer Normalization:-
Layer normalization normalizes input across the features instead of normalizing input features across the batch dimension in batch normalization.

Pictorial representation of Layer Normalization:-

![image](https://user-images.githubusercontent.com/51078583/121733303-efeaf480-cb10-11eb-879b-15daa9ebeaa6.png)

For example:-

![image](https://user-images.githubusercontent.com/51078583/121732451-e9a84880-cb0f-11eb-984e-27a3d6ffcad1.png)

## Group Normalization:-

As the name suggests, Group Normalization normalizes over group of channels for each training examples. We can say that, Group Norm is in between Instance Norm and Layer Norm.

When we put all the channels into a single group, group normalization becomes Layer normalization. And, when we put each channel into different groups it becomes Instance normalization.

Pictorial representation of Layer Normalization:-

![image](https://user-images.githubusercontent.com/51078583/121733404-11e47700-cb11-11eb-9571-4a09de4c94f4.png)

For example:-

![image](https://user-images.githubusercontent.com/51078583/121732706-2e33e400-cb10-11eb-99f4-26a9ae3f0d4f.png)

    
## Models and their Performance:-
Dropout = 0.03

Epoches = 20

Apart from the three combinations asked in the assignment I have also tried other combinations, please take a look below.

|Normalization|L1 Regularization|	L2 Regularization | Params Count | Best Train Accuracy	|Best Test Accuracy| Link to Logs|
|--|--|--|--|--|--|--|
|Layer Normalization| - | - |43208 |98.91 |99.62|[Layer Norm Logs](https://github.com/NSR9/EVA8/blob/main/Session-5/logs/layer_norm/layer_norm)| 
|Group Normalization| - | - | 7704| 98.72|99.51 |[Group Norm Logs](https://github.com/NSR9/EVA8/blob/main/Session-5/logs/group_norm/group_norm)|
|Batch Normalization| Yes | - |7704 |97.84 |99.35 |[Batch Norm_L1 Logs](https://github.com/NSR9/EVA8/blob/main/Session-5/logs/batch_norm_l1/batch_norm_l1)|
|Layer Normalization| Yes | - |43208 |97.33 |99.06 | [Layer Norm_L1 Logs](https://github.com/NSR9/EVA8/blob/main/Session-5/logs/layer_norm_l1/layer_norm_l1)|
|Group Normalization| Yes | - |7704| 98.26|99.34 |[Group Norm_L1 Logs](https://github.com/NSR9/EVA8/blob/main/Session-5/logs/group_norm_l1/group_norm_l1) |
|Batch Normalization| Yes | Yes |7704 |97.87 | 99.4|[Batch Norm_L1_L2 Logs](https://github.com/NSR9/EVA8/blob/main/Session-5/logs/batch_norm_l1_l2/batch_norm_l1_l2)|
 
## Graphs and Plots (All 6 models mentioned above is compared):-
|Graph 1: Training Loss for all 3 models together|Graph 2: Test/Validation Loss for all 3 models together|
|--|--|
|![image](https://user-images.githubusercontent.com/51078583/121725761-4bb08000-cb07-11eb-98de-296e91f6a74b.png)|![image](https://user-images.githubusercontent.com/51078583/121725803-59fe9c00-cb07-11eb-818f-ca5cb510792d.png)|

|Graph 3: Training Accuracy for all 3 models together|Graph 4: Test/Validation Accuracy for all 3 models together|
|--|--|
|![image](https://user-images.githubusercontent.com/51078583/121725858-6b47a880-cb07-11eb-8e3a-241b8395cbfc.png)|![image](https://user-images.githubusercontent.com/51078583/121725872-726eb680-cb07-11eb-8d88-ac7bf339ff76.png)|

## Misclassified Images:-

|Group Normalization|Layer Normalization|
|--|--|
|![image](https://user-images.githubusercontent.com/51078583/121727153-27ee3980-cb09-11eb-9172-063f4e97c418.png)|![image](https://user-images.githubusercontent.com/51078583/121726971-e9587f00-cb08-11eb-992a-6da138d4404a.png)|

|Layer Normalization + L1|Group Normalization + L1|
|--|--|
|![image](https://user-images.githubusercontent.com/51078583/121727381-76033d00-cb09-11eb-9229-dc66640ba2e1.png)|![image](https://user-images.githubusercontent.com/51078583/121727434-861b1c80-cb09-11eb-9bf8-eb6ffdd70f90.png)|

|Batch Normalization + L1|Batch Normalization + L1 + L2|
|--|--|
|![image](https://user-images.githubusercontent.com/51078583/121727909-2bce8b80-cb0a-11eb-9cc7-1a151565f973.png)|![image](https://user-images.githubusercontent.com/51078583/121727948-3db02e80-cb0a-11eb-9bab-6c2d4b1dba49.png)|



