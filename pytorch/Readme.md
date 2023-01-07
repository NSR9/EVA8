

# Fundamentals of Nueral Networks and Convolutions

I have written our neural network to solve the problem for the assignment. We have used the MNIST dataset and modified with adding the random no.

## **Data Representation**

Building a custom dataset class by using MNIST dataset:
This custom dataset is composed of:

MNIST Image
MNIST Image Label
One hot encoded vector representing a random number.
Sum of the label and the random number (2 + 3 from above).
init() function:
This class is named as "MNIST_Fusion".
This function takes 4 parameters in which one is self and others are train_set, test_set, train variable.
The train variable is used to determine which dataset to load. If train is "True" the training dataset is loaded or else if "False" the test dataset is loaded.
The core logic of the function is, it takes each image from MNIST train_set and slices to get 10 fixed pixel values into a tensor.
Then the obtained tensor is converted in to a one hot vector after some transformations.
Along with this even the sum of image label and the random number has been calculated which is termed as sum label.
Hence, the Image, image label, one hot vector of random number and sum are generated as part of init() function.
getitem() function:
This function returns a particular value of the dataset based on the given index. Thus, this function makes the dataset iterable.

len():
This function returns the length of the dataset.

![Data Representation]()

## **Network Architecture**
Network:

7 convolution layers
2 fully connected layers
2 max pooling layers
7th conv layers will give 10 outputs so that concatenation with encoded random tensor will of same dimemsion
Adjusted padding so that 7th conv layer be give 10 outputs
Concatination of the two inputs:

The output of the 7th conv layer contains the 10 tensor values of the MNIST image input.

![Network Architecture](https://user-images.githubusercontent.com/50147394/119181866-7bbdb380-ba72-11eb-9f8d-8f0e5718380a.jpg)

## **Network Summary**

![Network Summary](https://user-images.githubusercontent.com/50147394/119182925-ae1be080-ba73-11eb-9117-076d2cd8157c.jpg)

## **Loss Calculation for the best model**
Two losses are calculated for both MNIST_Image and Sum. I used "cross_entropy" as it is the best choice for a calssification problem.


## **Training Logs for the best model**

![Training Logs](https://user-images.githubusercontent.com/50147394/119184501-bc6afc00-ba75-11eb-9716-91e350e4d5a4.JPG)


## **Testing Logs for the best model**

![Testing Logs](https://user-images.githubusercontent.com/50147394/119184617-e45a5f80-ba75-11eb-844c-6368ac093215.JPG)





