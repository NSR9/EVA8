


# CHALLENGE - Write a neural network that can:
## Take 2 inputs:
* an image from the MNIST dataset (say 5), and
* a random number between 0 and 9, (say 7)
## Gives two outputs:
* the "number" that was represented by the MNIST image (predict 5), and
* the "sum" of this number with the random number and the input image to the network (predict 5 + 7 = 12)
                 
* you can mix fully connected layers and convolution layers
* you can use one-hot encoding to represent the random number input and the "summed" output.
* Random number (7) can be represented as 0 0 0 0 0 0 0 1 0 0
* Sum (13) can be represented as: 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0
* 0b1101 (remember that 4 digits in binary can at max represent 15, so we may need to go for 5 digits. i.e. 10010
##Your code MUST be:
* well documented (via readme file on GitHub and comments in the code)
* must mention the data representation
* must mention your data generation strategy (basically the class/method you are using for random number generation)
* must mention how you have combined the two inputs (basically which layer you are combining)
* must mention how you are evaluating your results 
* must mention "what" results you finally got and how did you evaluate your results
* must mention what loss function you picked and why!
* training MUST happen on the GPU
* Accuracy is not really important for the SUM
Once done, upload the code with short training logs in the readme file from colab to GitHub, and share the GitHub link (public repository)


# My Solution
I have written our neural network to solve the problem for the assignment. I have used the MNIST dataset and modified with adding the random number. 

## **Building a custom dataset class by using MNIST dataset:**
This custom dataset is composed of:

* MNIST Image
* MNIST Image Label
* One hot encoded vector representing a random number.
* Sum of the label and the random number (2 + 3 from above).
## **__init__() function:**
* This class is named as "MNIST_Fusion".
* This function takes 4 parameters in which one is self and others are train_set, test_set, train variable.
* The train variable is used to determine which dataset to load. If train is "True" the training dataset is loaded or else if "False" the test dataset is loaded.
* The core logic of the function is, it takes each image from MNIST train_set and slices to get 10 fixed pixel values into a tensor.
* Then the obtained tensor is converted in to a one hot vector after some transformations.
* Along with this even the sum of image label and the random number has been calculated which is termed as sum label.
* Hence, the Image, image label, one hot vector of random number and sum are generated as part of init() function.
## **__getitem__() function:**
This function returns a particular value of the dataset based on the given index. Thus, this function makes the dataset iterable.

## **__len__() function:**
This function returns the length of the dataset.

![Data Representation](https://github.com/NSR9/EVA8/blob/main/pytorch/dataset%20rep.png)

## **Network Architecture**
Network:

* 7 convolution layers
* 2 fully connected layers
* 2 max pooling layers
* 7th conv layers will give 10 outputs so that concatenation with encoded random tensor will of same dimemsion
* Adjusted padding so that 7th conv layer be give 10 outputs

Concatination of the two inputs:

The output of the 7th conv layer contains the 10 tensor values of the MNIST image input.

![Network Architecture](https://github.com/NSR9/EVA8/blob/main/pytorch/WhatsApp%20Image%202023-01-06%20at%204.48.15%20PM.jpeg)

## **Network Summary**

![Network Summary](https://github.com/NSR9/EVA8/blob/main/pytorch/Screenshot%202023-01-06%20at%202.49.30%20PM.png)

## **Loss Calculation for the best model**
Two losses are calculated for both MNIST_Image and Sum. I used "cross_entropy" as it is the best choice for a calssification problem.


## **Training Logs for the best model**

![Training Logs](https://github.com/NSR9/EVA8/blob/main/pytorch/Screenshot%202023-01-06%20at%204.32.23%20PM.png)


## **Testing Logs for the best model**

![Testing Logs](https://github.com/NSR9/EVA8/blob/main/pytorch/Screenshot%202023-01-06%20at%204.57.31%20PM.png)





