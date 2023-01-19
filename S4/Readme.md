## **Problem Statement**

### **WRITE DOWN THE CODE FOR MNIST CLASSIFICATION WITH FOLLOWING CONSTRAINTS:-**
* 99.4% validation accuracy
* Less than 20k Parameters
* Less than 20 Epochs
* Can use BN, Dropout, a Fully connected layer, have used GAP.

## **Proposed Network (Best Network):-**

### **Network Block :**

![image](https://user-images.githubusercontent.com/51078583/120019024-8f829000-c005-11eb-8e6d-2756b71a4f72.png)

#### Conv Block 1
* 2D Convolution number of kernels 8, followed with Batch Normalization and 2D Dropout of 0.1 
* 2D Convolution number of kernels 16, followed with Batch Normalization and 2D Dropout of 0.1
* 2D Convolution number of kernels 32, followed with Batch Normalization and 2D Dropout of 0.1
#### Transition Layer 1
* 2D Max Pooling to reduce the size of the channel to 14
* 2d Convolution with kernel size 1 reducing the number of channels to 8
#### Conv Block 2
* 2D Convolution number of kernels 16, followed with Batch Normalization and 2D Dropout of 0.1 
* 2D Convolution number of kernels 16, followed with Batch Normalization and 2D Dropout of 0.1
* 2D Convolution number of kernels 16, followed with Batch Normalization and 2D Dropout of 0.1
#### Transition Layer 2
* 2D Max Pooling to reduce the size of the channel to 7
#### Conv Block 3
* 2D Convolution number of kernels 16, followed with Batch Normalization and 2D Dropout of 0.1 
* 2D Convolution number of kernels 10 (Avoid Batch Normalization and Dropout in Last layer before GAP)
#### Global Average Pooling
* Global Average pooling with a size 3 and no Padding to return a 10 x 1 x 1 as the value to go to log_softmax 

## **Best Model Summary:-**
#### [Github_link](https://github.com/NSR9/Extensive-Vision-AI/blob/main/Assignment_4/Part%202/Final_Submission_Assignment_4_NB_3.ipynb)
### Enhancements to the Model:
       * Random Rotation of 5 applied as Data Augumentation Methodlogy
       * Activation Function as ReLU is used after conv layers
       * MaxPool Layer of 2 x 2 is used twice in the network. 
       * Conv 1X1 is used in the transition layer for reducing the number of channels
       * Added batch normalization after every conv layer
       * Added dropout of 0.1 after each conv layer
       * Added Global average pooling to get output classes.
       * Use learning rate of 0.01 and momentum 0.9
       
### Goals Achieved:-
* Parameters count reduced to as low as **15,970**
* Achieved an highest accuracy of **99.50%** at the **18th Epoch**. Getting an accuracy of greater than **99.40%** for the first time at **12th epoch** itself. 
* Achieved a final Accuracy of **99.47%** after 19 Epochs.



### Logs for Best Model:-

 0%|          | 0/469 [00:00<?, ?it/s]<ipython-input-2-213c9ea7ab98>:64: UserWarning: Implicit dimension choice for log_softmax has been deprecated. Change the call to include dim=X as an argument.
  return F.log_softmax(x)
epoch=1 loss=0.0821683183 batch_id=00468: 100%|██████████| 469/469 [00:22<00:00, 20.92it/s]

Test set: Average loss: 0.0567, Accuracy: 9825/10000 (98.25%)

epoch=2 loss=0.1212320030 batch_id=00468: 100%|██████████| 469/469 [00:20<00:00, 23.00it/s]

Test set: Average loss: 0.0380, Accuracy: 9879/10000 (98.79%)

epoch=3 loss=0.0965330601 batch_id=00468: 100%|██████████| 469/469 [00:20<00:00, 22.95it/s]

Test set: Average loss: 0.0381, Accuracy: 9879/10000 (98.79%)

epoch=4 loss=0.0571977235 batch_id=00468: 100%|██████████| 469/469 [00:21<00:00, 22.14it/s]

Test set: Average loss: 0.0301, Accuracy: 9905/10000 (99.05%)

epoch=5 loss=0.1049664840 batch_id=00468: 100%|██████████| 469/469 [00:20<00:00, 22.98it/s]

Test set: Average loss: 0.0276, Accuracy: 9926/10000 (99.26%)

epoch=6 loss=0.0404731818 batch_id=00468: 100%|██████████| 469/469 [00:20<00:00, 22.80it/s]

Test set: Average loss: 0.0250, Accuracy: 9917/10000 (99.17%)

epoch=7 loss=0.0662922040 batch_id=00468: 100%|██████████| 469/469 [00:21<00:00, 21.85it/s]

Test set: Average loss: 0.0231, Accuracy: 9918/10000 (99.18%)

epoch=8 loss=0.0173322111 batch_id=00468: 100%|██████████| 469/469 [00:20<00:00, 22.96it/s]

Test set: Average loss: 0.0225, Accuracy: 9930/10000 (99.30%)

epoch=9 loss=0.0503057055 batch_id=00468: 100%|██████████| 469/469 [00:20<00:00, 22.92it/s]

Test set: Average loss: 0.0205, Accuracy: 9930/10000 (99.30%)

epoch=10 loss=0.0347639285 batch_id=00468: 100%|██████████| 469/469 [00:21<00:00, 21.94it/s]

Test set: Average loss: 0.0211, Accuracy: 9929/10000 (99.29%)

epoch=11 loss=0.0299503785 batch_id=00468: 100%|██████████| 469/469 [00:20<00:00, 22.66it/s]

Test set: Average loss: 0.0195, Accuracy: 9936/10000 (99.36%)

epoch=12 loss=0.0112679536 batch_id=00468: 100%|██████████| 469/469 [00:20<00:00, 23.11it/s]

Test set: Average loss: 0.0196, Accuracy: 9941/10000 (99.41%)

epoch=13 loss=0.0362340398 batch_id=00468: 100%|██████████| 469/469 [00:20<00:00, 22.96it/s]

Test set: Average loss: 0.0193, Accuracy: 9935/10000 (99.35%)

epoch=14 loss=0.0162401777 batch_id=00468: 100%|██████████| 469/469 [00:20<00:00, 22.83it/s]

Test set: Average loss: 0.0193, Accuracy: 9931/10000 (99.31%)

epoch=15 loss=0.0135711310 batch_id=00468: 100%|██████████| 469/469 [00:20<00:00, 22.71it/s]

Test set: Average loss: 0.0197, Accuracy: 9936/10000 (99.36%)

epoch=16 loss=0.0165842734 batch_id=00468: 100%|██████████| 469/469 [00:20<00:00, 22.97it/s]

Test set: Average loss: 0.0211, Accuracy: 9932/10000 (99.32%)

epoch=17 loss=0.0304080173 batch_id=00468: 100%|██████████| 469/469 [00:21<00:00, 22.07it/s]

Test set: Average loss: 0.0190, Accuracy: 9939/10000 (99.39%)

epoch=18 loss=0.0544954538 batch_id=00468: 100%|██████████| 469/469 [00:20<00:00, 22.94it/s]

Test set: Average loss: 0.0206, Accuracy: 9927/10000 (99.27%)

epoch=19 loss=0.0472117402 batch_id=00468: 100%|██████████| 469/469 [00:23<00:00, 20.23it/s]

Test set: Average loss: 0.0178, Accuracy: 9937/10000 (99.37%)       
      
### **Validation Loss Curve:-**

![image](https://user-images.githubusercontent.com/51078583/120013747-c5704600-bffe-11eb-840e-ad2ae3d49969.png)


## **Expirement Models:-**

#### **Experiment Model 1 Summary:-**
#### [Github_link](https://github.com/NSR9/Extensive-Vision-AI/blob/main/Assignment_4/Part%202/Experiment_Nb_1.ipynb)
**Enhancements**

       * Activation Function as ReLU is used after conv layers
       * MaxPool Layer of 2 x 2 is used twice in the network.
       * Added batch normalization after every conv layer
       * Added dropout of 0.069 after each conv layer
       * Added Global average pooling before the FC layer and then added the FC to get output classes.
       * Use learning rate of 0.02 and momentum 0.8

* **Paramerters Used** - **14,906** 
* **Best Accuracy** - **99.49% at the 16th Epoch**

![image](https://user-images.githubusercontent.com/51078583/120001574-8daed180-bff1-11eb-90ae-291d5cfc5ed0.png)

**Drawbacks**
* Fully connected layers are used. 
* 1 x 1 Conv layers not used to reduce the number of channels 




