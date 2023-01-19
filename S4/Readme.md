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

## Model Summary:-**

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
* Achieved an highest accuracy of **99.43%** at the **17th Epoch**. 
* Achieved a final Accuracy of **99.40%** in the 19th Epoch.
* Accuracy of "99.40" was maintained in the last 5 epochs. Probably because I trained twice. 



### Logs for Best Model:-
       0%|          | 0/469 [00:00<?, ?it/s]<ipython-input-10-213c9ea7ab98>:64: UserWarning: Implicit dimension choice for log_softmax has been deprecated. Change the call to include dim=X as an argument.
        return F.log_softmax(x)
      epoch=1 loss=0.1400429457 batch_id=00468: 100%|██████████| 469/469 [00:20<00:00, 22.39it/s]

      Test set: Average loss: 0.0613, Accuracy: 9811/10000 (98.11%)

      epoch=2 loss=0.1945169717 batch_id=00468: 100%|██████████| 469/469 [00:20<00:00, 22.43it/s]

      Test set: Average loss: 0.0408, Accuracy: 9877/10000 (98.77%)

      epoch=3 loss=0.0895658657 batch_id=00468: 100%|██████████| 469/469 [00:22<00:00, 21.16it/s]

      Test set: Average loss: 0.0416, Accuracy: 9872/10000 (98.72%)

      epoch=4 loss=0.0531668067 batch_id=00468: 100%|██████████| 469/469 [00:21<00:00, 22.23it/s]

      Test set: Average loss: 0.0311, Accuracy: 9888/10000 (98.88%)

      epoch=5 loss=0.1106830910 batch_id=00468: 100%|██████████| 469/469 [00:20<00:00, 22.70it/s]

      Test set: Average loss: 0.0254, Accuracy: 9910/10000 (99.10%)

      epoch=6 loss=0.0468724035 batch_id=00468: 100%|██████████| 469/469 [00:21<00:00, 21.78it/s]

      Test set: Average loss: 0.0253, Accuracy: 9910/10000 (99.10%)

      epoch=7 loss=0.0998379365 batch_id=00468: 100%|██████████| 469/469 [00:20<00:00, 23.21it/s]

      Test set: Average loss: 0.0233, Accuracy: 9926/10000 (99.26%)

      epoch=8 loss=0.0132545279 batch_id=00468: 100%|██████████| 469/469 [00:20<00:00, 22.80it/s]

      Test set: Average loss: 0.0249, Accuracy: 9914/10000 (99.14%)

      epoch=9 loss=0.0705702230 batch_id=00468: 100%|██████████| 469/469 [00:20<00:00, 22.60it/s]

      Test set: Average loss: 0.0227, Accuracy: 9929/10000 (99.29%)

      epoch=10 loss=0.0187641028 batch_id=00468: 100%|██████████| 469/469 [00:20<00:00, 22.73it/s]

      Test set: Average loss: 0.0223, Accuracy: 9924/10000 (99.24%)

      epoch=11 loss=0.0281922817 batch_id=00468: 100%|██████████| 469/469 [00:20<00:00, 22.77it/s]

      Test set: Average loss: 0.0201, Accuracy: 9933/10000 (99.33%)

      epoch=12 loss=0.0133332424 batch_id=00468: 100%|██████████| 469/469 [00:21<00:00, 21.94it/s]

      Test set: Average loss: 0.0200, Accuracy: 9933/10000 (99.33%)

      epoch=13 loss=0.0511022955 batch_id=00468: 100%|██████████| 469/469 [00:20<00:00, 22.41it/s]

      Test set: Average loss: 0.0229, Accuracy: 9928/10000 (99.28%)

      epoch=14 loss=0.0261212494 batch_id=00468: 100%|██████████| 469/469 [00:20<00:00, 22.85it/s]

      Test set: Average loss: 0.0217, Accuracy: 9934/10000 (99.34%)

      epoch=15 loss=0.0213738903 batch_id=00468: 100%|██████████| 469/469 [00:21<00:00, 21.76it/s]

      Test set: Average loss: 0.0187, Accuracy: 9942/10000 (99.42%)

      epoch=16 loss=0.0517582856 batch_id=00468: 100%|██████████| 469/469 [00:20<00:00, 23.29it/s]

      Test set: Average loss: 0.0193, Accuracy: 9941/10000 (99.41%)

      epoch=17 loss=0.0528213531 batch_id=00468: 100%|██████████| 469/469 [00:20<00:00, 23.19it/s]

      Test set: Average loss: 0.0186, Accuracy: 9943/10000 (99.43%)

      epoch=18 loss=0.0319731496 batch_id=00468: 100%|██████████| 469/469 [00:20<00:00, 23.43it/s]

      Test set: Average loss: 0.0202, Accuracy: 9942/10000 (99.42%)

      epoch=19 loss=0.0092981961 batch_id=00468: 100%|██████████| 469/469 [00:20<00:00, 23.19it/s]

      Test set: Average loss: 0.0189, Accuracy: 9940/10000 (99.40%)


### **Validation Loss Curve:-**

![image](https://github.com/NSR9/EVA8/blob/main/S4/training_s3_curve.png)







