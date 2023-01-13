## **Problem Statement:-**

Your new target is:
* 99.4% (this must be consistently shown in your last few epochs, and not a one-time achievement)
* Less than or equal to 15 Epochs
* Less than 10000 Parameters (additional points for doing this in less than 8000 pts)
* Do this in exactly 3 steps
* Each File must have "target, result, analysis" TEXT block (either at the start or the end)
* Keep Receptive field calculations handy for each of your models. 

## **Step Models:-**
### **Target, Result and Analysis of Step models:-**

|[Notebook_1(Github_Link)](https://github.com/NSR9/EVA8/blob/main/S4/NoteBook_1.ipynb)![image](https://github.com/NSR9/EVA8/blob/main/S4/Notebook1.png)|[Notebook_2(Github_Link)](https://github.com/NSR9/EVA8/blob/main/S4/NoteBook_2.ipynb)![image](https://github.com/NSR9/EVA8/blob/main/S4/Notebook2.png)|
|--|--|

[Notebook_4(Github_Link)](https://github.com/NSR9/Extensive-Vision-AI/blob/main/Assignment_5/Notebook_4(Final)_StepLR_7.5KParam.ipynb)![image](https://github.com/NSR9/EVA8/blob/main/S4/Notebook3.png)|
|--|--|

### **Feature Implementation in Step models:-**

**Here is the Tabular representation of the features implemented in the four step models**

| Model | Params Count | 1x1 conv layer| Maxpooling | Batch Normalization | Dropout | FC Layer| GAP | Image Augumentation | Optimizer | Schedular | 
|--|--|--|--|--|--|--|--|--|--|--|
|Notebook-1| 5,024 | Yes | Yes | No | No | No | No | No | SGD | No | 
|Notebook-2| 8,052 | Yes | Yes | Yes | Yes(0.1)| No | Yes | No | SGD | No | 
|Notebook-3| 7,228  | Yes | Yes | Yes | Yes(0.05) | No | Yes | Yes(RandomRoatation) | SGD | Yes(StepLR) |  


**Note:-**
* ReLU, Batch Normalization and Dropout if Implemented is added to each Conv Layer expect the prediction Layer.
* Image Augumentation was applied on the traning dataset while the test dataset was left untouched.

### **Receptive Field Calulation of Models:-**
**Formulae**

<img src="https://user-images.githubusercontent.com/51078583/120814031-17b7e680-c56c-11eb-8a87-7bd01dd2c849.png" width=400 height=400>

#### Final Notebook
[Notebook_4(Github_Link)](https://github.com/NSR9/Extensive-Vision-AI/blob/main/Assignment_5/Notebook_4(Final)_StepLR_7.5KParam.ipynb)
|Operation|In_Dim|Out_Dim|In_channels|Out_channels|Pad|Stride|Jin|Jout|Rf_in|Rf_out|
|--|--|--|--|--|--|--|--|--|--|--|
|Convolution|28x28|28x28|1|8|1|0|1|1|1x1|3x3|
|Convolution|28x28|28x28|8|8|1|0|1|1|3x3|5x5|
|Max Pool|28x28|14x14|8|8|0|2|1|2|5x5|6x6|
|Convolution|14x14|12x12|8|10|0|1|2|2|6x6|10x10|
|Convolution|12x12|10x10|10|10|0|1|2|2|10x10|14x14|
|Convolution|10x10|8x8|10|10|0|1|2|2|14x14|18x18|
|Convolution|8x8|6x6|10|16|0|1|2|2|18x18|22x22|
|Convolution|6x6|4x4|16|16|0|1|2|2|22x22|26x26|
|GAP|4x4|1x1|16|16|1|6|2|2|26x26|36x36|
|Convolution|1x1|1x1|16|10|0|1|2|2|36x36|36x36|

## **Proposed Network (Best Network - Notebook_4):-**

### **Network Diagram:-**

![Network Diagram](https://user-images.githubusercontent.com/50147394/120842177-a9275880-c56c-11eb-82bd-3bea15401348.png)


### **Network Block:-**
#### Conv Block 1
* 2D Convolution number of kernels 8, followed with Batch Normalization and 2D Dropout of 0.05
* 2D Convolution number of kernels 8, followed with Batch Normalization and 2D Dropout of 0.05
#### Transition Layer 1
* 2D Max Pooling to reduce the size of the channel to 12
#### Conv Block 2
* 2D Convolution number of kernels 10, followed with Batch Normalization and 2D Dropout of 0.05
* 2D Convolution number of kernels 10, followed with Batch Normalization and 2D Dropout of 0.05
* 2D Convolution number of kernels 10, followed with Batch Normalization and 2D Dropout of 0.05
* 2D Convolution number of kernels 16, followed with Batch Normalization and 2D Dropout of 0.05
* 2D Convolution number of kernels 16, followed with Batch Normalization and 2D Dropout of 0.05
#### Global Average Pooling
* Global Average pooling with a size 6 and Padding 1 to return a 16 x 1 x 1 as output dimensions
#### Conv Block 3
* 2D Convolution number of kernels 10, followed with Batch Normalization and 2D Dropout of 0.05

## Model Summary:-
![image](https://user-images.githubusercontent.com/51078583/120816521-72ead880-c56e-11eb-9c11-d0b1682fff2d.png)

### Goals Achived:-
In the Notebook 3 we achieved the goal of 99.4% accuracy. But the model was not stable with that accuracy. Notebook_4 achieved all the required goals. 
* The Target was achieved with **less than 8,000 Parameters**, exactly **7,228**.
* Achieved Accuracy of **99.40% in the 11th Epoch** and Highest achieved accuracy was of **99.41 at the 18th Epoch**. 
* The model was **not overfitting** and the Gap beween the Training and testing accuracy was very less(Can be seen in the Training-Validation curve.)

### Logs of Final Model:-
```
0%|          | 0/469 [00:00<?, ?it/s]<ipython-input-2-bbbdcc178c05>:70: UserWarning: Implicit dimension choice for log_softmax has been deprecated. Change the call to include dim=X as an argument.
  return F.log_softmax(x)
Epoch1 : Loss=0.19866915047168732  Accuracy=83.44 Batch_id=468: 100%|██████████| 469/469 [00:20<00:00, 23.06it/s]

Test set: Average loss: 0.0970, Accuracy: 9781/10000 (97.81%)

Epoch2 : Loss=0.178505077958107  Accuracy=97.13 Batch_id=468: 100%|██████████| 469/469 [00:20<00:00, 22.85it/s]

Test set: Average loss: 0.0520, Accuracy: 9870/10000 (98.70%)

Epoch3 : Loss=0.06099693849682808  Accuracy=97.70 Batch_id=468: 100%|██████████| 469/469 [00:20<00:00, 22.80it/s]

Test set: Average loss: 0.0405, Accuracy: 9891/10000 (98.91%)

Epoch4 : Loss=0.041738320142030716  Accuracy=97.97 Batch_id=468: 100%|██████████| 469/469 [00:20<00:00, 23.07it/s]

Test set: Average loss: 0.0336, Accuracy: 9904/10000 (99.04%)

Epoch5 : Loss=0.12109624594449997  Accuracy=98.09 Batch_id=468: 100%|██████████| 469/469 [00:20<00:00, 22.67it/s]

Test set: Average loss: 0.0332, Accuracy: 9903/10000 (99.03%)

Epoch6 : Loss=0.04045925661921501  Accuracy=98.27 Batch_id=468: 100%|██████████| 469/469 [00:21<00:00, 22.31it/s]

Test set: Average loss: 0.0254, Accuracy: 9925/10000 (99.25%)

Epoch7 : Loss=0.08622000366449356  Accuracy=98.52 Batch_id=468: 100%|██████████| 469/469 [00:21<00:00, 21.56it/s]

Test set: Average loss: 0.0228, Accuracy: 9938/10000 (99.38%)

Epoch8 : Loss=0.02999991364777088  Accuracy=98.61 Batch_id=468: 100%|██████████| 469/469 [00:20<00:00, 22.58it/s]

Test set: Average loss: 0.0223, Accuracy: 9931/10000 (99.31%)

Epoch9 : Loss=0.02057088352739811  Accuracy=98.65 Batch_id=468: 100%|██████████| 469/469 [00:20<00:00, 22.62it/s]

Test set: Average loss: 0.0218, Accuracy: 9935/10000 (99.35%)

Epoch10 : Loss=0.0549580417573452  Accuracy=98.60 Batch_id=468: 100%|██████████| 469/469 [00:20<00:00, 23.04it/s]

Test set: Average loss: 0.0216, Accuracy: 9938/10000 (99.38%)

Epoch11 : Loss=0.04271766170859337  Accuracy=98.61 Batch_id=468: 100%|██████████| 469/469 [00:20<00:00, 22.54it/s]

Test set: Average loss: 0.0216, Accuracy: 9940/10000 (99.40%)

Epoch12 : Loss=0.018340855836868286  Accuracy=98.66 Batch_id=468: 100%|██████████| 469/469 [00:20<00:00, 22.80it/s]

Test set: Average loss: 0.0221, Accuracy: 9931/10000 (99.31%)

Epoch13 : Loss=0.026749903336167336  Accuracy=98.64 Batch_id=468: 100%|██████████| 469/469 [00:20<00:00, 22.68it/s]

Test set: Average loss: 0.0211, Accuracy: 9936/10000 (99.36%)

Epoch14 : Loss=0.017289668321609497  Accuracy=98.73 Batch_id=468: 100%|██████████| 469/469 [00:20<00:00, 22.80it/s]

Test set: Average loss: 0.0208, Accuracy: 9937/10000 (99.37%)

Epoch15 : Loss=0.038884490728378296  Accuracy=98.69 Batch_id=468: 100%|██████████| 469/469 [00:21<00:00, 21.81it/s]

Test set: Average loss: 0.0206, Accuracy: 9938/10000 (99.38%)

Epoch16 : Loss=0.063925601541996  Accuracy=98.70 Batch_id=468: 100%|██████████| 469/469 [00:20<00:00, 23.03it/s]

Test set: Average loss: 0.0209, Accuracy: 9938/10000 (99.38%)

Epoch17 : Loss=0.07100596278905869  Accuracy=98.68 Batch_id=468: 100%|██████████| 469/469 [00:20<00:00, 22.48it/s]

Test set: Average loss: 0.0209, Accuracy: 9939/10000 (99.39%)

Epoch18 : Loss=0.016464291140437126  Accuracy=98.77 Batch_id=468: 100%|██████████| 469/469 [00:20<00:00, 22.58it/s]

Test set: Average loss: 0.0207, Accuracy: 9941/10000 (99.41%)

Epoch19 : Loss=0.044668663293123245  Accuracy=98.73 Batch_id=468: 100%|██████████| 469/469 [00:20<00:00, 22.95it/s]

Test set: Average loss: 0.0208, Accuracy: 9935/10000 (99.35%)
```

### Training-Validation Curve:-
![image](https://user-images.githubusercontent.com/51078583/120818119-f22cdc00-c56f-11eb-9f05-094773989f82.png)

### Incorrect image:-
Some of the incorrect predicted images.

![image](https://user-images.githubusercontent.com/51078583/120818292-18eb1280-c570-11eb-9cf0-1d49092c7b74.png)


