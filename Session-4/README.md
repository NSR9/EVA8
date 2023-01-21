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

[Notebook_1(Github_Link)](https://github.com/NSR9/EVA8/blob/main/Session-4/S4_step_1_code_setup.ipynb)

![image](https://github.com/NSR9/EVA8/blob/main/Session-4/s4-nbk-1.png)

[Notebook_2(Github_Link)](https://github.com/NSR9/EVA8/blob/main/Session-4/S4_step_2_getting_skeleton_right.ipynb)

![image](https://github.com/NSR9/EVA8/blob/main/Session-4/s4-nbk-2.png)

[Notebook_3(Github_Link)](https://github.com/NSR9/EVA8/blob/main/Session-4/S4_step_3_adding_fancy_stuff.ipynb)

![image](https://github.com/NSR9/EVA8/blob/main/Session-4/s4-nbk-3.png)

[Notebook_4(Github_Link)](https://github.com/NSR9/EVA8/blob/main/Session-4/S4_step_3_adding_fancy_stuff_lessthan_8k.ipynb)

![image](https://github.com/NSR9/EVA8/blob/main/Session-4/s4-nbk-4.png)


### **Feature Implementation in Step models:-**

**Here is the Tabular representation of the features implemented in the four step models**

| Model | Params Count | 1x1 conv layer| Maxpooling | Batch Normalization | Dropout | FC Layer| GAP | Image Augumentation | Optimizer | Schedular | 
|--|--|--|--|--|--|--|--|--|--|--|
|Notebook-1| 6.3 M | No  | Yes | No | No | No | No | No | SGD | No | 
|Notebook-2| 9,946 | Yes | Yes | Yes | No | No | No | No | No | No | 
|Notebook-3| 8,878 | Yes | Yes | Yes | Yes(0.05)| No | Yes | Yes(RandomRotation) | SGD | Yes(StepLR) | 
|Notebook-4| 7,760 | Yes | Yes | Yes | Yes(0.03) | No | Yes | Yes(RandomRoatation) | SGD | Yes(StepLR) | 


**Note:-**
* ReLU, Batch Normalization and Dropout if Implemented is added to each Conv Layer expect the prediction Layer.
* Image Augumentation was applied on the traning dataset while the test dataset was left untouched.

### **Receptive Field Calulation of Models:-**
**Formulae**

<img src="https://user-images.githubusercontent.com/51078583/120814031-17b7e680-c56c-11eb-8a87-7bd01dd2c849.png" width=400 height=400>



## **Proposed Network (Best Network - [Notebook_3(Github_Link)](https://github.com/NSR9/EVA8/blob/main/Session-4/S4_step_3_adding_fancy_stuff.ipynb)):-**

### **Network Diagram:-**
![image](https://github.com/NSR9/EVA8/blob/main/Session-4/s4-ntwrk.jpeg)
![image](https://github.com/NSR9/EVA8/blob/main/Session-4/s4-ntwrk-2.jpeg)


## Model Summary:-
![Model Summary](https://github.com/NSR9/EVA8/blob/main/Session-4/s4-modelsummary.png)


### Goals Achived:-
In the Notebook 3 we achieved the goal of 99.4% accuracy. But the model was not stable with that accuracy. Notebook_4 achieved all the required goals. 
* The Target was achieved with **less than 8,000 Parameters**, exactly **7,760** at 8th epoch itself.
* Achieved Accuracy of **99.40% in the 12th Epoch** and the model was consistant until 15 Epochs with less than 10k network parameters. 
* Highest achieved accuracy was of **99.43 at the 14th Epoch**. 
* Both the models were **not overfitting** and the Gap beween the Training and testing accuracies was very less.

### Logs of Final Model with less than 10k parameters:-
```
  0%|          | 0/469 [00:00<?, ?it/s]<ipython-input-129-3bd8ac46fe15>:79: UserWarning: Implicit dimension choice for log_softmax has been deprecated. Change the call to include dim=X as an argument.
  return F.log_softmax(x)
Epoch=0 Loss=0.15047958493232727 Batch_id=468 Accuracy=88.88: 100%|██████████| 469/469 [00:06<00:00, 70.12it/s]

Test set: Average loss: 0.0843, Accuracy: 9769/10000 (97.69%)

Epoch=1 Loss=0.04363046959042549 Batch_id=468 Accuracy=97.52: 100%|██████████| 469/469 [00:06<00:00, 70.21it/s]

Test set: Average loss: 0.0436, Accuracy: 9859/10000 (98.59%)

Epoch=2 Loss=0.020706115290522575 Batch_id=468 Accuracy=98.07: 100%|██████████| 469/469 [00:06<00:00, 69.99it/s]

Test set: Average loss: 0.0427, Accuracy: 9869/10000 (98.69%)

Epoch=3 Loss=0.02163967303931713 Batch_id=468 Accuracy=98.26: 100%|██████████| 469/469 [00:06<00:00, 69.20it/s]

Test set: Average loss: 0.0289, Accuracy: 9913/10000 (99.13%)

Epoch=4 Loss=0.007190991193056107 Batch_id=468 Accuracy=98.42: 100%|██████████| 469/469 [00:06<00:00, 71.36it/s]

Test set: Average loss: 0.0263, Accuracy: 9922/10000 (99.22%)

Epoch=5 Loss=0.06494271010160446 Batch_id=468 Accuracy=98.72: 100%|██████████| 469/469 [00:06<00:00, 69.51it/s]

Test set: Average loss: 0.0221, Accuracy: 9926/10000 (99.26%)

Epoch=6 Loss=0.010541562922298908 Batch_id=468 Accuracy=98.89: 100%|██████████| 469/469 [00:06<00:00, 69.54it/s]

Test set: Average loss: 0.0212, Accuracy: 9936/10000 (99.36%)

Epoch=7 Loss=0.09586929529905319 Batch_id=468 Accuracy=98.86: 100%|██████████| 469/469 [00:06<00:00, 71.72it/s]

Test set: Average loss: 0.0203, Accuracy: 9939/10000 (99.39%)

Epoch=8 Loss=0.02109113335609436 Batch_id=468 Accuracy=98.91: 100%|██████████| 469/469 [00:06<00:00, 71.54it/s]

Test set: Average loss: 0.0205, Accuracy: 9937/10000 (99.37%)

Epoch=9 Loss=0.05349855124950409 Batch_id=468 Accuracy=98.89: 100%|██████████| 469/469 [00:06<00:00, 70.63it/s]

Test set: Average loss: 0.0215, Accuracy: 9938/10000 (99.38%)

Epoch=10 Loss=0.028526214882731438 Batch_id=468 Accuracy=98.94: 100%|██████████| 469/469 [00:06<00:00, 68.89it/s]

Test set: Average loss: 0.0210, Accuracy: 9935/10000 (99.35%)

Epoch=11 Loss=0.02253641188144684 Batch_id=468 Accuracy=98.92: 100%|██████████| 469/469 [00:06<00:00, 70.47it/s]

Test set: Average loss: 0.0202, Accuracy: 9941/10000 (99.41%)

Epoch=12 Loss=0.04347610101103783 Batch_id=468 Accuracy=98.93: 100%|██████████| 469/469 [00:06<00:00, 70.01it/s]

Test set: Average loss: 0.0202, Accuracy: 9942/10000 (99.42%)

Epoch=13 Loss=0.07090843468904495 Batch_id=468 Accuracy=99.00: 100%|██████████| 469/469 [00:06<00:00, 70.18it/s]

Test set: Average loss: 0.0197, Accuracy: 9943/10000 (99.43%)

Epoch=14 Loss=0.05884191393852234 Batch_id=468 Accuracy=98.92: 100%|██████████| 469/469 [00:06<00:00, 69.14it/s]

Test set: Average loss: 0.0205, Accuracy: 9940/10000 (99.40%)
```
### Logs of Model with less than 8k parameters:-
```   
   0%|          | 0/469 [00:00<?, ?it/s]<ipython-input-44-960def53352f>:79: UserWarning: Implicit dimension choice for log_softmax has been deprecated. Change the call to include dim=X as an argument.
    return F.log_softmax(x)
  Epoch=0 Loss=0.10496430844068527 Batch_id=468 Accuracy=87.57: 100%|██████████| 469/469 [00:06<00:00, 71.38it/s]

  Test set: Average loss: 0.0743, Accuracy: 9793/10000 (97.93%)

  Epoch=1 Loss=0.05292993783950806 Batch_id=468 Accuracy=97.24: 100%|██████████| 469/469 [00:06<00:00, 75.21it/s]

  Test set: Average loss: 0.0480, Accuracy: 9854/10000 (98.54%)

  Epoch=2 Loss=0.01899578981101513 Batch_id=468 Accuracy=97.82: 100%|██████████| 469/469 [00:06<00:00, 74.45it/s]

  Test set: Average loss: 0.0421, Accuracy: 9864/10000 (98.64%)

  Epoch=3 Loss=0.11797485500574112 Batch_id=468 Accuracy=98.11: 100%|██████████| 469/469 [00:06<00:00, 76.26it/s]

  Test set: Average loss: 0.0371, Accuracy: 9880/10000 (98.80%)

  Epoch=4 Loss=0.02002876065671444 Batch_id=468 Accuracy=98.25: 100%|██████████| 469/469 [00:06<00:00, 76.00it/s]

  Test set: Average loss: 0.0296, Accuracy: 9907/10000 (99.07%)

  Epoch=5 Loss=0.049595773220062256 Batch_id=468 Accuracy=98.59: 100%|██████████| 469/469 [00:06<00:00, 74.64it/s]

  Test set: Average loss: 0.0229, Accuracy: 9935/10000 (99.35%)

  Epoch=6 Loss=0.08297168463468552 Batch_id=468 Accuracy=98.68: 100%|██████████| 469/469 [00:06<00:00, 74.21it/s]

  Test set: Average loss: 0.0219, Accuracy: 9938/10000 (99.38%)

  Epoch=7 Loss=0.06245630979537964 Batch_id=468 Accuracy=98.75: 100%|██████████| 469/469 [00:06<00:00, 74.64it/s]

  Test set: Average loss: 0.0215, Accuracy: 9942/10000 (99.42%)

  Epoch=8 Loss=0.016733292490243912 Batch_id=468 Accuracy=98.78: 100%|██████████| 469/469 [00:06<00:00, 74.80it/s]

  Test set: Average loss: 0.0220, Accuracy: 9938/10000 (99.38%)

  Epoch=9 Loss=0.04134121909737587 Batch_id=468 Accuracy=98.79: 100%|██████████| 469/469 [00:06<00:00, 74.71it/s]

  Test set: Average loss: 0.0214, Accuracy: 9934/10000 (99.34%)

  Epoch=10 Loss=0.09046515077352524 Batch_id=468 Accuracy=98.89: 100%|██████████| 469/469 [00:06<00:00, 71.87it/s]

  Test set: Average loss: 0.0215, Accuracy: 9932/10000 (99.32%)

  Epoch=11 Loss=0.010343118570744991 Batch_id=468 Accuracy=98.85: 100%|██████████| 469/469 [00:06<00:00, 75.21it/s]

  Test set: Average loss: 0.0210, Accuracy: 9938/10000 (99.38%)

  Epoch=12 Loss=0.16206704080104828 Batch_id=468 Accuracy=98.84: 100%|██████████| 469/469 [00:06<00:00, 73.26it/s]

  Test set: Average loss: 0.0211, Accuracy: 9937/10000 (99.37%)

  Epoch=13 Loss=0.009042645804584026 Batch_id=468 Accuracy=98.83: 100%|██████████| 469/469 [00:06<00:00, 74.41it/s]

  Test set: Average loss: 0.0206, Accuracy: 9938/10000 (99.38%)
```


### Training-Validation Curves of model with less than 10k parameters:-
![image](https://github.com/NSR9/EVA8/blob/main/Session-4/s4-t%26tgrpahs.png)


