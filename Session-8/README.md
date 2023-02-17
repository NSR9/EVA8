# Building a Custom network for CIFAR 10 classification
## Problem statement:-

1. Write a custom ResNet architecture for CIFAR10 that has the following architecture:
  1. PrepLayer - Conv 3x3 s1, p1) >> BN >> RELU [64k]
  2. Layer1 -
    1. X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
    2. R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k] 
    3. Add(X, R1)
  3. Layer 2 -
    1. Conv 3x3 [256k]
    2. MaxPooling2D
    3. BN
    4. ReLU
  4. Layer 3 -
    1. X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]
    2. R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]
    3. Add(X, R2)
  5. MaxPooling with Kernel Size 4
  6. FC Layer 
  7. SoftMax
2. Uses One Cycle Policy such that:
  1. Total Epochs = 24
  2. Max at Epoch = 5
  3. LRMIN = FIND
  4. LRMAX = FIND
  5. NO Annihilation
3. Uses this transform -RandomCrop 32, 32 (after padding of 4) >> FlipLR >> Followed by CutOut(8, 8)
4. Batch size = 512
5. Target Accuracy: 90% (93% for late submission or double scores). 
6. NO score if your code is not modular. Your collab must be importing your GitHub package, and then just running the model. I should be able to find the custom_resnet.py model in your GitHub repo that you'd be training.

## Parameters:
- Batch Size - 512
- Transforms -
  - Padding(4,4)
  - Random Crop (32,32)
  - Flip Lr
  - Cutout(8,8)
- Model Total Parameters - 6,573,120
- Range Test
  - Max Lr - 0.02
  - Min Lr - 0.001
  - Epochs - 24
- Loss function - CrossEntropyLoss()
- Optimiser - SGD
  - Weight Decay - 0.05
  - Mometum - 0.9
- Scheduler - One Cycl2 Lr
  - epoch - 24
  - Max Lr - 0.012400000000000001
  - no of steps - 98
  - pct start - 0.0125
  - cyclic momentum -False

## Model Summary:

![image](https://github.com/NSR9/EVA8/blob/main/Session-8/Screenshot%202023-02-17%20at%2010.19.27%20AM.png)

## Training logs:
    EPOCH: 1 LR: 0.002
    Loss=1.222569227218628 Batch_id=97 Accuracy=40.72: 100%|██████████| 98/98 [00:26<00:00,  3.74it/s]

    Test set: Average loss: 0.0026, Accuracy: 5393/10000 (53.93%)

    EPOCH: 2 LR: 0.006502480958895718
    Loss=1.126226544380188 Batch_id=97 Accuracy=56.43: 100%|██████████| 98/98 [00:26<00:00,  3.70it/s]

    Test set: Average loss: 0.0022, Accuracy: 6136/10000 (61.36%)

    EPOCH: 3 LR: 0.011004961917791435
    Loss=0.8941907286643982 Batch_id=97 Accuracy=65.99: 100%|██████████| 98/98 [00:26<00:00,  3.77it/s]

    Test set: Average loss: 0.0017, Accuracy: 7083/10000 (70.83%)

    EPOCH: 4 LR: 0.015507442876687155
    Loss=0.8762430548667908 Batch_id=97 Accuracy=70.79: 100%|██████████| 98/98 [00:26<00:00,  3.74it/s]

    Test set: Average loss: 0.0016, Accuracy: 7243/10000 (72.43%)

    EPOCH: 5 LR: 0.01999801553274371
    Loss=0.687553882598877 Batch_id=97 Accuracy=75.78: 100%|██████████| 98/98 [00:26<00:00,  3.75it/s]

    Test set: Average loss: 0.0015, Accuracy: 7614/10000 (76.14%)

    EPOCH: 6 LR: 0.01909765538868609
    Loss=0.5024750828742981 Batch_id=97 Accuracy=80.38: 100%|██████████| 98/98 [00:25<00:00,  3.78it/s]

    Test set: Average loss: 0.0012, Accuracy: 7924/10000 (79.24%)

    EPOCH: 7 LR: 0.018197295244628466
    Loss=0.4539174437522888 Batch_id=97 Accuracy=82.34: 100%|██████████| 98/98 [00:25<00:00,  3.81it/s]

    Test set: Average loss: 0.0011, Accuracy: 8196/10000 (81.96%)

    EPOCH: 8 LR: 0.01729693510057084
    Loss=0.41762083768844604 Batch_id=97 Accuracy=84.28: 100%|██████████| 98/98 [00:25<00:00,  3.87it/s]

    Test set: Average loss: 0.0013, Accuracy: 7838/10000 (78.38%)

    EPOCH: 9 LR: 0.016396574956513216
    Loss=0.3347751200199127 Batch_id=97 Accuracy=86.33: 100%|██████████| 98/98 [00:25<00:00,  3.88it/s]

    Test set: Average loss: 0.0010, Accuracy: 8288/10000 (82.88%)

    EPOCH: 10 LR: 0.015496214812455595
    Loss=0.3862503468990326 Batch_id=97 Accuracy=87.86: 100%|██████████| 98/98 [00:25<00:00,  3.89it/s]

    Test set: Average loss: 0.0009, Accuracy: 8434/10000 (84.34%)

    EPOCH: 11 LR: 0.01459585466839797
    Loss=0.3398081362247467 Batch_id=97 Accuracy=88.62: 100%|██████████| 98/98 [00:25<00:00,  3.87it/s]

    Test set: Average loss: 0.0008, Accuracy: 8611/10000 (86.11%)

    EPOCH: 12 LR: 0.013695494524340348
    Loss=0.3508572578430176 Batch_id=97 Accuracy=89.88: 100%|██████████| 98/98 [00:25<00:00,  3.82it/s]

    Test set: Average loss: 0.0008, Accuracy: 8607/10000 (86.07%)

    EPOCH: 13 LR: 0.012795134380282725
    Loss=0.3430391550064087 Batch_id=97 Accuracy=90.54: 100%|██████████| 98/98 [00:26<00:00,  3.75it/s]

    Test set: Average loss: 0.0008, Accuracy: 8731/10000 (87.31%)

    EPOCH: 14 LR: 0.011894774236225102
    Loss=0.23337534070014954 Batch_id=97 Accuracy=91.49: 100%|██████████| 98/98 [00:26<00:00,  3.73it/s]

    Test set: Average loss: 0.0010, Accuracy: 8304/10000 (83.04%)

    EPOCH: 15 LR: 0.01099441409216748
    Loss=0.2546903192996979 Batch_id=97 Accuracy=92.36: 100%|██████████| 98/98 [00:26<00:00,  3.76it/s]

    Test set: Average loss: 0.0008, Accuracy: 8685/10000 (86.85%)

    EPOCH: 16 LR: 0.010094053948109857
    Loss=0.20041696727275848 Batch_id=97 Accuracy=93.36: 100%|██████████| 98/98 [00:26<00:00,  3.77it/s]

    Test set: Average loss: 0.0009, Accuracy: 8480/10000 (84.80%)

    EPOCH: 17 LR: 0.009193693804052234
    Loss=0.17288218438625336 Batch_id=97 Accuracy=94.05: 100%|██████████| 98/98 [00:26<00:00,  3.77it/s]

    Test set: Average loss: 0.0007, Accuracy: 8815/10000 (88.15%)

    EPOCH: 18 LR: 0.008293333659994611
    Loss=0.23435097932815552 Batch_id=97 Accuracy=94.55: 100%|██████████| 98/98 [00:25<00:00,  3.82it/s]

    Test set: Average loss: 0.0007, Accuracy: 8855/10000 (88.55%)

    EPOCH: 19 LR: 0.0073929735159369864
    Loss=0.13967843353748322 Batch_id=97 Accuracy=95.44: 100%|██████████| 98/98 [00:25<00:00,  3.84it/s]

    Test set: Average loss: 0.0007, Accuracy: 8881/10000 (88.81%)

    EPOCH: 20 LR: 0.006492613371879362
    Loss=0.12370126694440842 Batch_id=97 Accuracy=95.97: 100%|██████████| 98/98 [00:25<00:00,  3.89it/s]

    Test set: Average loss: 0.0007, Accuracy: 8911/10000 (89.11%)

    EPOCH: 21 LR: 0.005592253227821739
    Loss=0.11253122985363007 Batch_id=97 Accuracy=96.52: 100%|██████████| 98/98 [00:25<00:00,  3.91it/s]

    Test set: Average loss: 0.0007, Accuracy: 8908/10000 (89.08%)

    EPOCH: 22 LR: 0.004691893083764116
    Loss=0.08909299224615097 Batch_id=97 Accuracy=97.09: 100%|██████████| 98/98 [00:25<00:00,  3.88it/s]

    Test set: Average loss: 0.0006, Accuracy: 9028/10000 (90.28%)

    EPOCH: 23 LR: 0.0037915329397064934
    Loss=0.07658335566520691 Batch_id=97 Accuracy=97.56: 100%|██████████| 98/98 [00:25<00:00,  3.85it/s]

    Test set: Average loss: 0.0005, Accuracy: 9115/10000 (91.15%)

    EPOCH: 24 LR: 0.0028911727956488706
    Loss=0.0639895647764206 Batch_id=97 Accuracy=97.97: 100%|██████████| 98/98 [00:25<00:00,  3.82it/s]

    Test set: Average loss: 0.0005, Accuracy: 9121/10000 (91.21%)
    
    
## LR Finder:
  
### One Cycle policy
Similar to Cyclic Learning Rate, but here we have only one Cycle. The correct combination of momemtum, weight decay, Learning rate, batch size does magic. One Cycle Policy will not increase accuracy, but the reasons to use it are

It reduces the time it takes to reach "near" to your accuracy.
It allows us to know if we are going right early on.
It let us know what kind of accuracies we can target with given model.
It reduces the cost of training.
It reduces the time to deploy
Both Cyclic Learning rate and One Cycle Policy was introduced by LESLIE SMITH

## Max LR:

LR Finder curve:

![image](https://github.com/NSR9/EVA8/blob/main/Session-8/Screenshot%202023-02-17%20at%2011.46.10%20AM.png)

The flatest part of the curve represents the max LR for One cycle LR. 

    epoch = 1 Lr = 0.001  Loss=1.4825639724731445 Batch_id=97 Accuracy=37.03: 100%|██████████| 98/98 [00:26<00:00,  3.75it/s]
    epoch = 2 Lr = 0.0029  Loss=1.282240390777588 Batch_id=97 Accuracy=41.97: 100%|██████████| 98/98 [00:24<00:00,  3.94it/s]
    epoch = 3 Lr = 0.0048  Loss=1.3062806129455566 Batch_id=97 Accuracy=42.05: 100%|██████████| 98/98 [00:25<00:00,  3.79it/s]
    epoch = 4 Lr = 0.006699999999999999  Loss=1.2352296113967896 Batch_id=97 Accuracy=38.25: 100%|██████████| 98/98 [00:26<00:00,  3.76it/s]
    epoch = 5 Lr = 0.0086  Loss=1.378053069114685 Batch_id=97 Accuracy=32.62: 100%|██████████| 98/98 [00:25<00:00,  3.84it/s]
    epoch = 6 Lr = 0.0105  Loss=1.6600961685180664 Batch_id=97 Accuracy=23.69: 100%|██████████| 98/98 [00:25<00:00,  3.86it/s]
    epoch = 7 Lr = 0.012400000000000001  Loss=1.909290075302124 Batch_id=97 Accuracy=18.94: 100%|██████████| 98/98 [00:25<00:00,  3.77it/s]
    epoch = 8 Lr = 0.014300000000000002  Loss=1.978029489517212 Batch_id=97 Accuracy=18.77: 100%|██████████| 98/98 [00:25<00:00,  3.78it/s]
    epoch = 9 Lr = 0.016200000000000003  Loss=1.801030158996582 Batch_id=97 Accuracy=19.37: 100%|██████████| 98/98 [00:25<00:00,  3.90it/s]
    epoch = 10 Lr = 0.0181  Loss=1.9502837657928467 Batch_id=97 Accuracy=19.78: 100%|██████████| 98/98 [00:26<00:00,  3.76it/s]

**Max LR for oncecycle policy is at epoch 7 with 0.012400000000000001**
    
Lr for 24 epoch:

![image](https://github.com/NSR9/EVA8/blob/main/Session-8/Screenshot%202023-02-17%20at%2010.21.36%20AM.png)


**Max LR is at the 5th epoch**
  
## Results:

- Best Train Accuracy - 97.97%(24th epoch)
- Best Test Accuracy - 91.21%(24th epoch)
- Acheived >93.8% accuracy at 17th Epoch itself.


### Validation loss curve:
![image](https://github.com/NSR9/EVA8/blob/main/Session-8/Screenshot%202023-02-17%20at%2010.22.05%20AM.png)

### Missclassified Images:

![image](https://github.com/NSR9/EVA8/blob/main/Session-8/misclassifiedImages.png)



