## Assignment
1. Check this Repo out: https://github.com/kuangliu/pytorch-cifar Links to an external site. 
2. You are going to follow the same structure for your Code from now on
 
Problem:

1. Run this network Links to an external site..
2. Fix the network above:
3. change the code such that it uses GPU and change the architecture to C1C2C3C40 (No MaxPooling, but 3 3x3 layers with stride of 2 instead) (If you can figure out how to use Dilated kernels here instead of MP or strided convolution, then 200pts extra!)
total RF must be more than 44
4. one of the layers must use Depthwise Separable Convolution
5. one of the layers must use Dilated Convolution
6. use GAP (compulsory):- add FC after GAP to target #of classes (optional)
7. use albumentation library and apply:
   horizontal flip
   shiftScaleRotate
   coarseDropout (max_holes = 1, max_height=16px, max_width=1, min_holes = 1, min_height=16px, min_width=16px, fill_value=(mean of your dataset),       mask_fill_value = None)
8. achieve 85% accuracy, as many epochs as you want. Total Params to be less than 200k.
9. upload to Github


## Model 
Below is the model code that is modified as per assignment.


     class Net(nn.Module):
       def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),  # 32 32
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),  # 32 32
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.1),
        )
        self.trans1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 1), padding=0, bias=False)  # 32 32
        )
        self.conv2 = nn.Sequential(
            # nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),# 32 30
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), groups=32, padding=0, bias=False),
            # depth_conv
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 1), padding=0, bias=False),  # point_conv
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.1),

            # nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),# 30 28
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), groups=32, padding=0, bias=False),
            # depth_conv
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 1), padding=0, bias=False),  # point_conv
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.1),

            # nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),# 28 26
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), groups=32, padding=0, bias=False),
            # depth_conv
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 1), padding=0, bias=False),  # point_conv
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.1),
        )
        # dilation
        self.trans2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3, 3), stride=1, padding=0, dilation=2),  # 26 22
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            # nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),# 22 20
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), groups=16, padding=0, bias=False),
            # depth_conv
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 1), padding=0, bias=False),  # point_conv
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.1),

            # nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=0, bias=False),# 20 18
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), groups=32, padding=0, bias=False),
            # depth_conv
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), padding=0, bias=False),  # point_conv
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.1),

            # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=0, bias=False),# 18 16
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), groups=64, padding=0, bias=False),
            # depth_conv
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 1), padding=0, bias=False),  # point_conv
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.1),
        )

        self.trans3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=2, padding=0),  # 16 7
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=(1, 1), padding=0, bias=False)  # 7 7
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),  # 7 5
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=0, bias=False),  # 5 3
        )

        self.avgpool2d = nn.AvgPool2d(kernel_size=3)

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=10, kernel_size=(1, 1), padding=0, bias=False), )

    def forward(self, x):
        x = self.conv1(x)
        x = self.trans1(x)
        x = self.conv2(x)
        x = self.trans2(x)
        x = self.conv3(x)
        x = self.trans3(x)
        x = self.conv4(x)
        x = self.avgpool2d(x)
        x = self.conv5(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)


## Model Summary
Below is the model summary of the above custom model.

    ----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
    ================================================================
            Conv2d-1           [-1, 32, 32, 32]             864
              ReLU-2           [-1, 32, 32, 32]               0
       BatchNorm2d-3           [-1, 32, 32, 32]              64
           Dropout-4           [-1, 32, 32, 32]               0
            Conv2d-5           [-1, 64, 32, 32]          18,432
              ReLU-6           [-1, 64, 32, 32]               0
       BatchNorm2d-7           [-1, 64, 32, 32]             128
           Dropout-8           [-1, 64, 32, 32]               0
            Conv2d-9           [-1, 32, 32, 32]           2,048
           Conv2d-10           [-1, 32, 30, 30]             288
           Conv2d-11           [-1, 32, 30, 30]           1,024
             ReLU-12           [-1, 32, 30, 30]               0
      BatchNorm2d-13           [-1, 32, 30, 30]              64
          Dropout-14           [-1, 32, 30, 30]               0
           Conv2d-15           [-1, 32, 28, 28]             288
           Conv2d-16           [-1, 32, 28, 28]           1,024
             ReLU-17           [-1, 32, 28, 28]               0
      BatchNorm2d-18           [-1, 32, 28, 28]              64
          Dropout-19           [-1, 32, 28, 28]               0
           Conv2d-20           [-1, 32, 26, 26]             288
           Conv2d-21           [-1, 32, 26, 26]           1,024
             ReLU-22           [-1, 32, 26, 26]               0
      BatchNorm2d-23           [-1, 32, 26, 26]              64
          Dropout-24           [-1, 32, 26, 26]               0
           Conv2d-25           [-1, 16, 22, 22]           4,624
             ReLU-26           [-1, 16, 22, 22]               0
           Conv2d-27           [-1, 16, 20, 20]             144
           Conv2d-28           [-1, 32, 20, 20]             512
             ReLU-29           [-1, 32, 20, 20]               0
      BatchNorm2d-30           [-1, 32, 20, 20]              64
          Dropout-31           [-1, 32, 20, 20]               0
           Conv2d-32           [-1, 32, 18, 18]             288
           Conv2d-33           [-1, 64, 18, 18]           2,048
             ReLU-34           [-1, 64, 18, 18]               0
      BatchNorm2d-35           [-1, 64, 18, 18]             128
          Dropout-36           [-1, 64, 18, 18]               0
           Conv2d-37           [-1, 64, 16, 16]             576
           Conv2d-38           [-1, 64, 16, 16]           4,096
             ReLU-39           [-1, 64, 16, 16]               0
      BatchNorm2d-40           [-1, 64, 16, 16]             128
          Dropout-41           [-1, 64, 16, 16]               0
           Conv2d-42             [-1, 64, 7, 7]          36,928
             ReLU-43             [-1, 64, 7, 7]               0
           Conv2d-44             [-1, 16, 7, 7]           1,024
           Conv2d-45             [-1, 32, 5, 5]           4,608
             ReLU-46             [-1, 32, 5, 5]               0
      BatchNorm2d-47             [-1, 32, 5, 5]              64
          Dropout-48             [-1, 32, 5, 5]               0
           Conv2d-49             [-1, 64, 3, 3]          18,432
        AvgPool2d-50             [-1, 64, 1, 1]               0
           Conv2d-51             [-1, 10, 1, 1]             640
    ================================================================
    Total params: 99,968
    Trainable params: 99,968
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.01
    Forward/backward pass size (MB): 8.11
    Params size (MB): 0.38
    Estimated Total Size (MB): 8.50
    ----------------------------------------------------------------

## Albumentations Code

      
     class Transforms:
       def __init__(self, transforms: A.Compose):
        self.transforms = transforms

       def __call__(self, img, *args, **kwargs):
        return self.transforms(image=np.array(img))
       
       train_transforms = A.Compose([
        A.HorizontalFlip(p=0.3),
        A.Cutout(num_holes=1, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, interpolation=cv2.INTER_LINEAR,
                           border_mode=cv2.BORDER_REFLECT_101, always_apply=False, p=0.5),
        A.CoarseDropout(max_holes = 1, max_height=16, max_width=16, min_holes = 1, min_height=16,
                        min_width=16, fill_value=(0.4914, 0.4822, 0.4465), mask_fill_value = None),
        A.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)),
        ToTensorV2(),
        ])
        test_transform = A.Compose([
        A.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)),
        ToTensorV2(),
        ])

        train_transform = Transforms(train_transform)
        test_transform = Transforms(test_transform)
    


## Training Logs
```
  Epoch 1:
Loss=1.9319469928741455 Batch_id=390 Accuracy=19.71: 100%|██████████| 391/391 [00:12<00:00, 30.96it/s]

Test set: Average loss: 0.0148, Accuracy: 13456/50000 (26.91%)

Epoch 2:
Loss=1.8663288354873657 Batch_id=390 Accuracy=30.15: 100%|██████████| 391/391 [00:12<00:00, 32.21it/s]

Test set: Average loss: 0.0132, Accuracy: 18225/50000 (36.45%)

Epoch 3:
Loss=1.7375551462173462 Batch_id=390 Accuracy=35.53: 100%|██████████| 391/391 [00:12<00:00, 32.23it/s]

Test set: Average loss: 0.0120, Accuracy: 21070/50000 (42.14%)

Epoch 4:
Loss=1.636221170425415 Batch_id=390 Accuracy=38.80: 100%|██████████| 391/391 [00:12<00:00, 31.82it/s]

Test set: Average loss: 0.0113, Accuracy: 23239/50000 (46.48%)

Epoch 5:
Loss=1.5757230520248413 Batch_id=390 Accuracy=42.33: 100%|██████████| 391/391 [00:12<00:00, 31.75it/s]

Test set: Average loss: 0.0106, Accuracy: 24983/50000 (49.97%)

Epoch 6:
Loss=1.5518168210983276 Batch_id=390 Accuracy=44.49: 100%|██████████| 391/391 [00:12<00:00, 32.13it/s]

Test set: Average loss: 0.0100, Accuracy: 26791/50000 (53.58%)

Epoch 7:
Loss=1.4507933855056763 Batch_id=390 Accuracy=46.75: 100%|██████████| 391/391 [00:12<00:00, 32.15it/s]

Test set: Average loss: 0.0091, Accuracy: 28805/50000 (57.61%)

Epoch 8:
Loss=1.8700387477874756 Batch_id=390 Accuracy=48.35: 100%|██████████| 391/391 [00:12<00:00, 32.24it/s]

Test set: Average loss: 0.0090, Accuracy: 29194/50000 (58.39%)

Epoch 9:
Loss=1.333655834197998 Batch_id=390 Accuracy=50.18: 100%|██████████| 391/391 [00:12<00:00, 32.07it/s]

Test set: Average loss: 0.0085, Accuracy: 30350/50000 (60.70%)

Epoch 10:
Loss=1.3229820728302002 Batch_id=390 Accuracy=51.73: 100%|██████████| 391/391 [00:12<00:00, 32.09it/s]

Test set: Average loss: 0.0083, Accuracy: 31044/50000 (62.09%)

Epoch 11:
Loss=1.2850595712661743 Batch_id=390 Accuracy=53.07: 100%|██████████| 391/391 [00:12<00:00, 32.09it/s]

Test set: Average loss: 0.0080, Accuracy: 31683/50000 (63.37%)

Epoch 12:
Loss=1.0220142602920532 Batch_id=390 Accuracy=54.24: 100%|██████████| 391/391 [00:12<00:00, 32.08it/s]

Test set: Average loss: 0.0075, Accuracy: 32712/50000 (65.42%)

Epoch 13:
Loss=1.1229416131973267 Batch_id=390 Accuracy=55.75: 100%|██████████| 391/391 [00:12<00:00, 31.84it/s]

Test set: Average loss: 0.0074, Accuracy: 33002/50000 (66.00%)

Epoch 14:
Loss=1.3014681339263916 Batch_id=390 Accuracy=56.23: 100%|██████████| 391/391 [00:12<00:00, 32.16it/s]

Test set: Average loss: 0.0073, Accuracy: 33371/50000 (66.74%)

Epoch 15:
Loss=0.9482961893081665 Batch_id=390 Accuracy=57.23: 100%|██████████| 391/391 [00:12<00:00, 32.02it/s]

Test set: Average loss: 0.0070, Accuracy: 33967/50000 (67.93%)

Epoch 16:
Loss=1.1313725709915161 Batch_id=390 Accuracy=57.73: 100%|██████████| 391/391 [00:12<00:00, 32.04it/s]

Test set: Average loss: 0.0072, Accuracy: 33411/50000 (66.82%)

Epoch 17:
Loss=1.3001049757003784 Batch_id=390 Accuracy=58.69: 100%|██████████| 391/391 [00:12<00:00, 32.20it/s]

Test set: Average loss: 0.0066, Accuracy: 34808/50000 (69.62%)

Epoch 18:
Loss=0.9624654054641724 Batch_id=390 Accuracy=59.66: 100%|██████████| 391/391 [00:12<00:00, 32.21it/s]

Test set: Average loss: 0.0066, Accuracy: 34869/50000 (69.74%)

Epoch 19:
Loss=1.2257417440414429 Batch_id=390 Accuracy=60.23: 100%|██████████| 391/391 [00:12<00:00, 32.16it/s]

Test set: Average loss: 0.0062, Accuracy: 36010/50000 (72.02%)

Epoch 20:
Loss=1.0949043035507202 Batch_id=390 Accuracy=60.75: 100%|██████████| 391/391 [00:12<00:00, 32.13it/s]

Test set: Average loss: 0.0068, Accuracy: 34668/50000 (69.34%)

Epoch 21:
Loss=1.0482901334762573 Batch_id=390 Accuracy=61.05: 100%|██████████| 391/391 [00:12<00:00, 32.18it/s]

Test set: Average loss: 0.0062, Accuracy: 35982/50000 (71.96%)

Epoch 22:
Loss=1.0679572820663452 Batch_id=390 Accuracy=62.14: 100%|██████████| 391/391 [00:12<00:00, 32.17it/s]

Test set: Average loss: 0.0061, Accuracy: 36244/50000 (72.49%)

Epoch 23:
Loss=1.026137351989746 Batch_id=390 Accuracy=62.76: 100%|██████████| 391/391 [00:12<00:00, 32.15it/s]

Test set: Average loss: 0.0058, Accuracy: 36817/50000 (73.63%)

Epoch 24:
Loss=1.1055901050567627 Batch_id=390 Accuracy=63.00: 100%|██████████| 391/391 [00:12<00:00, 32.18it/s]

Test set: Average loss: 0.0058, Accuracy: 36945/50000 (73.89%)

Epoch 25:
Loss=0.9780522584915161 Batch_id=390 Accuracy=63.38: 100%|██████████| 391/391 [00:12<00:00, 32.21it/s]

Test set: Average loss: 0.0056, Accuracy: 37384/50000 (74.77%)

Epoch 26:
Loss=1.1005077362060547 Batch_id=390 Accuracy=63.67: 100%|██████████| 391/391 [00:12<00:00, 32.19it/s]

Test set: Average loss: 0.0055, Accuracy: 37608/50000 (75.22%)

Epoch 27:
Loss=1.039062738418579 Batch_id=390 Accuracy=64.21: 100%|██████████| 391/391 [00:12<00:00, 32.16it/s]

Test set: Average loss: 0.0056, Accuracy: 37405/50000 (74.81%)

Epoch 28:
Loss=0.9591469764709473 Batch_id=390 Accuracy=64.40: 100%|██████████| 391/391 [00:12<00:00, 32.23it/s]

Test set: Average loss: 0.0052, Accuracy: 38299/50000 (76.60%)

Epoch 29:
Loss=1.3432056903839111 Batch_id=390 Accuracy=65.06: 100%|██████████| 391/391 [00:12<00:00, 31.91it/s]

Test set: Average loss: 0.0053, Accuracy: 38159/50000 (76.32%)

Epoch 30:
Loss=1.3049218654632568 Batch_id=390 Accuracy=65.22: 100%|██████████| 391/391 [00:12<00:00, 32.20it/s]

Test set: Average loss: 0.0051, Accuracy: 38599/50000 (77.20%)

Epoch 31:
Loss=1.020039677619934 Batch_id=390 Accuracy=65.24: 100%|██████████| 391/391 [00:12<00:00, 32.19it/s]

Test set: Average loss: 0.0052, Accuracy: 38329/50000 (76.66%)

Epoch 32:
Loss=1.0561785697937012 Batch_id=390 Accuracy=65.88: 100%|██████████| 391/391 [00:12<00:00, 32.05it/s]

Test set: Average loss: 0.0050, Accuracy: 38830/50000 (77.66%)

Epoch 33:
Loss=0.7450627088546753 Batch_id=390 Accuracy=66.03: 100%|██████████| 391/391 [00:12<00:00, 32.31it/s]

Test set: Average loss: 0.0051, Accuracy: 38587/50000 (77.17%)

Epoch 34:
Loss=0.9814586639404297 Batch_id=390 Accuracy=66.35: 100%|██████████| 391/391 [00:12<00:00, 32.14it/s]

Test set: Average loss: 0.0050, Accuracy: 38603/50000 (77.21%)

Epoch 35:
Loss=0.8692092895507812 Batch_id=390 Accuracy=67.19: 100%|██████████| 391/391 [00:12<00:00, 32.22it/s]

Test set: Average loss: 0.0049, Accuracy: 38877/50000 (77.75%)

Epoch 36:
Loss=1.0598812103271484 Batch_id=390 Accuracy=67.24: 100%|██████████| 391/391 [00:12<00:00, 32.20it/s]

Test set: Average loss: 0.0047, Accuracy: 39482/50000 (78.96%)

Epoch 37:
Loss=0.7424447536468506 Batch_id=390 Accuracy=67.28: 100%|██████████| 391/391 [00:12<00:00, 32.14it/s]

Test set: Average loss: 0.0047, Accuracy: 39597/50000 (79.19%)

Epoch 38:
Loss=0.9241287112236023 Batch_id=390 Accuracy=67.43: 100%|██████████| 391/391 [00:12<00:00, 32.27it/s]

Test set: Average loss: 0.0049, Accuracy: 39101/50000 (78.20%)

Epoch 39:
Loss=0.8567928075790405 Batch_id=390 Accuracy=67.57: 100%|██████████| 391/391 [00:12<00:00, 32.17it/s]

Test set: Average loss: 0.0047, Accuracy: 39584/50000 (79.17%)

Epoch 40:
Loss=0.82984459400177 Batch_id=390 Accuracy=68.17: 100%|██████████| 391/391 [00:12<00:00, 32.14it/s]

Test set: Average loss: 0.0046, Accuracy: 39819/50000 (79.64%)

Epoch 41:
Loss=1.040114164352417 Batch_id=390 Accuracy=68.09: 100%|██████████| 391/391 [00:12<00:00, 32.16it/s]

Test set: Average loss: 0.0045, Accuracy: 39879/50000 (79.76%)

Epoch 42:
Loss=0.8292804956436157 Batch_id=390 Accuracy=68.56: 100%|██████████| 391/391 [00:12<00:00, 32.26it/s]

Test set: Average loss: 0.0044, Accuracy: 40160/50000 (80.32%)

Epoch 43:
Loss=0.8361564874649048 Batch_id=390 Accuracy=68.48: 100%|██████████| 391/391 [00:12<00:00, 32.23it/s]

Test set: Average loss: 0.0045, Accuracy: 40012/50000 (80.02%)

Epoch 44:
Loss=0.9400971531867981 Batch_id=390 Accuracy=68.89: 100%|██████████| 391/391 [00:12<00:00, 32.16it/s]

Test set: Average loss: 0.0042, Accuracy: 40622/50000 (81.24%)

Epoch 45:
Loss=0.9415705800056458 Batch_id=390 Accuracy=69.23: 100%|██████████| 391/391 [00:12<00:00, 32.21it/s]

Test set: Average loss: 0.0043, Accuracy: 40354/50000 (80.71%)

Epoch 46:
Loss=0.9464705586433411 Batch_id=390 Accuracy=69.00: 100%|██████████| 391/391 [00:12<00:00, 32.19it/s]

Test set: Average loss: 0.0043, Accuracy: 40324/50000 (80.65%)

Epoch 47:
Loss=0.9263855814933777 Batch_id=390 Accuracy=69.41: 100%|██████████| 391/391 [00:12<00:00, 32.17it/s]

Test set: Average loss: 0.0044, Accuracy: 40097/50000 (80.19%)

Epoch 48:
Loss=0.8793121576309204 Batch_id=390 Accuracy=69.46: 100%|██████████| 391/391 [00:12<00:00, 32.14it/s]

Test set: Average loss: 0.0044, Accuracy: 40167/50000 (80.33%)

Epoch 49:
Loss=0.8472908139228821 Batch_id=390 Accuracy=69.33: 100%|██████████| 391/391 [00:12<00:00, 32.15it/s]

Test set: Average loss: 0.0040, Accuracy: 41110/50000 (82.22%)

Epoch 50:
Loss=0.7445904016494751 Batch_id=390 Accuracy=69.59: 100%|██████████| 391/391 [00:12<00:00, 32.14it/s]

Test set: Average loss: 0.0044, Accuracy: 40272/50000 (80.54%)

Epoch 51:
Loss=0.6672817468643188 Batch_id=390 Accuracy=69.76: 100%|██████████| 391/391 [00:12<00:00, 32.21it/s]

Test set: Average loss: 0.0042, Accuracy: 40751/50000 (81.50%)

Epoch 52:
Loss=0.7102442383766174 Batch_id=390 Accuracy=69.93: 100%|██████████| 391/391 [00:12<00:00, 32.23it/s]

Test set: Average loss: 0.0040, Accuracy: 41015/50000 (82.03%)

Epoch 53:
Loss=0.6452576518058777 Batch_id=390 Accuracy=70.34: 100%|██████████| 391/391 [00:12<00:00, 32.20it/s]

Test set: Average loss: 0.0040, Accuracy: 41251/50000 (82.50%)

Epoch 54:
Loss=0.8463571667671204 Batch_id=390 Accuracy=70.67: 100%|██████████| 391/391 [00:12<00:00, 32.07it/s]

Test set: Average loss: 0.0043, Accuracy: 40430/50000 (80.86%)

Epoch 55:
Loss=0.694664478302002 Batch_id=390 Accuracy=70.80: 100%|██████████| 391/391 [00:12<00:00, 32.21it/s]

Test set: Average loss: 0.0040, Accuracy: 41257/50000 (82.51%)

Epoch 56:
Loss=0.6081429123878479 Batch_id=390 Accuracy=70.30: 100%|██████████| 391/391 [00:12<00:00, 32.23it/s]

Test set: Average loss: 0.0039, Accuracy: 41397/50000 (82.79%)

Epoch 57:
Loss=0.7525203227996826 Batch_id=390 Accuracy=70.89: 100%|██████████| 391/391 [00:12<00:00, 32.21it/s]

Test set: Average loss: 0.0039, Accuracy: 41323/50000 (82.65%)

Epoch 58:
Loss=0.5900155901908875 Batch_id=390 Accuracy=71.02: 100%|██████████| 391/391 [00:12<00:00, 32.22it/s]

Test set: Average loss: 0.0038, Accuracy: 41582/50000 (83.16%)

Epoch 59:
Loss=0.7440377473831177 Batch_id=390 Accuracy=71.05: 100%|██████████| 391/391 [00:12<00:00, 32.30it/s]

Test set: Average loss: 0.0037, Accuracy: 41810/50000 (83.62%)

Epoch 60:
Loss=1.1001076698303223 Batch_id=390 Accuracy=71.21: 100%|██████████| 391/391 [00:12<00:00, 32.20it/s]

Test set: Average loss: 0.0038, Accuracy: 41690/50000 (83.38%)

Epoch 61:
Loss=0.8495674133300781 Batch_id=390 Accuracy=71.41: 100%|██████████| 391/391 [00:12<00:00, 32.18it/s]

Test set: Average loss: 0.0037, Accuracy: 41899/50000 (83.80%)

Epoch 62:
Loss=1.0007635354995728 Batch_id=390 Accuracy=71.58: 100%|██████████| 391/391 [00:12<00:00, 32.26it/s]

Test set: Average loss: 0.0036, Accuracy: 41934/50000 (83.87%)

Epoch 63:
Loss=1.0214214324951172 Batch_id=390 Accuracy=71.88: 100%|██████████| 391/391 [00:12<00:00, 32.23it/s]

Test set: Average loss: 0.0037, Accuracy: 41780/50000 (83.56%)

Epoch 64:
Loss=1.0360605716705322 Batch_id=390 Accuracy=71.87: 100%|██████████| 391/391 [00:12<00:00, 32.22it/s]

Test set: Average loss: 0.0037, Accuracy: 41948/50000 (83.90%)

Epoch 65:
Loss=0.7565363645553589 Batch_id=390 Accuracy=72.13: 100%|██████████| 391/391 [00:12<00:00, 32.19it/s]

Test set: Average loss: 0.0036, Accuracy: 41944/50000 (83.89%)

Epoch 66:
Loss=0.6473024487495422 Batch_id=390 Accuracy=72.11: 100%|██████████| 391/391 [00:12<00:00, 32.27it/s]

Test set: Average loss: 0.0036, Accuracy: 42102/50000 (84.20%)

Epoch 67:
Loss=0.8159270286560059 Batch_id=390 Accuracy=72.21: 100%|██████████| 391/391 [00:12<00:00, 32.19it/s]

Test set: Average loss: 0.0034, Accuracy: 42433/50000 (84.87%)

Epoch 68:
Loss=0.5263410806655884 Batch_id=390 Accuracy=72.44: 100%|██████████| 391/391 [00:12<00:00, 32.02it/s]

Test set: Average loss: 0.0035, Accuracy: 42344/50000 (84.69%)

Epoch 69:
Loss=0.6764646172523499 Batch_id=390 Accuracy=72.41: 100%|██████████| 391/391 [00:12<00:00, 32.12it/s]

Test set: Average loss: 0.0034, Accuracy: 42497/50000 (84.99%)

Epoch 70:
Loss=0.6828130483627319 Batch_id=390 Accuracy=72.76: 100%|██████████| 391/391 [00:12<00:00, 32.13it/s]

Test set: Average loss: 0.0034, Accuracy: 42447/50000 (84.89%)

Epoch 71:
Loss=0.8280344009399414 Batch_id=390 Accuracy=73.06: 100%|██████████| 391/391 [00:12<00:00, 31.95it/s]

Test set: Average loss: 0.0033, Accuracy: 42686/50000 (85.37%)

Epoch 72:
Loss=0.7433973550796509 Batch_id=390 Accuracy=73.22: 100%|██████████| 391/391 [00:12<00:00, 31.87it/s]

Test set: Average loss: 0.0034, Accuracy: 42584/50000 (85.17%)

Epoch 73:
Loss=0.7077459096908569 Batch_id=390 Accuracy=72.96: 100%|██████████| 391/391 [00:12<00:00, 31.98it/s]

Test set: Average loss: 0.0033, Accuracy: 42796/50000 (85.59%)

Epoch 74:
Loss=0.787909209728241 Batch_id=390 Accuracy=73.28: 100%|██████████| 391/391 [00:12<00:00, 32.13it/s]

Test set: Average loss: 0.0033, Accuracy: 42706/50000 (85.41%)

Epoch 75:
Loss=0.9525569677352905 Batch_id=390 Accuracy=73.69: 100%|██████████| 391/391 [00:12<00:00, 31.62it/s]

Test set: Average loss: 0.0032, Accuracy: 42823/50000 (85.65%)

Epoch 76:
Loss=0.49505701661109924 Batch_id=390 Accuracy=73.77: 100%|██████████| 391/391 [00:12<00:00, 31.62it/s]

Test set: Average loss: 0.0032, Accuracy: 43014/50000 (86.03%)

Epoch 77:
Loss=0.7295758128166199 Batch_id=390 Accuracy=73.87: 100%|██████████| 391/391 [00:12<00:00, 32.07it/s]

Test set: Average loss: 0.0032, Accuracy: 42987/50000 (85.97%)

Epoch 78:
Loss=0.7520471811294556 Batch_id=390 Accuracy=74.07: 100%|██████████| 391/391 [00:12<00:00, 32.11it/s]

Test set: Average loss: 0.0031, Accuracy: 43205/50000 (86.41%)

Epoch 79:
Loss=0.680796205997467 Batch_id=390 Accuracy=74.07: 100%|██████████| 391/391 [00:12<00:00, 32.19it/s]

Test set: Average loss: 0.0031, Accuracy: 43082/50000 (86.16%)

Epoch 80:
Loss=0.8116587400436401 Batch_id=390 Accuracy=74.22: 100%|██████████| 391/391 [00:12<00:00, 32.26it/s]

Test set: Average loss: 0.0031, Accuracy: 43176/50000 (86.35%)

Epoch 81:
Loss=0.7192744016647339 Batch_id=390 Accuracy=74.51: 100%|██████████| 391/391 [00:12<00:00, 32.24it/s]

Test set: Average loss: 0.0030, Accuracy: 43301/50000 (86.60%)

Epoch 82:
Loss=1.027642846107483 Batch_id=390 Accuracy=74.21: 100%|██████████| 391/391 [00:12<00:00, 32.20it/s]

Test set: Average loss: 0.0030, Accuracy: 43289/50000 (86.58%)

Epoch 83:
Loss=0.8259971737861633 Batch_id=390 Accuracy=74.73: 100%|██████████| 391/391 [00:12<00:00, 32.16it/s]

Test set: Average loss: 0.0030, Accuracy: 43350/50000 (86.70%)

Epoch 84:
Loss=0.7256308197975159 Batch_id=390 Accuracy=74.52: 100%|██████████| 391/391 [00:12<00:00, 32.08it/s]

Test set: Average loss: 0.0030, Accuracy: 43473/50000 (86.95%)

Epoch 85:
Loss=0.6837905645370483 Batch_id=390 Accuracy=74.70: 100%|██████████| 391/391 [00:12<00:00, 32.18it/s]

Test set: Average loss: 0.0030, Accuracy: 43453/50000 (86.91%)

Epoch 86:
Loss=0.7762473821640015 Batch_id=390 Accuracy=74.97: 100%|██████████| 391/391 [00:12<00:00, 32.20it/s]

Test set: Average loss: 0.0030, Accuracy: 43423/50000 (86.85%)

Epoch 87:
Loss=0.7077261805534363 Batch_id=390 Accuracy=75.13: 100%|██████████| 391/391 [00:12<00:00, 32.36it/s]

Test set: Average loss: 0.0030, Accuracy: 43400/50000 (86.80%)

Epoch 88:
Loss=0.774002730846405 Batch_id=390 Accuracy=75.12: 100%|██████████| 391/391 [00:12<00:00, 32.29it/s]

Test set: Average loss: 0.0030, Accuracy: 43442/50000 (86.88%)

Epoch 89:
Loss=0.6279183626174927 Batch_id=390 Accuracy=74.81: 100%|██████████| 391/391 [00:12<00:00, 32.17it/s]

Test set: Average loss: 0.0030, Accuracy: 43462/50000 (86.92%)

Epoch 90:
Loss=0.6579517126083374 Batch_id=390 Accuracy=75.06: 100%|██████████| 391/391 [00:12<00:00, 31.90it/s]

Test set: Average loss: 0.0030, Accuracy: 43467/50000 (86.93%)

```

## Training and validation accuracy curves
![plots](https://github.com/NSR9/EVA8/blob/main/Session-6/s6-img.png))

## Class-wise performance:
```
Accuracy of plane : 88 %
Accuracy of   car : 92 %
Accuracy of  bird : 77 %
Accuracy of   cat : 76 %
Accuracy of  deer : 90 %
Accuracy of   dog : 72 %
Accuracy of  frog : 86 %
Accuracy of horse : 87 %
Accuracy of  ship : 94 %
Accuracy of truck : 94 %
```



