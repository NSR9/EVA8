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
   coarseDropout (max_holes = 1, max_height=16px, max_width=1, min_holes = 1, min_height=16px, min_width=16px, fill_value=(mean of your dataset), mask_fill_value = None)
8. achieve 85% accuracy, as many epochs as you want. Total Params to be less than 200k.
9. upload to Github


## Model 
Below is the model code that is modified as per assignment.
'''



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
'''

## Model Summary

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




