```text
Random seed set to 42; Cudnn benchmarking is False.
Number of sequences passed to encode	: 1000
Dropped due to more N count (> 3)	: 6
Dropped due to outside length spec	: 19

Camformer (Original) model structure:
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1          [-1, 512, 110, 1]          20,992
        Conv2dSame-2          [-1, 512, 110, 1]               0
       BatchNorm2d-3          [-1, 512, 110, 1]           1,024
              ReLU-4          [-1, 512, 110, 1]               0
         MaxPool2d-5          [-1, 512, 110, 1]               0
           Dropout-6          [-1, 512, 110, 1]               0
            Conv2d-7          [-1, 512, 110, 1]       2,621,952
        Conv2dSame-8          [-1, 512, 110, 1]               0
       BatchNorm2d-9          [-1, 512, 110, 1]           1,024
             ReLU-10          [-1, 512, 110, 1]               0
        MaxPool2d-11          [-1, 512, 110, 1]               0
          Dropout-12          [-1, 512, 110, 1]               0
           Conv2d-13          [-1, 512, 110, 1]       2,621,952
       Conv2dSame-14          [-1, 512, 110, 1]               0
      BatchNorm2d-15          [-1, 512, 110, 1]           1,024
             ReLU-16          [-1, 512, 110, 1]               0
        MaxPool2d-17          [-1, 512, 110, 1]               0
          Dropout-18          [-1, 512, 110, 1]               0
           Conv2d-19          [-1, 512, 110, 1]       2,621,952
       Conv2dSame-20          [-1, 512, 110, 1]               0
      BatchNorm2d-21          [-1, 512, 110, 1]           1,024
             ReLU-22          [-1, 512, 110, 1]               0
        MaxPool2d-23          [-1, 512, 110, 1]               0
          Dropout-24          [-1, 512, 110, 1]               0
           Conv2d-25          [-1, 512, 110, 1]       2,621,952
       Conv2dSame-26          [-1, 512, 110, 1]               0
      BatchNorm2d-27          [-1, 512, 110, 1]           1,024
             ReLU-28          [-1, 512, 110, 1]               0
        MaxPool2d-29           [-1, 512, 26, 1]               0
          Dropout-30           [-1, 512, 26, 1]               0
           Conv2d-31           [-1, 512, 26, 1]       2,621,952
       Conv2dSame-32           [-1, 512, 26, 1]               0
      BatchNorm2d-33           [-1, 512, 26, 1]           1,024
             ReLU-34           [-1, 512, 26, 1]               0
        MaxPool2d-35           [-1, 512, 26, 1]               0
          Dropout-36           [-1, 512, 26, 1]               0
           Linear-37                  [-1, 256]       3,408,128
             ReLU-38                  [-1, 256]               0
          Dropout-39                  [-1, 256]               0
           Linear-40                  [-1, 256]          65,792
             ReLU-41                  [-1, 256]               0
          Dropout-42                  [-1, 256]               0
           Linear-43                    [-1, 1]             257
================================================================
Total params: 16,611,073
Trainable params: 16,611,073
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 12.86
Params size (MB): 63.37
```

Estimated Total Size (MB): 76.22
----------------------------------------------------------------
```
