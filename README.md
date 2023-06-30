# Session 9 Assignment

Target of the assignment is to build a neural network that not only obtains a 85% accuracy on the CIFAR10 dataset but also satisfies the following constraints:
- Number of parameters < 200K
- C1C2C3C4O architecture where $C_i$ refers to a convolution block made up of 3 convolution layers and $O$ should necessarily contain GAP
- Contains at least one dilated convolution and depthwise separable convolution layer
- Total RF > 44
- 3 albumentation transforms - Horizontal Flip, Shift Scale Rotate and Coarse Dropout (~Cutout)
- Modular code

## Repo Breakdown
- Data transforms in albumentation are present in `transform.py`
- Custom model classes with subcomponent definitions are in `model.py`
- Trainer class with corresponding train & test methods are in `training.py`
- Utility & convenience functions are bundled together in `utils.py`
- Highest performing model weights are found in the `weights` directory

## Model Summary

````
===============================================================================================
Layer (type:depth-idx)                        Output Shape              Param #
===============================================================================================
Net                                           [10]                      --
├─ConvBlock: 1-1                              --                        --
│    └─Sequential: 2-1                        --                        --
│    │    └─Sequential: 3-1                   [1, 32, 32, 32]           219
│    │    └─Sequential: 3-2                   [1, 32, 32, 32]           1,408
│    │    └─Sequential: 3-3                   [1, 32, 32, 32]           1,408
├─ConvBlock: 1-2                              --                        --
│    └─Sequential: 2-2                        --                        --
│    │    └─Sequential: 3-4                   [1, 64, 32, 32]           2,528
│    │    └─Sequential: 3-5                   [1, 64, 32, 32]           4,864
│    │    └─Sequential: 3-6                   [1, 64, 32, 32]           4,864
├─ConvBlock: 1-3                              --                        --
│    └─Sequential: 2-3                        --                        --
│    │    └─Sequential: 3-7                   [1, 128, 32, 32]          9,152
│    │    └─Sequential: 3-8                   [1, 128, 32, 32]          17,920
│    │    └─Sequential: 3-9                   [1, 128, 32, 32]          17,920
├─ConvBlock: 1-4                              --                        --
│    └─Sequential: 2-4                        --                        --
│    │    └─Sequential: 3-10                  [1, 224, 32, 32]          30,496
│    │    └─Sequential: 3-11                  [1, 224, 32, 32]          52,864
│    │    └─Sequential: 3-12                  [1, 224, 32, 32]          52,864
├─Sequential: 1-5                             [1, 10, 1, 1]             --
│    └─AdaptiveAvgPool2d: 2-5                 [1, 224, 1, 1]            --
│    └─Conv2d: 2-6                            [1, 10, 1, 1]             2,250
===============================================================================================
Total params: 198,757
Trainable params: 198,757
Non-trainable params: 0
Total mult-adds (M): 198.48
===============================================================================================
Input size (MB): 0.01
Forward/backward pass size (MB): 31.22
Params size (MB): 0.80
Estimated Total Size (MB): 32.03
===============================================================================================
````

## Intuitions & Observation

- I chose depthwise separable convolution layers as the main workhorse given the lower parameter count of depthwise separable convolution layers over traditional pointwise convolution layers without that big of a compromise on performance
- This also allowed a much higher kernel count for each depthwise separable convolution layer without going over 200K parameter limit!
- Given the additional points (!) and RF > 44 requirement, I implemented dilated convolution layers as the 3rd layer in every convolution block
- Albumentation transformations were made interoperable with torchvision dataloaders by creating a custom class around `A.Compose(...)` which returns the transformed numpy array images (as opposed to dictionaries)
- I unfortunately did not hit the 85% accuracy mark because of a shortage of time (I was using 30 epochs). Future work include:
  - Revamping module design (code training with custom classes taking ~1min23s, an order of magnitude higher than using default classes)
  - Setting baseline with strided convolutions as well
  - Experimenting with one cycle policy (currently using LR on plateau which takes far too long to update)

