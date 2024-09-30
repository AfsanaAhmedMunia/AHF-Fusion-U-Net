# AHF-Fusion-U-Net
Attention-Guided Hierarchical Fusion U-Net for Uncertainty-driven Medical Image Segmentation
# Attention-Guided Hierarchical Fusion U-Net for Uncertainty-driven Medical Image Segmentation

## Description
This repository contains the code for the paper titled **"Attention-Guided Hierarchical Fusion U-Net for Uncertainty-driven Medical Image Segmentation"**. In this paper, we propose two versions of a network:
1. A regular segmentation network.
2. An uncertainty-aware segmentation network.

The base code for the U-Net architecture was adapted from the following GitHub repository:  
[https://github.com/milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet)

## Installation and Libraries
The code was executed using **Google Colab Pro**. Below are the libraries required to run the code:

```python
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
from imutils import paths
from sklearn.model_selection import train_test_split
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.transforms import CenterCrop
from torch.utils.data import DataLoader, Dataset
import os
import cv2
```

To install the required libraries in your environment, you can use the following command:

pip install numpy matplotlib imutils scikit-learn torch torchvision opencv-python

