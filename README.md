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
```bash
pip install numpy matplotlib imutils scikit-learn torch torchvision opencv-python
```
## Usage
To run the code, follow these steps:

1. **Save the Datasets**: Download the datasets (linked below) and save them to your Google Drive.
2. **Open the Notebook**: Open the `.ipynb` notebook provided in this repository in Google Colab.
3. **Connect Google Drive**: In the Colab notebook, connect your Google Drive by running the cell that mounts it (provided in the notebook).
4. **Run the Code**: Execute all the cells in the notebook to replicate the experiments.

## Datasets
We used three different datasets to evaluate the performance of our models. Below are the references and links to access these datasets:

Breast Ultrasound Images Dataset
```
Citation:
@article{al2020dataset,
title={Dataset of breast ultrasound images},
author={Al-Dhabyani, Walid and Gomaa, Mohammed and Khaled, Hussien and Fahmy, Aly},
journal={Data in brief},
volume={28},
pages={104863},
year={2020},
publisher={Elsevier}
}
```

Skin Lesion Dataset (ISIC 2016)
```
Citation:
@article{gutman2016skin,
title={Skin lesion analysis toward melanoma detection: A challenge at the international symposium on biomedical imaging (ISBI) 2016, hosted by the international skin imaging collaboration (ISIC)},
author={Gutman, David and Codella, Noel CF and Celebi, Emre and Helba, Brian and Marchetti, Michael and Mishra, Nabin and Halpern, Allan},
journal={arXiv preprint arXiv:1605.01397},
year={2016}
}
```
Skin Lesion Dataset (ISIC 2017)
```
Citation:
@inproceedings{codella2018skin,
title={Skin lesion analysis toward melanoma detection: A challenge at the 2017 international symposium on biomedical imaging (ISBI), hosted by the international skin imaging collaboration (ISIC)},
author={Codella, Noel CF and Gutman, David and Celebi, M Emre and others},
booktitle={2018 IEEE 15th international symposium on biomedical imaging (ISBI 2018)},
pages={168--172},
year={2018},
organization={IEEE}
}
```
