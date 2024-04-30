# Super-Resolution with RRDBNet

This repository contains code for performing image super-resolution using the RRDBNet architecture. RRDBNet is a deep neural network architecture designed for single image super-resolution, capable of generating high-quality high-resolution images from low-resolution inputs.

## Overview

Image super-resolution is the process of enhancing the resolution and quality of an image. This is particularly useful in scenarios where high-resolution images are required but only low-resolution images are available. RRDBNet achieves this by learning a mapping from low-resolution images to high-resolution images, leveraging the power of deep learning techniques.

## Features

- **RRDBNet Architecture**: Utilizes a modified Residual in Residual Dense Block (RRDB) architecture for image super-resolution.
- **Generator and Discriminator**: Implements a generator and discriminator network for training RRDBNet using adversarial learning.
- **Inception Score Evaluation**: Evaluates the quality of super-resolved images using Inception Score.
- **Video Super-Resolution**: Applies super-resolution to videos, enhancing their quality.

## Requirements

```plaintext
torch>=1.7.1
torchvision>=0.8.2
numpy>=1.19.2
matplotlib>=3.3.2
Pillow>=8.0.1
requests>=2.24.0
scikit-image>=0.17.2
opencv-python>=4.5.1.48
albumentations>=0.5.2

# Steps

git clone https://github.com/yagnaVNK/ESRGAN.git

pip install -r requirements.txt

