# Super-Resolution using ESRGAN

This repository contains an implementation of the Enhanced Super-Resolution Generative Adversarial Network (ESRGAN) for single image super-resolution. The model is trained to generate high-resolution images from low-resolution input images.

## Model Architecture

The generator model is based on the ESRGAN architecture, which consists of the following components:
- Initial convolutional layer
- Residual in Residual Dense Blocks (RRDB)
- Convolutional layer
- Upsampling blocks
- Final convolutional layers

The pre-trained generator model weights are stored in `gen.pth`.

## Results

We demonstrate the performance of the super-resolution model by comparing the original low-resolution image, the super-resolution image generated by our model, and the image upscaled using bicubic interpolation. The comparison is performed using the Inception Score (IS) metric.

![Comparison of Original, Super-Resolution](./Model/test_4/original_image.png)(./Model/test_4/super_resolution_image.png.png)

As shown in the figure, the super-resolution image generated by our model achieves a higher Inception Score compared to the original low-resolution image and the bicubic upscaled image. This indicates that the super-resolution model is capable of generating high-quality, realistic images.

## Usage

To run the super-resolution demo, execute the `test.ipynb` notebook. The notebook performs the following steps:
1. Load the pre-trained generator model from `gen.pth`.
2. Download a random low-resolution image from the internet.
3. Generate the super-resolution image using the loaded model.
4. Perform 4x bicubic upscaling for comparison.
5. Calculate the Inception Score for the original, super-resolution, and bicubic upscaled images.
6. Plot the results for visual comparison.

## File Paths

- `gen.pth`: Pre-trained generator model weights.
- `test.ipynb`: Jupyter notebook for running the super-resolution demo.
- `config.py`: Configuration file containing test image transformations.

Models can be downloaded from the google drive: 
```
https://drive.google.com/file/d/1DI_ME8z8CF3xyRQ4d7m45kEHt9I1R-Aq/view?usp=sharing, 
https://drive.google.com/file/d/1XCtnYkdWp0-Oo94LzutxsQ140TCazf-p/view?usp=sharing
```
## Requirements

- Python 3.x
- PyTorch
- torchvision
- matplotlib
- Pillow
- requests
- scipy
- numpy

## Acknowledgements

This implementation is based on the ESRGAN paper:

- [ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks](https://arxiv.org/abs/1809.00219)

The Inception Score metric is introduced in:

- [Improved Techniques for Training GANs](https://arxiv.org/abs/1606.03498)

## License

This project is licensed under the [MIT License](LICENSE).