# 🧠 From Dense Nets to ResNets: MNIST & CIFAR-10

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow%20%2F%20Keras-FF6F00?logo=tensorflow&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white)

A hands-on progression through image-classification architectures in Keras, from a plain fully-connected network to convolutional nets and residual blocks, all in one notebook.

## Results on MNIST

| Model | Test accuracy |
|---|---|
| Fully-connected ANN | **97.96%** |
| CNN (Conv2D + MaxPooling) | **99.05%** |

Adding convolutions cuts the error rate roughly in half, because the network can finally exploit the spatial structure of digits.

## What's inside `main file.ipynb`

1. **Simple ANN on MNIST:** a dense network as the baseline.
2. **CNN on MNIST:** convolution and pooling layers for spatial feature learning.
3. **CIFAR-10 pipeline:** loads the raw CIFAR-10 Python batches straight from the pickled files.
4. **ANN vs. CNN on CIFAR-10:** the same comparison on harder 32×32 colour images across 10 classes.
5. **ResNet-style model:** residual blocks built with the Keras functional API (`Add` skip connections).

## Run

```bash
pip install tensorflow numpy
jupyter notebook "main file.ipynb"
```

The MNIST (`mnist.npz`) and CIFAR-10 (`cifar-10-batches-py/`) datasets are included, so it runs offline.
