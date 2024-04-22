# CNNBuilder
## Overview

`CNNBuilder.py` is a Python module designed for constructing, training, and manipulating Convolutional Neural Networks (CNNs) using PyTorch. It includes support for creating standard and residual network architectures, quantization, pruning, and conversion to sparse formats. This module is particularly useful for researchers and developers working on deep learning projects involving image recognition tasks with CIFAR-10 dataset as a built-in example.

## Features

- **Model Creation**: Generate standard CNN or ResNet architectures.
- **Quantization**: Apply quantization to reduce model size and potentially improve inference speed on compatible hardware.
- **Pruning**: Implement pruning to increase sparsity within the model, potentially leading to reduced model size and faster inference.
- **Sparse Conversion**: Convert model weights to a sparse format post-training.
- **Training and Evaluation**: Train models with detailed configurations and evaluate them using CIFAR-10.

## Usage
### Creating a Model
To create a model, use the makeModel method from the CNNFactory class. Specify the architecture type ('cnn' or 'resnet'), and whether to enable quantization:
```python
ResNetModel = CNNFactory.makeModel(architecture_type='resnet', res_block_configs=res_block_configs, num_classes=10)
CNNModel = CNNFactory.makeModel(architecture_type='cnn', conv_layer_configs=conv_layer_configs, fc_layer_configs=fc_layer_configs, num_classes=10)
```
You need to specify whether or not a model will be quantized or pruned by setting their respective paramaters to false.

### Training a Model
To train a model, use the trainModel method. You can specify epochs, whether to enable quantization or pruning, and other training parameters:
```python
trained_model_quantized = CNNFactory.trainModel(model, EPOCHS = 30, ENABLE_QUANTIZATION = True, DATA_ROOT = "./data", Save_Name = Save_Name)
trained_model_pruned = CNNFactory.trainModel(model, EPOCHS = 30, ENABLE_PRUNING=True, pruning_epochs = 20, starting_sparsity = 0.1, target_sparsity = 0., DATA_ROOT = "./data", Save_Name = Save_Name)
```
Once again you need to set paramaters specific to quantization and pruning if you would like your training to leverage work quantized or pruned models

### Saving and Loading Models
Save a model using the saveModel method:
```python
CNNFactory.saveModel(model, './saved_models', 'model_name')
```
Load a model with the loadModel method. This method can be directly configure to load a specific type of model or it can leverage an existing model as a reference for the architecture:
```python
### configuration to load a model
loaded_model = CNNFactory.loadModel(path = "./saved_models", name = saveName, architecture_type='resnet', quantTrue=True, res_block_configs=res_block_configs, num_classes=10)
## configuration to load with a reference model
CNNFactory.loadModel(reference_model=model, path = "./saved_models", name = "ResNet")
```
If you are loading a model that has been quantized (not just QAT actually quantized) then make sure you set quantized to True. quantTrue is seperate from this and is for loading a model that has been quantization trained (QAT) but not yet quantized.
