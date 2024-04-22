# CNNBuilder
## Overview

`CNNBuilder.py` is a Python module designed for constructing, training, and manipulating Convolutional Neural Networks (CNNs) using PyTorch. It includes support for creating standard and residual network architectures, quantization, pruning, and conversion to sparse formats. This module is particularly useful for researchers and developers working on deep learning projects involving image recognition tasks with CIFAR-10 dataset as a built-in example.

## Features

- **Model Creation**: Generate standard CNN or ResNet architectures.
- **Quantization**: Apply quantization to reduce model size and potentially improve inference speed on compatible hardware.
- **Pruning**: Implement pruning to increase sparsity within the model, potentially leading to reduced model size and faster inference.
- **Sparse Conversion**: Convert model weights to a sparse format post-training.
- **Training and Evaluation**: Train models with detailed configurations and evaluate them using CIFAR-10.
- **Model Configs**: Instead of explicitly configuring models every time you can also use model configs to quickly load and make CNNs

## Usage
### Creating a Model
To create a model, use the makeModel method from the CNNFactory class. Specify the architecture type ('cnn' or 'resnet'), and whether to enable quantization:
```python
conv_layer_configs = [
{'in_channels': 3, 'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1},
{'in_channels': 32, 'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1},
{'in_channels': 64, 'out_channels': 128, 'kernel_size': 3, 'stride': 2, 'padding': 1}
]

fc_layer_configs = [
    {'out_features': 256},
    {'out_features': 128}
]

res_block_configs = [
    {'in_channels': 3, 'out_channels': 16, 'stride': 1, 'num_blocks': 3},
    {'in_channels': 16, 'out_channels': 32, 'stride': 2, 'num_blocks': 3},
    {'in_channels': 32, 'out_channels': 64, 'stride': 2, 'num_blocks': 3}
]

ResNetModel = CNNFactory.makeModel(architecture_type='resnet', res_block_configs=res_block_configs, num_classes=10)
CNNModel = CNNFactory.makeModel(architecture_type='cnn', conv_layer_configs=conv_layer_configs, fc_layer_configs=fc_layer_configs, num_classes=10)
```
You can also use model configs to make a model
```python

ResNetConfig = {
    'architecture_type': 'resnet',
    'quantTrue': False,
    'res_block_configs': res_block_configs,
    'num_classes': 10
    }

    ResNetModel = CNNFactory.makeModel(**ResNetConfig)
```
You need to specify whether or not a model will be a QAT (quantization aware) model by setting quantTrue to True or False

### Training a Model
To train a model, use the trainModel method. You can specify epochs, whether to enable quantization or pruning, and other training parameters:
```python
trained_model_quantized = CNNFactory.trainModel(model, EPOCHS = 30, ENABLE_QUANTIZATION = True, DATA_ROOT = "./data", Save_Name = Save_Name)
trained_model_pruned = CNNFactory.trainModel(model, EPOCHS = 30, ENABLE_PRUNING=True, pruning_epochs = 20, starting_sparsity = 0.1, target_sparsity = 0., DATA_ROOT = "./data", Save_Name = Save_Name)
```
Once again you need to set paramaters specific to quantization and now also pruning if you would like your training to leverage quantized or pruned model architectures

### Saving and Loading Models
Save a model using the saveModel method:
```python
CNNFactory.saveModel(model, './saved_models', 'model_name')
```
You can include a config and quantization/pruning paramaters to make it easier to load in the future
```python
CNNFactory.saveModel(net, "./saved_models", save_name="ModelWithConfig", config=ModelConfig, quantized = false) # quantized = false is unnecessary but included for clarity
```
Load a model with the loadModel method. This method can directly load a model from a pth if that model has a config. If not you can specificy the configuration of the model or you can leverage an existing model as a reference for the configuration (make sure everyhting matches up though):
```python
# load a model with a config
## The paramaters for quantized and pruned should be saved in the pth as well so no need to specify
model = CNNFactory.loadModel(path = "./saved_models", name = "ModelWithConfig") # The paramaters for quantized and pruned should be saved in the pth as well so no need to specify
# configuration to load a model
model = CNNFactory.loadModel(path = "./saved_models", name = saveName, architecture_type='resnet', quantTrue=True, res_block_configs=res_block_configs, num_classes=10)
# configuration to load with a reference model
model = CNNFactory.loadModel(reference_model=model, path = "./saved_models", name = "ResNet")
```
If you are loading a model that has been quantized (not just QAT actually quantized) then make sure you set quantized to True. quantTrue is seperate from quantized. quantTrue is for loading a model that has been quantization trained (QAT) but not yet quantized.
