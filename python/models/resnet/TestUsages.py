import CNNBuilder
from CNNBuilder import CNNFactory
import torch

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

ResNetConfig = {
    'architecture_type': 'resnet',
    'quantTrue': False,
    'res_block_configs': res_block_configs,
    'num_classes': 10
    }

CNNConfig = {
    'architecture_type': 'cnn',
    'quantTrue': False,
    'conv_layer_configs': conv_layer_configs,
    'fc_layer_configs': fc_layer_configs,
    'num_classes': 10
    }

ResNetQuantConfig = {
    'architecture_type': 'resnet',
    'quantTrue': True,
    'res_block_configs': res_block_configs,
    'num_classes': 10

}
def main():
    # test1()
    # test2()

    # This is a config test for savin and loading a QUANTIZED model using configs
    net = CNNFactory.makeModel(**ResNetQuantConfig)

    CNNFactory.quantizeModel(net)
    CNNFactory.saveModel(net, "./saved_models", save_name="ResNetQuantConfigTest", config=ResNetConfig, quantized=True)

    device = torch.device("cuda:2")
    net = CNNFactory.loadModel(path = "./saved_models", name = "ResNetConfigTest")
    CNNFactory.evaluate_model(model=net, device=device, batch_size=1, num_steps=100, DATA_ROOT="./data")

def test1():
    # You can make and Train a normal Resnet like in this function
    normalResnet = makeNormalResNet(Save_Name = "ResNet_Normal")
    
    # You can make and Train a quantized Resnet like in this function
    quantizedResnet = makeAndTrainQuantizedResNet(Save_Name = "ResNet_QuantizationTrained")
    CNNFactory.saveModel(quantizedResnet, "./saved_models", save_name="ResNet_Quantized") ## This is also saving the quantized version of the model

    # You can make and Train a CNN like in this function
    prunedCNN = makeAndTrainCNNPruned(Save_Name = "CNN_Pruned")

    # You can load a model from a pth file like this
    ## either using the same setup you used for training
    quantizationTrainedResNet = CNNFactory.loadModel(path = "./saved_models", name = "ResNet_QuantizationTrained", architecture_type='resnet', quantTrue=True, res_block_configs=res_block_configs, num_classes=10)
    ## or passing it a reference model
    quantizationTrainedResNet = CNNFactory.loadModel(path = "./saved_models", name = "ResNet_QuantizationTrained", reference_model=quantizationTrainedResNet)
    ## Remember if you are loading a model that has been qunatized you need to specify quantized=True when loading from a pth
    quantizedResNetLoaded= CNNFactory.loadModel(path = "./saved_models", name = "ResNet_Quantized", architecture_type='resnet', quantTrue=True, res_block_configs=res_block_configs, num_classes=10, quantized=True)
    
    print("EOT")

def test2():
    # Here is some more explicit saving and loading example without function abstraction or training
    net = CNNFactory.makeModel(architecture_type='resnet', quantTrue=True, res_block_configs=res_block_configs, num_classes=10)
    CNNFactory.saveModel(net, "./saved_models", save_name="ResNetTest1")
    net = CNNFactory.loadModel(path = "./saved_models", name = "ResNetTest1", architecture_type='resnet', quantTrue=True, res_block_configs=res_block_configs, num_classes=10)
    CNNFactory.quantizeModel(net)
    CNNFactory.saveModel(net, "./saved_models", save_name="ResNetTest2")
    net = CNNFactory.loadModel(path = "./saved_models", name = "ResNetTest2", architecture_type='resnet', quantTrue=True, res_block_configs=res_block_configs, num_classes=10, quantized=True) #Note that you need to specify quantized=True when loading a quantized model
    net = CNNFactory.loadModel(reference_model=net, path = "./saved_models", name = "ResNetTest2")
    print("EOT")

def makeNormalResNet(Save_Name = "ResNet_Normal"):
    net = CNNFactory.makeModel(architecture_type='resnet', quantTrue=False, res_block_configs=res_block_configs, num_classes=10) #This is how you make a resnet model
    net = CNNFactory.trainModel(net, EPOCHS = 30, ENABLE_QUANTIZATION = False, DATA_ROOT = "./data", Save_Name = Save_Name) #This is how you train a model
    print ("ResNet size before quantization:")
    CNNFactory.checkSize(net) # This is how you check the size of a model
    return net


def makeAndTrainQuantizedResNet(Save_Name = "ResNet_Quantization_Trained"):
    net = CNNFactory.makeModel(architecture_type='resnet', quantTrue=True, res_block_configs=res_block_configs, num_classes=10) #This is how you make a resnet quantized model
    net = CNNFactory.trainModel(net, EPOCHS = 1, ENABLE_QUANTIZATION = True, DATA_ROOT = "./data", Save_Name = Save_Name) #This is how you train a model
    print ("Quantized ResNet size before quantization:")
    CNNFactory.checkSize(net) # This is how you check the size of a model
    CNNFactory.quantizeModel(net) # Keep in mind you still need to quantize the model to get significant size reduction also this method also qunatizes our model
    print ("Quantized ResNet size after quantization:")
    CNNFactory.checkSize(net)
    return net


def makeAndTrainCNNPruned(Save_Name = "CNN_Pruned"):
    net = CNNFactory.makeModel(architecture_type='cnn', conv_layer_configs=conv_layer_configs, fc_layer_configs=fc_layer_configs, num_classes=10)
    net = CNNFactory.trainModel(net, EPOCHS = 3, ENABLE_PRUNING=True, pruning_epochs = 2, starting_sparsity = 0.1, target_sparsity = 0.5, DATA_ROOT = "./data", Save_Name = Save_Name)
    print ("CNN size after pruning (for some reason larger right now):")
    CNNFactory.checkSize(net)
    return net

if __name__ == "__main__":
    main()
