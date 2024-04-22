import CNNBuilder
from CNNBuilder import CNNFactory

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

def main():
    # You can make and Train a Resnet like in this function
    quantizedResnet = makeAndTrainQuantizedResNet(Save_Name = "ResNet_QuantizationTrained.pth")
    # You can make and Train a CNN like in this function
    # prunedCNN = makeAndTrainCNNPruned(Save_Name = "CNN_Pruned.pth")
    # You can load a model from a pth file like this
    ## either using the same setup you used for training
    loadedModel = CNNFactory.loadModel(model_path = "./saved_models/ResNet_QuantizationTrained.pth.pth", architecture_type='resnet', quantTrue=True, res_block_configs=res_block_configs, num_classes=10)
    ## or passing it a reference model
    CNNFactory.saveModel(quantizedResnet, "./saved_models", save_name="ResNet_Quantized.pth")
    loadedModel = CNNFactory.loadModel(model_path = "./saved_models/ResNet_Quantized.pth.pth", reference_model = quantizedResnet)
    # You can also save a model to a pth file like this
    CNNFactory.saveModel(loadedModel, "./saved_models/ResNet_Quantized.pth", save_name="loadedModel.pth")


def makeAndTrainQuantizedResNet(Save_Name = "ResNet_Quantized.pth"):
    net = CNNFactory.makeModel(architecture_type='resnet', quantTrue=True, res_block_configs=res_block_configs, num_classes=10) #This is how you make a resnet quantized model
    net = CNNFactory.trainModel(net, EPOCHS = 1, ENABLE_QUANTIZATION = True, DATA_ROOT = "./data", Save_Name = Save_Name) #This is how you train a model
    print ("Quantized ResNet size before quantization:")
    CNNFactory.checkSize(net) # This is how you check the size of a model
    CNNFactory.quantizeModel(net) # Keep in mind you still need to quantize the model to get significant size reduction
    print ("Quantized ResNet size after quantization:")
    CNNFactory.checkSize(net)
    return net


def makeAndTrainCNNPruned(Save_Name = "CNN_Pruned.pth"):
    net = CNNFactory.makeModel(architecture_type='cnn', conv_layer_configs=conv_layer_configs, fc_layer_configs=fc_layer_configs, num_classes=10)
    net = CNNFactory.trainModel(net, EPOCHS = 3, ENABLE_PRUNING=True, pruning_epochs = 2, starting_sparsity = 0.1, target_sparsity = 0.5, DATA_ROOT = "./data", Save_Name = Save_Name)
    print ("CNN size after pruning (for some reason larger right now):")
    CNNFactory.checkSize(net)
    return net

if __name__ == "__main__":
    main()
