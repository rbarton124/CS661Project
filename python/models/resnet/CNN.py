# ===== Import necessary libraries =====
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torch.quantization

import time


# ===== Set up the SimpleNN model =====

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
class ResBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion*out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion*out_channels, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(self.expansion*out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class CNN(nn.Module):
    def __init__(self, architecture_type='cnn', conv_layer_configs=None, fc_layer_configs=None, res_block_configs=None, num_classes=10):
        super(CNN, self).__init__()
        
        # Initialization code for other parts of the class remains the same
        
        if architecture_type == 'cnn':
            self.myNetworkType = 'cnn'
            self.features = self._make_cnn_layers(conv_layer_configs)
            prev_features = conv_layer_configs[-1]['out_channels']
        elif architecture_type == 'resnet':
            self.myNetworkType = 'resnet'
            self.init_conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
            self.init_bn = nn.BatchNorm2d(16)
            self.init_relu = nn.ReLU(inplace=True)
            self.features, prev_features = self._make_resnet_layers(res_block_configs)
        else:
            raise ValueError("Unsupported architecture type: {}".format(architecture_type))
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = self._make_fc_layers(prev_features, fc_layer_configs, num_classes)

    def _make_cnn_layers(self, conv_layer_configs):
        layers = []
        for conv_layer in conv_layer_configs:
            in_channels = conv_layer['in_channels']
            out_channels = conv_layer['out_channels']
            kernel_size = conv_layer['kernel_size']
            stride = conv_layer['stride']
            padding = conv_layer['padding']
            block = CNNBlock(in_channels, out_channels, kernel_size, stride, padding)
            layers.append(block)
        in_channels = out_channels
        return nn.Sequential(*layers)
    
    def _make_resnet_layers(self, res_block_configs):
        layers = []
        for block_config in res_block_configs:
            in_channels = block_config['in_channels']
            out_channels = block_config['out_channels']
            num_blocks = block_config['num_blocks']
            stride = block_config['stride']
            
            layers.append(self._make_layer(ResBlock, in_channels, out_channels, num_blocks, stride))
            
            # Update in_channels for the next set of blocks
            in_channels = out_channels * ResBlock.expansion
        return nn.Sequential(*layers), in_channels

    def _make_layer(self, block, in_channels, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)  # First block might have a stride to downsample
        blocks = []
        for stride in strides:
            blocks.append(block(in_channels, out_channels, stride))
            in_channels = out_channels * block.expansion  # Update in_channels for the next block
        return nn.Sequential(*blocks)

    def _make_fc_layers(self, prev_features, fc_layer_configs, num_classes):
        layers = nn.ModuleList()
        if fc_layer_configs is not None:
            for fc_layer in fc_layer_configs:
                layers.append(nn.Linear(prev_features, fc_layer['out_features']))
                prev_features = fc_layer['out_features']
            layers.append(nn.Linear(prev_features, num_classes))
        return layers

    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                x = F.relu(layer(x))
            else:
                x = layer(x)
        return x
    

class QuantizableCNN(CNN):
    def __init__(self, *args, **kwargs):
        super(QuantizableCNN, self).__init__(*args, **kwargs)
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
    
    def forward(self, x):
        x = self.quant(x)
        x = super().forward(x)
        x = self.dequant(x)
        return x


# ===== Define and set HyperParameters =====


## DataLoader
TRAIN_BATCH_SIZE = 64  # training batch size
VAL_BATCH_SIZE = 50  # validation batch size
NUM_WORKERS = 8  # number of workers for DataLoader

## Model
### CNN
conv_layer_configs = [
    {'in_channels': 3, 'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1},
    {'in_channels': 32, 'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1},
    {'in_channels': 64, 'out_channels': 128, 'kernel_size': 3, 'stride': 2, 'padding': 1}
]

fc_layer_configs = [
    {'out_features': 256},
    {'out_features': 128}
]

### ResNet
res_block_configs = [
    {'in_channels': 3, 'out_channels': 16, 'stride': 1, 'num_blocks': 3},
    {'in_channels': 16, 'out_channels': 32, 'stride': 2, 'num_blocks': 3},
    {'in_channels': 32, 'out_channels': 64, 'stride': 2, 'num_blocks': 3}
]

## Optimizer and scheduler
INITIAL_LR = 0.1  # initial learning rate
MOMENTUM = 0.9  # momentum for optimizer
REG = 1e-4  # L2 regularization strength
LR_PATIENCE = 5  # Patience for ReduceLROnPlateau scheduler
LR_FACTOR = 0.25  # Factor by which the learning rate will be reduced

## Training
EPOCHS = 3  # total number of training epochs
CHECKPOINT_FOLDER = "./saved_models"  # folder where models are saved

## Pruning and Quantization
ENABLE_QUANTIZATION = True

# ===== Set up preprocessing functions =====


## specify preprocessing function
mean = [0.4914, 0.4822, 0.4465]
std = [0.2023, 0.1994, 0.2010]
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])


# ===== Set up dataset and dataloader =====


DATA_ROOT = "./data"

## construct dataset
train_set = CIFAR10(
    root = DATA_ROOT, 
    train = True,
    download = True,
    transform = transform_train
)
val_set = CIFAR10(
    root = DATA_ROOT, 
    train = False, 
    download = True,
    transform = transform_val
)

## construct dataloader
train_loader = DataLoader(
    train_set, 
    batch_size= TRAIN_BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True
)
val_loader = DataLoader(
    val_set, 
    batch_size=VAL_BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True
)


# ===== Instantiate your SimpleNN model and deploy it to device =====


## specify the device for computation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if ENABLE_QUANTIZATION:
    net = QuantizableCNN(architecture_type='resnet', res_block_configs=res_block_configs, num_classes=10)
    net.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')  # Use 'fbgemm' for x86 or 'qnnpack' for ARM
    torch.quantization.prepare_qat(net, inplace=True)
else:
    # For CNN
    # net = CNN(architecture_type='cnn', conv_layer_configs=conv_layer_configs, fc_layer_configs=fc_layer_configs, num_classes=10)
    # For ResNet
    net = CNN(architecture_type='resnet', res_block_configs=res_block_configs, num_classes=10)

# deploy the network to device
net.to(device)

print(next(net.parameters()).device)

# ===== Set up the loss function and optimizer =====


## loss function
criterion = nn.CrossEntropyLoss() 

## Add optimizer
optimizer = optim.SGD(net.parameters(), lr=INITIAL_LR, momentum=MOMENTUM, weight_decay=REG)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=LR_PATIENCE, factor=LR_FACTOR)


# ===== Start the training process =====


best_val_acc = 0

print("==> Training starts!")
print("="*50)
for i in range(0, EPOCHS):    
    ## switch to train mode
    net.train()

    ## print the Epoch and learning rate
    current_learning_rate = optimizer.param_groups[0]['lr']
    print(f"Epoch {i}: with learning rate {current_learning_rate}")
    
    total_examples = 0
    correct_examples = 0
    train_loss = 0
    
    ## Train the model
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        ### copy inputs to device
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        ### compute the output and loss
        outputs = net(inputs)
        loss = criterion(outputs, targets.long())
        
        ### zero the gradient
        optimizer.zero_grad()
        
        ### backpropagation
        loss.backward()
        
        ### apply gradient and update the weights
        optimizer.step()
        
        ### count the number of correctly predicted samples in the current batch
        _, predicted = torch.max(outputs.data, 1)
        total_examples += targets.size(0)
        correct_examples += (predicted == targets).sum().item()
        train_loss += loss.item()*inputs.size(0)
                
    avg_loss = train_loss / len(train_loader)
    avg_acc = correct_examples / total_examples
    print("Training loss: %.4f, Training accuracy: %.4f" %(avg_loss, avg_acc))

    ## Validate on the validation dataset
    ## switch to eval mode
    net.eval()

    total_examples = 0
    correct_examples = 0
    val_loss = 0

    ## disable gradient during validation, which can save GPU memory
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            ### copy inputs to device
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            ### compute the output and loss
            outputs = net(inputs)
            loss = criterion(outputs, targets.long())
            
            ### count the number of correctly predicted samples in the current batch
            _, predicted = torch.max(outputs.data, 1)
            total_examples += targets.size(0)
            correct_examples += (predicted == targets).sum().item()
            val_loss += loss.item() * inputs.size(0)

    avg_loss = val_loss / len(val_loader)
    avg_acc = correct_examples / total_examples
    print("Validation loss: %.4f, Validation accuracy: %.4f" % (avg_loss, avg_acc))

    ## decay learning rate
    previous_learning_rate = current_learning_rate
    scheduler.step(val_loss)
    current_learning_rate = optimizer.param_groups[0]['lr']
    if previous_learning_rate != current_learning_rate:
        print(f"Learning rate decayed to {current_learning_rate}")
    
    ## save the model checkpoint
    if avg_acc > best_val_acc:
        best_val_acc = avg_acc
        if not os.path.exists(CHECKPOINT_FOLDER):
           os.makedirs(CHECKPOINT_FOLDER)
        print("Saving ...")
        state = {'state_dict': net.state_dict(),
                'epoch': i,
                'lr': current_learning_rate}
        saveName = 'CNN_quantizeTrained' if ENABLE_QUANTIZATION else 'CNN'
        torch.save(state, os.path.join(CHECKPOINT_FOLDER, saveName + '.pth'))
    print('')

print("="*50)
print(f"==> Optimization finished! Best validation accuracy: {best_val_acc:.4f}")