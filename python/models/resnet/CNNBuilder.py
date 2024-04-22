# ===== Import necessary libraries =====
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torch.quantization
from torch.quantization import QConfig

import torch.nn.utils.prune as prune




# ===== Set up the SimpleNN model =====

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, quantTrue=False):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
    def prune_weights(self, current_sparsity, target_sparsity):
        pruning_ratio = (target_sparsity - current_sparsity)/(1 - current_sparsity)
        prune.l1_unstructured(self.conv, 'weight', amount=pruning_ratio)


    def to_sparse(self):
        self.conv.weight = nn.Parameter(self.conv.weight.to_sparse())
    
class ResBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, quantTrue=False):
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
        
        self.quantadd = nn.quantized.FloatFunctional()

        self.quantTrue = quantTrue

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.quantTrue:
            if self.shortcut is not None:
                identity = self.shortcut(identity)
            out = self.quantadd.add(out, identity)
        else:
            if self.shortcut is not None:
                identity = self.shortcut(identity)
            out += identity

        out = F.relu(out)
        return out
    
    def prune_weights(self, current_sparsity, target_sparsity):
        pruning_ratio = (target_sparsity - current_sparsity)/(1 - current_sparsity)
        prune.l1_unstructured(self.conv1, 'weight', amount=pruning_ratio)
        prune.l1_unstructured(self.conv2, 'weight', amount=pruning_ratio)

    def to_sparse(self):
        self.conv1.weight = nn.Parameter(self.conv1.weight.to_sparse())
        self.conv2.weight = nn.Parameter(self.conv2.weight.to_sparse())

class CNN(nn.Module):
    def __init__(self, architecture_type='cnn', quantTrue = False, conv_layer_configs=None, fc_layer_configs=None, res_block_configs=None, num_classes=10):
        super(CNN, self).__init__()
        
        # Initialization code for other parts of the class remains the same
        self.myNetworkType = architecture_type
        if architecture_type == 'cnn':
            self.features = self._make_cnn_layers(conv_layer_configs)
            prev_features = conv_layer_configs[-1]['out_channels']
        elif architecture_type == 'resnet':
            self.init_conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
            self.init_bn = nn.BatchNorm2d(16)
            self.init_relu = nn.ReLU(inplace=True)
            self.features, prev_features = self._make_resnet_layers(res_block_configs, quantTrue)
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
    
    def _make_resnet_layers(self, res_block_configs, quantTrue=False):
        layers = []
        for block_config in res_block_configs:
            in_channels = block_config['in_channels']
            out_channels = block_config['out_channels']
            num_blocks = block_config['num_blocks']
            stride = block_config['stride']
            
            layers.append(self._make_layer(ResBlock, in_channels, out_channels, num_blocks, stride, quantTrue=quantTrue))
            
            # Update in_channels for the next set of blocks
            in_channels = out_channels * ResBlock.expansion
        return nn.Sequential(*layers), in_channels

    def _make_layer(self, block, in_channels, out_channels, num_blocks, stride, quantTrue = False):
        strides = [stride] + [1]*(num_blocks-1)  # First block might have a stride to downsample
        blocks = []
        for stride in strides:
            blocks.append(block(in_channels, out_channels, stride, quantTrue=quantTrue))
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
    

class QuantizablePrunableCNN(CNN):
    def __init__(self, *args, **kwargs):
        super(QuantizablePrunableCNN, self).__init__(*args, **kwargs)
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
    
    def forward(self, x):
        x = self.quant(x)
        x = super().forward(x)
        x = self.dequant(x)
        return x
    
    def prune_weights(self, current_sparsity, target_sparsity=0.5):
        for module in self.modules():
            if isinstance(module, (CNNBlock, ResBlock)):
                module.prune_weights(current_sparsity, target_sparsity)

    def convert_to_sparse(self):
        for module in self.modules():
            if isinstance(module, (CNNBlock, ResBlock)):
                module.to_sparse()

class CNNFactory:
    @staticmethod
    def makeModel(architecture_type, conv_layer_configs=None, fc_layer_configs=None, res_block_configs=None, num_classes=10, quantTrue=False):
        net = QuantizablePrunableCNN(architecture_type=architecture_type, quantTrue=quantTrue, conv_layer_configs=conv_layer_configs, fc_layer_configs=fc_layer_configs, res_block_configs=res_block_configs, num_classes=num_classes)
        if quantTrue:
            CNNFactory.prepQuant(net)
        return net
    
    @staticmethod
    def prepQuant(model):
        my_qconfig = QConfig(
        activation=torch.quantization.default_observer.with_args(dtype=torch.quint8),
        weight=torch.quantization.default_weight_observer.with_args(dtype=torch.qint8)
        )
        model.qconfig = my_qconfig
        torch.quantization.prepare_qat(model, inplace=True)
    
    @staticmethod
    def loadModel(model_path, reference_model=None, architecture_type=None, conv_layer_configs=None, fc_layer_configs=None, res_block_configs=None, num_classes=10, quantized=False, quantTrue=False):
        if reference_model is not None:
            model = reference_model
        else:
            if architecture_type is None:
                raise ValueError("architecture_type must be provided if no reference_model is given")
            model = CNNFactory.makeModel(architecture_type, conv_layer_configs, fc_layer_configs, res_block_configs, num_classes, quantTrue)
        loadedModel = torch.load(model_path)
        # if quantized:
        #     torch.quantization.convert(model.eval(), inplace=True)
        print(loadedModel)
        print(model)
        model.load_state_dict(loadedModel['state_dict'])
        return model

    
    @staticmethod
    def trainModel(net, EPOCHS = 5, ENABLE_QUANTIZATION = False, ENABLE_PRUNING = False, pruning_epochs = 3, starting_sparsity = 0.1, target_sparsity = 0.5, DATA_ROOT = "./data", Save_Name = "CNN.pth"):
        
        ## DataLoader
        TRAIN_BATCH_SIZE = 64  # training batch size
        VAL_BATCH_SIZE = 50  # validation batch size
        NUM_WORKERS = 8  # number of workers for DataLoader

        ## Optimizer and scheduler
        INITIAL_LR = 0.1  # initial learning rate
        MOMENTUM = 0.9  # momentum for optimizer
        REG = 1e-4  # L2 regularization strength
        LR_PATIENCE = 5  # Patience for ReduceLROnPlateau scheduler
        LR_FACTOR = 0.25  # Factor by which the learning rate will be reduced

        EARLY_STOP_PATIENCE = 2 * LR_PATIENCE  # Early stop patience is twice the LR patience
        EARLY_STOP_THRESHOLD = 1e-4  # Same as ReduceLROnPlateau threshold

        ## Training
        CHECKPOINT_FOLDER = "./saved_models"  # folder where models are saved

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

        # ===== Instantiate your model and deploy it to device =====


        ## specify the device for computation
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net.to(device)
        print(next(net.parameters()).device)

         # ===== Prepare for Training =====

        def make_pruning_schedule(pruning_epochs, total_epochs, start_sparsity, final_sparsity):
            sparsity_levels = np.linspace(start_sparsity, final_sparsity, num=pruning_epochs)
            for i in range(total_epochs - pruning_epochs):
                sparsity_levels = np.append(sparsity_levels, final_sparsity)
            return sparsity_levels

        # ===== Set up the loss function and optimizer =====

        ## loss function
        criterion = nn.CrossEntropyLoss() 

        ## Add optimizer
        optimizer = optim.SGD(net.parameters(), lr=INITIAL_LR, momentum=MOMENTUM, weight_decay=REG)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=LR_PATIENCE, factor=LR_FACTOR)

        pruning_schedule = make_pruning_schedule(pruning_epochs, EPOCHS, starting_sparsity, target_sparsity)

        # ===== Start the training process =====

       
        no_improve_epochs = 0
        early_stop_triggered = False

        best_val_loss = np.inf
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

            if ENABLE_PRUNING:
                current_sparsity = pruning_schedule[i-1] if i > 0 else 0
                target_sparsity = pruning_schedule[i]
                net.prune_weights(current_sparsity, target_sparsity)
                print(f"Pruning to {target_sparsity:.2%} sparsity level")

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

            if avg_loss + EARLY_STOP_THRESHOLD < best_val_loss:
                best_val_loss = avg_loss
                no_improve_epochs = 0  # Reset the improvement counter if there is an improvement
            else:
                no_improve_epochs += 1  # Increment if no improvement

            # Check if early stopping is triggered
            if no_improve_epochs > EARLY_STOP_PATIENCE:
                print(f"Early stopping triggered after {i+1} epochs due to no significant improvement.")
                early_stop_triggered = True
                print("Plateau detected! Stopping early")
                break

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
                saveName = Save_Name
                torch.save(state, os.path.join(CHECKPOINT_FOLDER, saveName + '.pth'))
            print('')

        print("="*50)
        print(f"==> Optimization finished! Best validation accuracy: {best_val_acc:.4f}")
        return net
    
    @staticmethod
    def check_sparsity(model):
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                weight = module.weight.data
                if weight.is_sparse:
                    weight = weight.to_dense()
                weight = weight.cpu()
                sparsity = float(torch.sum(weight == 0)) / float(weight.nelement())
                print(f"Sparsity in {name}: {sparsity * 100:.2f}%")

    @staticmethod
    def checkSize(model):
        saveName = './TempModel.pth'
        torch.save(model.state_dict(), saveName)
        # Check the size of the model
        model_size_bytes = os.path.getsize(saveName)
        model_size_mb = model_size_bytes / (1024 * 1024) 
        print(f"Model Size: {model_size_mb:.2f} MB")
        os.remove(saveName)
        return model_size_mb
    
    @staticmethod
    def quantizeModel(model):
        torch.quantization.convert(model.eval(), inplace=True)
        return model
    
    @staticmethod
    def evaluate_model(model, data_loader):
        model.eval()
        total = 0
        correct = 0
        loss = 0
        with torch.no_grad():
            for data in data_loader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                loss += criterion(outputs, labels).item() * images.size(0)
        
        avg_loss = loss / total
        accuracy = 100 * correct / total
        print(f'Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
        return accuracy
    
    @staticmethod
    def saveModel(model, path, save_name):
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(model.state_dict(), os.path.join(path, save_name + '.pth'))
        print(f"Model saved at {os.path.join(path, save_name + '.pth')}")
        return model