from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union
from pydantic import BaseModel, ValidationError, validate_call

def enforce_types(func: Callable):
    return validate_call(config=dict(arbitrary_types_allowed=True, validate_return=True))(func)

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torch.quantization
from torch.quantization import QConfig, default_observer, default_per_channel_weight_observer

import torch.nn.utils.prune as prune

import numpy as np
import torchvision.transforms as T

from tqdm import tqdm

def bake_in_pruning(model: nn.Module) -> None:
    raise ValueError("Currently deprecated")
    for module in model.modules():
        # Check if the module has a 'weight' attribute that has been pruned
        if hasattr(module, 'weight') and prune.is_pruned(module):
            # Remove the pruning reparameterization and make it permanent
            prune.remove(module, 'weight')

def prune_model_weights(model: nn.Module, current_ratio: float, target_ratio: float, global_pruning: bool) -> None:
    raise ValueError("Currently deprecated")
    additional_pruning = (target_ratio - current_ratio) / (1 - current_ratio)
    parameters_to_prune = []
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and not isinstance(module, torch.nn.BatchNorm2d):
            parameters_to_prune.append((module, 'weight'))

    if global_pruning:
        # Global pruning
        all_tensors = torch.cat([p.flatten() for _, p in parameters_to_prune])
        threshold = torch.quantile(all_tensors.abs(), additional_pruning)
        for module, _ in parameters_to_prune:
            prune.custom_from_mask(module, name="weight", mask=module.weight.abs() > threshold)
    else:
        # Local pruning for each layer
        for module, name in parameters_to_prune:
            prune.l1_unstructured(module, name=name, amount=additional_pruning)

def check_sparsity(model: nn.Module) -> None:
    total_elements = 0
    total_zero_elements = 0
    raise ValueError("Currently deprecated")

    for module in model.modules():
        if hasattr(module, 'weight') and module.weight is not None:
            weight = module.weight.data
            total_elements += weight.numel()
            total_zero_elements += torch.sum(weight == 0).item()

    sparsity = (total_zero_elements / total_elements) * 100
    print(f"Total weight sparsity in the model: {sparsity:.2f}%")

def train_resnet_model(
    model: nn.Module, 
    resnet_block_size: int,
    prune_ratio: float,
    prune_epochs: int,
    total_epochs: int,
    prune_global: bool,
    quantize_dtype: Any
) -> None:
    # Load CIFAR10 data

    device = torch.device("cuda")
    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        T.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Define the transformations for the test data (without data augmentation)
    test_transform = transforms.Compose([
        T.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Load the CIFAR10 training data with the train_transform
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=8)

    # Load the CIFAR10 test data with the test_transform
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=8)

    criterion = nn.CrossEntropyLoss()
    accuracies = []

    # # Phase 1: Train up to prune_epochs
    # for epoch in range(prune_epochs):
    #     print(f"{epoch=}")
    #     model.train()
    #     running_loss = 0.0
    #     for i, (inputs, labels) in enumerate(tqdm(trainloader, desc=f"Epoch {epoch+1}/{total_epochs}", leave=False)):
    
    #         inputs, labels = inputs.to(device), labels.to(device)

    #         optimizer.zero_grad()
    #         outputs = model(inputs)
    #         loss = criterion(outputs, labels)
    #         loss.backward()
    #         optimizer.step()
    #         running_loss += loss.item()

    #     # Validate the model
    #     model.eval()
    #     correct = 0
    #     total = 0
    #     with torch.no_grad():
    #         for data in testloader:
    #             images, labels = data
    #             images, labels = images.cuda(), labels.cuda()
    #             outputs = model(images)
    #             _, predicted = torch.max(outputs.data, 1)
    #             total += labels.size(0)
    #             correct += (predicted == labels).sum().item()
        
    #     accuracy = 100 * correct / total
    #     accuracies.append(accuracy)
    #     print(f'Epoch {epoch+1}/{prune_epochs} - Loss: {running_loss / len(trainloader):.3f}, Accuracy: {accuracy:.2f}%')

    #     # prune the model
    #     current_ratio = epoch / prune_epochs * prune_ratio
    #     target_ratio = (epoch + 1) / prune_epochs * prune_ratio
    #     prune_model_weights(model, current_ratio, target_ratio, prune_global)
    #     check_sparsity(model)

    save_name = get_resnet_filename(resnet_block_size, quantize_dtype, "weights", prune_ratio, prune_global)
    checkpoint_name = "tmp_" + save_name
    best_accuracy = 0.0
    stagnant_epochs = 0
    max_stagnant_epochs = 5  # Number of epochs to allow without improvement before stopping

    # Phase 2: Continue training until total_epochs
    for epoch in range(total_epochs):
        print(f"{epoch=}")
        model.train()
        
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(tqdm(trainloader, desc=f"Epoch {epoch+1}/{total_epochs}", leave=False)):
            inputs, labels = inputs.to(device, dtype=quantize_dtype), labels.to(device)

            optimizer.zero_grad()
            outputs = None
            loss = None
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validate the model
        model.eval()
        correct = 0
        total = 0
        valid_loss = 0.0
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device, dtype=quantize_dtype), labels.to(device)
                outputs = None
                loss = None

                outputs = model(images)
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                valid_loss += loss.item()

        accuracy = 100 * correct / total
        accuracies.append(accuracy)
        print(f'Epoch {epoch+1}/{total_epochs} - Loss: {running_loss / len(trainloader):.3f}, Accuracy: {accuracy:.2f}%')

        scheduler.step()

        # Checkpoint the model if it has the best validation accuracy so far
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), checkpoint_name)
            print(f"Saved new best model with accuracy: {accuracy:.2f}%")
            stagnant_epochs = 0  # Reset stagnant epoch count on improvement

    # After training, load the best model
    model.load_state_dict(torch.load(checkpoint_name))

    # bake_in_pruning(model)

    # check_sparsity(model)

    torch.save(model.state_dict(), save_name)

def get_resnet_filename(
    resnet_block_size: int,
    quantize_dtype: Any,
    filetype: str, 
    pruning_ratio: Optional[float] = None,
    pruning_global: Optional[bool] = None,
) -> str:
    """
    Generates a filename for a ResNet model based on the configuration parameters.
    
    Args:
        resnet_block_size (int): The block size of the ResNet model (e.g., 18, 34, 50).
        quantize_bits (Optional[int]): The number of bits used for quantization, if any.
        pruning_ratio (Optional[float]): The ratio of pruning applied to the model.
        pruning_global (Optional[bool]): Whether the pruning was applied globally.
    
    Returns:
        str: A string representing the filename for the model.
    """
    filename = f"resnet{resnet_block_size}"
    
    if quantize_dtype not in [torch.float64, torch.float32, torch.float16, torch.bfloat16]:
        raise NotImplementedError("Only supports float64, 32, 16 and bfloat16")
    
    if pruning_ratio is not None:
        raise ValueError("Pruning is currently deprecated")
        # Ensuring the pruning ratio is represented in a consistent format in the filename.
        pruning_type = 'global' if pruning_global else 'local'
        filename += f"_pruned{pruning_ratio:.2f}_{pruning_type}"
    
    if filetype not in ["torchscript", "weights"]:
        raise ValueError("Unsupported file type")

    filename += f"_{filetype}_{quantize_dtype}.pth"
    return filename

def save_resnet_model(
    model: nn.Module,
    resnet_block_size: int,
    quantize_dtype: Any, 
    filetype: str,
    pruning_ratio: Optional[float] = None,
    pruning_global: Optional[bool] = None,
    save_path: str = "./saved_models"
):
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    filename = get_resnet_filename(resnet_block_size, quantize_dtype, filetype, pruning_ratio, pruning_global)
    file_path = os.path.join(save_path, filename)

    model.to(torch.device("cuda"))
    if filetype == "weights":
        torch.save(model.state_dict(), file_path)
    elif filetype == "torchscript":
        model.eval()
        # Use a dummy input that matches the CIFAR-10 input dimensions
        dummy_input = torch.randn(64, 3, 32, 32, dtype=quantize_dtype).to(torch.device("cuda"))
        traced_model = torch.jit.trace(model, dummy_input)
        torch.jit.save(traced_model, file_path)
    else:
        raise ValueError("Unsupported save type")

def load_resnet_model(
    resnet_block_size: int,
    quantize_dtype: Any,
    filetype: str,
    pruning_ratio: Optional[float] = None,
    pruning_global: Optional[bool] = None,
    save_path: str = "./saved_models"
) -> nn.Module:
    filename = get_resnet_filename(resnet_block_size, quantize_dtype, filetype, pruning_ratio, pruning_global)
    file_path = os.path.join(save_path, filename)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No saved model found at {file_path}")
    
    model = None
    if filetype == "weights":
        model = make_untrained_resnet_model(resnet_block_size, quantize_dtype)
        model.load_state_dict(torch.load(file_path))
    elif filetype == "torchscript":
        model = torch.jit.load(file_path)
    else:
        raise ValueError("Invalid filetype")

    return model

def make_untrained_resnet_model(
    resnet_block_size: int,
    quantize_dtype: Any
) -> nn.Module:
    
    model_options = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152
    }
    
    model_constructor = model_options.get(resnet_block_size)
    if model_constructor is None:
        raise ValueError(f"Unsupported ResNet block size: {resnet_block_size}")

    model = model_constructor(pretrained=False)

    model.apply(lambda m: m.to(dtype=quantize_dtype))
    return model

def model_size_in_bytes(model: nn.Module, include_gradients: bool=False) -> int:
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()  # num_elements * size_of_each_element
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    total_size = param_size + buffer_size

    if include_gradients:
        total_size *= 2  # Assuming gradients require the same amount of memory as parameters

    return total_size