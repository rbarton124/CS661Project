import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet10(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet10, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.linear = nn.Linear(64*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def resnet10():
    return ResNet10(BasicBlock, [1], num_classes=10)  # Using only 1 block for simplicity

if __name__ == "__main__":
    model = resnet10()
    model.eval()

    example_input = torch.rand(1, 3, 32, 32)  # CIFAR-10 images are 32x32 pixels with 3 channels

    with torch.no_grad():
        example_input = torch.rand(1, 3, 32, 32)

        example_output = model(example_input)
        
        print("Output shape:", example_output.shape)

        expected_shape = torch.Size([1, 10])
        if example_output.shape == expected_shape:
            print("Success: The output shape matches the expected shape:", expected_shape)
        else:
            print("Failure: The output shape does not match the expected shape. Got:", example_output.shape, "Expected:", expected_shape)


    # Use scripting to export the model (you could also use tracing)
    scripted_model = torch.jit.script(model)

    # Save the scripted model to a file
    scripted_model.save("dummy_cifar10_network.pt")