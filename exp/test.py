import torch
import torch.nn
import torch.optim
import torch.profiler
import torch.utils.data
import torchvision.datasets
import torchvision.models
import torchvision.transforms as T

mean = [0.4914, 0.4822, 0.4465]
std = [0.2023, 0.1994, 0.2010]
transform = T.Compose(
    [T.Resize(224),
     T.ToTensor(),
     T.Normalize(mean, std)])
train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)

device = torch.device("cuda:0")
# get a model for cifar10
# model = torchvision.models.resnet50(weights='IMAGENET1K_V1').cuda(device)
model = torch.jit.load('/home/cw541/661/project/resnet18.pt').cuda(device)
criterion = torch.nn.CrossEntropyLoss().cuda(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


def train(data):
    inputs, labels = data[0].to(device=device), data[1].to(device=device)
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def inference(data):
    inputs, labels = data[0].to(device=device), data[1].to(device=device)
    outputs = model(inputs)
    print(outputs.shape)
    _, preds = torch.max(outputs, 1)
    acc = torch.sum(preds == labels).item() / labels.size(0)
    loss = criterion(outputs, labels)
    return acc

if __name__ == '__main__':



    model.eval()

    with torch.no_grad():
        for step, batch_data in enumerate(test_loader):
            # print(step, batch_data[0].shape, batch_data[1].shape)
            if step > 0:
                break
            acc = inference(batch_data)
            print(acc)

    # scripted_model = torch.jit.script(model)
    # scripted_model.save("resnet50.pt")


