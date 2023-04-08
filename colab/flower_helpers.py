from collections import OrderedDict
from flwr.server.strategy import FedAvg
import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_data(train=True):
    """Load CIFAR-10 (training and test set)."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    dataset = CIFAR10("./dataset", train=train, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    return dataloader

class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
def get_weights(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]
    
def set_weights(model, weights) -> None:
    params_dict = zip(model.state_dict().keys(), weights)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)
    
    
def train(epochs, parameters, return_dict):
    """Train the network on the training set."""
    # Create model
    net = Net().to(DEVICE)
    # Load weights
    if parameters is not None:
        set_weights(net, parameters)
    # Load data (CIFAR-10)
    trainloader = load_data(train=True)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
    # Prepare return values
    return_dict["parameters"] = get_weights(net)
    return_dict["data_size"] = len(trainloader)


def test(parameters, return_dict):
    """Validate the network on the entire test set."""
    # Create model
    net = Net().to(DEVICE)
    # Load weights
    if parameters is not None:
        set_weights(net, parameters)
    # Load data (CIFAR-10)
    testloader = load_data(train=False)
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    # Prepare return values
    return_dict["loss"] = loss
    return_dict["accuracy"] = accuracy
    return_dict["data_size"] = len(testloader)
    
class FedAvgMp(FedAvg):
    """This class implements the FedAvg strategy for Multiprocessing context."""

    def configure_evaluate(self, server_round: int, 
                           parameters: fl.common.Parameters, 
                           client_manager: fl.server.client_manager.ClientManager):
        """Configure the next round of evaluation. Returns None since evaluation is made server side.
        You could comment this method if you want to keep the same behaviour as FedAvg."""
        return None