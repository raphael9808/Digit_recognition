import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

def relu(x):
    return np.maximum(0, x)

class MNIST(nn.Module):
    def __init__(self, lr = 1e-2, n_iter = 1000):
        super(MNIST, self).__init__()
        self.lr = lr
        self.n_iter = n_iter
        self.relu = nn.ReLU()
        self.logsoftmax = nn.LogSoftmax(dim=1)
        
        self.l1 = nn.Linear(28*28, 64) 
        self.l2 = nn.Linear(64, 64) 
        self.l3 = nn.Linear(64, 10)    
    
    def forward(self, X):
        X = X.view(-1, 28*28)
        X = self.relu(self.l1(X))
        X = self.relu(self.l2(X))
        X = self.logsoftmax(self.l3(X)) 
        return X
    
    def fit(self, train_loader, criterion, optimizer, epochs=5):
        for epoch in range(epochs):
            running_loss = 0.0
            for images, labels in train_loader:
                optimizer.zero_grad()
                outputs = self.forward(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            
            print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")
        
        print("Training Done")

    def predict(self, X):
        with torch.no_grad():
            X = X.view(-1, 28*28)
            outputs = self.forward(x)
            _, predicted = torch.max(outputs, 1)
        return predicted

    def test(self, test_loader, criterion):
        correct = 0
        total = 0
        running_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = self.forward(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f"Test Loss: {running_loss/len(test_loader)}")
        print(f"Test Accuracy: {accuracy}%")
        return accuracy
    



transform = transforms.Compose(
                [transforms.ToTensor(), 
                 transforms.Normalize((0.5,), (0.5,))]
            )
    
train_data = datasets.MNIST(root='mnist_data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    
test_data = datasets.MNIST(root='mnist_data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
    
model = MNIST()

criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), model.lr)
    
model.fit(train_loader, criterion, optimizer, epochs=5)
model.test(test_loader, criterion)
