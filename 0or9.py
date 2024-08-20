import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
import numpy as np


def relu(x):
    return np.maximum(0, x)

class Perceptron:
    def __init__(self, lr = 1e-2, n_iter = 10000):
        self.lr = lr
        self.n_iter = n_iter
        self.activation = relu
        self.w = None
        self.b = 0
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        for loop in range(self.n_iter):
            for i, x_i in enumerate(X):
                output = np.dot(x_i, self.w) + self.b
                y_predicted = self.activation(output)

                error = y[i] - y_predicted
                
                self.w += self.lr * error * x_i
                self.b += self.lr * error


    def predict(self, X):
        output = np.dot(X, self.w) + self.b
        y_predicted = self.activation(output)
        return y_predicted



transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1)) 
])

mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

def filter_09(dataset):
    data = []
    targets = []
    for img, label in dataset:
        if label == 0:  # 0
            data.append(img.numpy())
            targets.append(0)  
        elif label == 9:  # 9
            data.append(img.numpy())
            targets.append(1)  
    return np.array(data), np.array(targets)

X_train, y_train = filter_09(mnist_train)

print(f"Shape training : {X_train.shape}")
print(f"Shape label : {y_train.shape}")

# First test to classify 0 and 9
perceptron = Perceptron(lr=0.01, n_iter=1000)

perceptron.fit(X_train, y_train)

predictions = perceptron.predict(X_train)
print("Predictions :", predictions)

accuracy = np.mean(predictions == y_train)
print(f"Training accuracy : {accuracy * 100:.2f}%")


#Of course, the result is bad, the data points are clearly not separable, in the main branch, we perform a more general approach on the complete dataset

