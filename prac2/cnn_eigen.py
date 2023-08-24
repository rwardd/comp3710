# Prac 2 Question 2 Part 1
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn import preprocessing
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

#build random forest
# Download the data, if not already on disk and load it as numpy arrays
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
# introspect the images arrays to find the shapes (for plotting)
n_samples, h, w = lfw_people.images.shape
# for machine learning we use the 2 data directly (as relative pixel
# positions info is ignored by this model)
X = lfw_people.data
print(h, w)
n_features = X.shape[1]
# the label to predict is the id of the person
y = lfw_people.target
target_names = lfw_people.target_names
print(y)
n_classes = target_names.shape[0]
print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)
# Split into a training set and a test set using a stratified k fold
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)# Compute a PCA
#normalise
device = torch.device("cuda")
X_train = torch.from_numpy(X_train)
X_test = torch.from_numpy(X_test)
y_train = torch.from_numpy(y_train)
y_test = torch.from_numpy(y_test)
X_train = X_train.reshape(-1, 1, h, w)
X_test = X_test.reshape(-1, 1, h, w)
#print("X_train shape:", X_train.shape)

y_train.to(device)
y_test.to(device)
X_train.to(device)
X_test.to(device)
# Create training net
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3))
        self.fc1 = nn.Linear(4928, 7)

    def forward(self, x):
        x = torch.relu(F.max_pool2d(self.conv1(x),2))
        x = torch.relu(F.max_pool2d(self.conv2(x),2))
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        return x


net = Net()
net.cuda()
print(net)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001)

net.train()
for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(X_train, 0):
        labels = y_train.cuda()
        outputs = net(X_train.cuda())
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 1:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

net.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for i, data in enumerate(X_test, 0):
        labels = y_test.cuda()
        outputs = net(X_test.cuda())
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print("Test Accuracy: {} %".format(100 * correct / total))






