```python

```


```python
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

C1 = 'adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib'
C2 = 'large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa'
C3 = 'normal'
C4 = 'squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa'
class_list = [C1, C2, C3, C4]
class2idx = {C1:0, C2:1, C3:2, C4:3}

class_list_test = ['adenocarcinoma', 'large.cell.carcinoma', 'normal', 'squamous.cell.carcinoma']
class2idx_test = {'adenocarcinoma':0, 'large.cell.carcinoma':1, 'normal':2, 'squamous.cell.carcinoma':3}
class CustomDataset(Dataset):
    def __init__(self, split:str, transform=None):
        self.split = split
        self.transform = transform
        self.path = os.path.join('Data', split)
        self.data = self.getdata(self.path)
    
    def getdata(self, path):
        data = []
        if self.split != 'test':
            for c in class_list:
                splited_path = os.path.join(path, c)
                filenames = os.listdir(splited_path)
                for filename in filenames:
                    filepath = os.path.join(splited_path, filename)
                    image = Image.open(filepath).convert('L')
                    image = image.resize((224, 224))
                    # image = np.array(image)
                    if self.transform:
                        image = self.transform(image)

                    label = class2idx[c]
                    data.append((image, label))
        if self.split == 'test':
            for c in class_list_test:
                splited_path = os.path.join(path, c)
                filenames = os.listdir(splited_path)
                for filename in filenames:
                    filepath = os.path.join(splited_path, filename)
                    image = Image.open(filepath).convert('L')
                    image = image.resize((224, 224))
                    # image = np.array(image)
                    if self.transform:
                        image = self.transform(image)

                    label = class2idx_test[c]
                    data.append((image, label))
        return data            
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])
BATHC_SIZE=64
trainset = CustomDataset(split='train', transform=transform)
trainloader = DataLoader(trainset, batch_size=BATHC_SIZE, shuffle=True)

validset = CustomDataset(split='valid', transform=transform)
validloader = DataLoader(validset, batch_size=BATHC_SIZE, shuffle=False)

testset = CustomDataset(split='test', transform=transform)
testloader = DataLoader(testset, batch_size=BATHC_SIZE, shuffle=False)



class ResnetPretrained(nn.Module):
    def __init__(self, ):
        super(ResnetPretrained, self).__init__()
        resnet = models.resnet50(pretrained=True)
        num_features = resnet.fc.in_features
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.fc = nn.Linear(num_features, len(class_list))
    
    def forward(self, x):
        feature_map = self.resnet(x)
        feature_map = feature_map.squeeze(-1).squeeze(-1)
        output = self.fc(feature_map)
        return feature_map, output

device = torch.device('cuda')
model = ResnetPretrained().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=1e-3)
for epoch in range(50):
    loss = 0.0
    model.train()
    for i, data in enumerate(trainloader):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        feature_map, outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        loss += loss.item()
    model.eval()
    acc = 0
    cnt = 0
    for i, data in enumerate(validloader):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        feature_map, outputs = model(inputs)
        outputs = outputs.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        for i in range(labels.shape[0]):
            if np.argmax(outputs[i]) == labels[i]:
                acc+=1
            cnt+=1
    print('Epoch: {}, trainingloss:{}, valid acc: {}'.format(epoch, loss, acc/cnt))

acc = 0
cnt = 0
for i, data in enumerate(testloader):
    inputs, labels = data
    inputs = inputs.to(device)
    labels = labels.to(device)

    feature_map, outputs = model(inputs)
    outputs = outputs.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    for i in range(labels.shape[0]):
        if np.argmax(outputs[i]) == labels[i]:
            acc+=1
        cnt+=1

print(acc/cnt)


torch.save(model.state_dict(), 'resnet_model.pth')

```

    Epoch: 0, trainingloss:2.733670711517334, valid acc: 0.4027777777777778
    Epoch: 1, trainingloss:2.4588937759399414, valid acc: 0.3888888888888889
    Epoch: 2, trainingloss:2.461911916732788, valid acc: 0.375
    Epoch: 3, trainingloss:2.376035213470459, valid acc: 0.375
    Epoch: 4, trainingloss:2.140150785446167, valid acc: 0.4027777777777778
    Epoch: 5, trainingloss:2.18689227104187, valid acc: 0.4305555555555556
    Epoch: 6, trainingloss:2.0134503841400146, valid acc: 0.5138888888888888
    Epoch: 7, trainingloss:2.0806262493133545, valid acc: 0.5555555555555556
    Epoch: 8, trainingloss:1.987488865852356, valid acc: 0.5555555555555556
    Epoch: 9, trainingloss:1.9394519329071045, valid acc: 0.5694444444444444
    Epoch: 10, trainingloss:1.6935786008834839, valid acc: 0.5833333333333334
    Epoch: 11, trainingloss:1.8041878938674927, valid acc: 0.5694444444444444
    Epoch: 12, trainingloss:1.7261731624603271, valid acc: 0.5833333333333334
    Epoch: 13, trainingloss:1.585608959197998, valid acc: 0.5833333333333334
    Epoch: 14, trainingloss:1.6966722011566162, valid acc: 0.5694444444444444
    Epoch: 15, trainingloss:1.6590579748153687, valid acc: 0.5833333333333334
    Epoch: 16, trainingloss:1.459916591644287, valid acc: 0.5833333333333334
    Epoch: 17, trainingloss:1.4166159629821777, valid acc: 0.5833333333333334
    Epoch: 18, trainingloss:1.372911810874939, valid acc: 0.5833333333333334
    Epoch: 19, trainingloss:1.4947402477264404, valid acc: 0.5972222222222222
    Epoch: 20, trainingloss:1.2820326089859009, valid acc: 0.5833333333333334
    Epoch: 21, trainingloss:1.2950630187988281, valid acc: 0.5972222222222222
    Epoch: 22, trainingloss:1.150707721710205, valid acc: 0.6666666666666666
    Epoch: 23, trainingloss:1.06769597530365, valid acc: 0.6388888888888888
    Epoch: 24, trainingloss:1.2190148830413818, valid acc: 0.6666666666666666
    Epoch: 25, trainingloss:1.011473298072815, valid acc: 0.6944444444444444
    Epoch: 26, trainingloss:0.9482385516166687, valid acc: 0.6805555555555556
    Epoch: 27, trainingloss:1.06576669216156, valid acc: 0.6527777777777778
    Epoch: 28, trainingloss:0.9234473705291748, valid acc: 0.6944444444444444
    Epoch: 29, trainingloss:0.9255753755569458, valid acc: 0.6944444444444444
    Epoch: 30, trainingloss:1.1280956268310547, valid acc: 0.7222222222222222
    Epoch: 31, trainingloss:0.8170953989028931, valid acc: 0.6944444444444444
    Epoch: 32, trainingloss:1.028016448020935, valid acc: 0.7222222222222222
    Epoch: 33, trainingloss:1.1151254177093506, valid acc: 0.7361111111111112
    Epoch: 34, trainingloss:0.8381235599517822, valid acc: 0.7361111111111112
    Epoch: 35, trainingloss:0.9484105706214905, valid acc: 0.7361111111111112
    Epoch: 36, trainingloss:0.7734196186065674, valid acc: 0.75
    Epoch: 37, trainingloss:0.8496738076210022, valid acc: 0.7083333333333334
    Epoch: 38, trainingloss:0.7556465268135071, valid acc: 0.75
    Epoch: 39, trainingloss:0.6622350215911865, valid acc: 0.7222222222222222
    Epoch: 40, trainingloss:0.5654382705688477, valid acc: 0.7638888888888888
    Epoch: 41, trainingloss:0.6285737156867981, valid acc: 0.7777777777777778
    Epoch: 42, trainingloss:0.5178877711296082, valid acc: 0.7638888888888888
    Epoch: 43, trainingloss:0.6529508829116821, valid acc: 0.7361111111111112
    Epoch: 44, trainingloss:0.8560333251953125, valid acc: 0.7638888888888888
    Epoch: 45, trainingloss:0.5916526317596436, valid acc: 0.7777777777777778
    Epoch: 46, trainingloss:0.5384730100631714, valid acc: 0.8055555555555556
    Epoch: 47, trainingloss:0.4458844065666199, valid acc: 0.7916666666666666
    Epoch: 48, trainingloss:0.5200281143188477, valid acc: 0.8055555555555556
    Epoch: 49, trainingloss:0.5302483439445496, valid acc: 0.8055555555555556
    0.7777777777777778



```python
import numpy as np
import pandas
import sklearn
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

class ResnetPretrained(nn.Module):
    def __init__(self, ):
        super(ResnetPretrained, self).__init__()
        resnet = models.resnet50(pretrained=True)
        num_features = resnet.fc.in_features
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.fc = nn.Linear(num_features, len(class_list))
    
    def forward(self, x):
        feature_map = self.resnet(x)
        feature_map = feature_map.squeeze(-1).squeeze(-1)
        output = self.fc(feature_map)
        return feature_map, output

### Neural Network
C1 = 'adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib'
C2 = 'large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa'
C3 = 'normal'
C4 = 'squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa'
class_list = [C1, C2, C3, C4]
class2idx = {C1:0, C2:1, C3:2, C4:3}

class_list_test = ['adenocarcinoma', 'large.cell.carcinoma', 'normal', 'squamous.cell.carcinoma']
class2idx_test = {'adenocarcinoma':0, 'large.cell.carcinoma':1, 'normal':2, 'squamous.cell.carcinoma':3}
class CustomDataset(Dataset):
    def __init__(self, split:str, transform=None):
        self.split = split
        self.transform = transform
        self.path = os.path.join('Data', split)
        self.data = self.getdata(self.path)
    
    def getdata(self, path):
        data = []
        if self.split != 'test':
            for c in class_list:
                splited_path = os.path.join(path, c)
                filenames = os.listdir(splited_path)
                for filename in filenames:
                    filepath = os.path.join(splited_path, filename)
                    image = Image.open(filepath).convert('L')
                    image = image.resize((224, 224))
                    # image = np.array(image)
                    if self.transform:
                        image = self.transform(image)

                    label = class2idx[c]
                    data.append((image, label))
        if self.split == 'test':
            for c in class_list_test:
                splited_path = os.path.join(path, c)
                filenames = os.listdir(splited_path)
                for filename in filenames:
                    filepath = os.path.join(splited_path, filename)
                    image = Image.open(filepath).convert('L')
                    image = image.resize((224, 224))
                    # image = np.array(image)
                    if self.transform:
                        image = self.transform(image)

                    label = class2idx_test[c]
                    data.append((image, label))
        return data            
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])
BATHC_SIZE=4
trainset = CustomDataset(split='train', transform=transform)
trainloader = DataLoader(trainset, batch_size=BATHC_SIZE, shuffle=True)

validset = CustomDataset(split='valid', transform=transform)
validloader = DataLoader(validset, batch_size=BATHC_SIZE, shuffle=False)

testset = CustomDataset(split='test', transform=transform)
testloader = DataLoader(testset, batch_size=BATHC_SIZE, shuffle=False)

device = torch.device('cuda')
model = ResnetPretrained()
model.load_state_dict(torch.load('resnet_model.pth'))
model.to(device)
model.eval()
### Generate feature map
acc = 0
cnt = 0
train_featuremaps = []
train_labels = []
valid_featuremaps = []
valid_labels = []
for i, data in enumerate(trainloader):
    inputs, labels = data
    inputs = inputs.to(device)
    labels = labels.to(device)
    feature_map, outputs = model(inputs)
    feature_map = feature_map.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    train_labels += list(labels)
    for i in range(len(labels)):
        train_featuremaps.append(feature_map[i])

for i, data in enumerate(validloader):
    inputs, labels = data
    inputs = inputs.to(device)
    labels = labels.to(device)
    feature_map, outputs = model(inputs)
    feature_map = feature_map.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    
    valid_labels += list(labels)
    for i in range(len(labels)):
        valid_featuremaps.append(feature_map[i])

### Use neural network to test
acc = 0
cnt = 0
prediction_neuralnetwork = []
test_labels = []
test_feature_map = []
for i, data in enumerate(testloader):
    inputs, labels = data
    inputs = inputs.to(device)
    labels = labels.to(device)
    feature_map, outputs = model(inputs)
    outputs = outputs.detach().cpu().numpy()
    feature_map = feature_map.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    test_labels += list(labels)
    for i in range(labels.shape[0]):
        test_feature_map.append(feature_map[i])
        prediction_neuralnetwork.append(np.argmax(outputs[i]))
        if np.argmax(outputs[i]) == labels[i]:
            acc+=1
        cnt+=1

print("Accuracy of Neural Network: {}".format(acc/cnt))

# SVM
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
X_train = np.array(train_featuremaps)
y_train = np.array(train_labels)
X_valid = np.array(valid_featuremaps)
y_valid = np.array(valid_labels)
X_train = np.vstack((X_train, X_valid))
y_train = np.hstack((y_train, y_valid))
X_test = np.array(test_feature_map)
y_test = np.array(test_labels)
svm_classifier = SVC()

svm_classifier.fit(X_train, y_train)

prediction_svm = svm_classifier.predict(X_test)

accuracy = accuracy_score(y_test, prediction_svm)
print("Accuracy of SVM: {}".format(accuracy))


### Xgboost
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
xgb_classifier = XGBClassifier()

xgb_classifier.fit(X_train, y_train)

prediction_xgboost = xgb_classifier.predict(X_test)

accuracy = accuracy_score(y_test, prediction_xgboost)
print("Accuracy of Xgboost:{}".format(accuracy))

### KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
k = 3  
knn_classifier = KNeighborsClassifier(n_neighbors=k)

knn_classifier.fit(X_train, y_train)

prediction_knn = knn_classifier.predict(X_test)
accuracy = accuracy_score(y_test, prediction_knn)
print("Accuracy of kNN:{}".format(accuracy))


### NB
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
gnb_classifier = GaussianNB()

gnb_classifier.fit(X_train, y_train)

prediction_nb = gnb_classifier.predict(X_test)

accuracy = accuracy_score(y_test, prediction_nb)
print("Accuracy of Naive Bayes:{}".format(accuracy))


## ensemble
acc = 0
cnt = 0
print(len(test_labels))
for i in range(len(test_labels)):
    voted = [0, 0, 0, 0]
    voted[prediction_neuralnetwork[i]] += 1
    voted[prediction_svm[i]] += 1
    voted[prediction_xgboost[i]] += 1
    voted[prediction_knn[i]] += 1
    voted[prediction_nb[i]] += 1
    result = np.argmax(voted)
    if result == test_labels[i]:
        acc += 1
    cnt += 1
print('Accuracy of Ensemble: {}'.format(acc/cnt))
```

    Accuracy of Neural Network: 0.7777777777777778
    Accuracy of SVM: 0.8380952380952381
    Accuracy of Xgboost:0.7428571428571429
    Accuracy of kNN:0.8539682539682539
    Accuracy of Naive Bayes:0.7555555555555555
    315
    Accuracy of Ensemble: 0.8222222222222222
