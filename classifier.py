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
        feature_map = feature_map.squeeze()
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
BATHC_SIZE=64
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
