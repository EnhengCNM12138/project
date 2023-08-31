import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import os

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
        feature_map = feature_map.squeeze()
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
