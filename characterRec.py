import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as pdata
import torchvision
from torchvision import transforms
from torchvision.io import read_image
import glob
import matplotlib.pyplot as plt

characters = ['0','1','2','3','4','5','6','7','8','9',
        'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',
        'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

dataset_path = './dataset/character_dataset'
train_ratio = 0.7
validation_ratio = 0.2
data_num = 109584

# hyperparameter
lr=0.001
batch_size=128
epoch=100

seed = 500
torch.random.manual_seed(seed)

class MyDataset(pdata.Dataset):
    def __init__(self, dataset_dir, transform=None, target_transform=None):
        self.dataset_dir = os.path.abspath(dataset_dir)
        self.image_list = [os.path.basename(path) for path in glob.glob(os.path.join(self.dataset_dir, "*.png"))]

        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        image_path = os.path.join(self.dataset_dir, image_name)
        img = read_image(image_path)
        image, _ = os.path.splitext(image_name)
        label = int(image.split("_")[-1])
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)
        return img, label

transform = transform = transforms.Compose([
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize((0.5,), (0.5,))])
target_transform = transforms.Lambda(lambda y : torch.zeros(len(characters), dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
dataset = MyDataset(dataset_path, transform, target_transform)

train_len = int(data_num*train_ratio)
val_len = int(data_num*validation_ratio)
traindata, validata, testdata = pdata.random_split(dataset, [train_len, val_len, len(dataset)-train_len-val_len])
train_loader = pdata.DataLoader(traindata, batch_size=batch_size, shuffle=True, num_workers=2)
vali_loader = pdata.DataLoader(validata, batch_size=batch_size)
test_loader = pdata.DataLoader(testdata, batch_size=batch_size)

class Model(nn.Module):
    def __init__(self, input, output):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input, out_channels=8, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)

        self.linear1 = nn.Linear(in_features=6*6*32, out_features=1024)
        self.relu4 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.85)
        self.linear2 = nn.Linear(in_features=1024, out_features=512)
        self.relu5 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.85)
        self.linear3 = nn.Linear(in_features=512, out_features=output)
        self.softmax = nn.Softmax(1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = torch.flatten(x, 1)
        x = self.linear1(x)
        x = self.relu4(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.relu5(x)
        x = self.dropout2(x)
        x = self.linear3(x)
        x = self.softmax(x)
        
        return x

device = "cuda" if torch.cuda.is_available() else "cpu"
net = Model(1, len(characters))

def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.0)

net.apply(init_weights)
net.to(device)

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=0.0001)

# start training
train_loss_history = []
vali_loss_history = []
train_acc_history = []
vali_acc_history = []

min_vali_loss = np.inf

for i in range(epoch):
    train_loss =0.0
    train_acc = 0.0
    net.train()
    step = 0
    for j, data in enumerate(train_loader):
        images, labels = data
        images = images.cuda()
        labels = labels.cuda()
        optimizer.zero_grad()
        predicted_output = net(images)
        fit = loss(predicted_output,labels)
        #fit.requires_grad_(True)
        fit.backward()
        optimizer.step()
        train_loss += fit.item()
        step += 1
        
        if step % 50 == 0:
            _, pred = predicted_output.max(1)
            num_correct = (pred==labels).sum().item()
            acc = num_correct / images.shape[0]
            print(f'epoch {i}, step {step}, train loss:{loss.fit.item()}, train accuray:{acc}')
            train_acc += acc
        train_loss = train_loss / step
        train_acc = acc / step
        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)
    
    vali_loss = 0.0
    vali_acc = 0.0
    net.eval()
    step = 0
    with torch.grad:
        for j, data in enumerate(vali_loader):
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            predicted_output = net(images)
            fit = loss(predicted_output,labels)
            vali_loss += fit.item()
            _, pred = predicted_output.max(1)
            num_correct = (pred==labels).sum().item()
            acc = num_correct / images.shape[0]
            vali_acc += acc
            step += 1
        vali_loss = vali_loss / step
        vali_acc = acc / step
        vali_loss_history.append(vali_loss)
        vali_acc_history.append(vali_acc)
    print(f'epoch {i}, step {step}, train loss:{train_loss}, train accuray:{train_acc}, validation loss:{vali_loss}, validation accuray:{vali_acc}')
    
    if min_vali_loss > vali_loss:
            min_vali_loss = vali_loss
            torch.save(net.state_dict(), "./saved_model.pth")
            
            
plt.plot(range(epoch),train_loss_history,'-',linewidth=3,label='Train error')
plt.plot(range(epoch),vali_loss_history,'-',linewidth=3,label='validation error')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid(True)
plt.legend()

plt.plot(range(epoch),train_acc_history,'-',linewidth=3,label='Train accuracy')
plt.plot(range(epoch),vali_acc_history,'-',linewidth=3,label='validation accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.grid(True)
plt.legend()


# test
model_path = "./model/saved_model.pth"
net.load_state_dict(model_path)
net.eval()
test_loss = 0.0
test_acc = 0.0
step = 0
for j, data in enumerate(test_loader):
    images, labels = data
    images = images.cuda()
    labels = labels.cuda()
    predicted_output = net(images)
    fit = loss(predicted_output,labels)
    test_loss += fit.item()
    _, pred = predicted_output.max(1)
    num_correct = (pred==labels).sum().item()
    acc = num_correct / images.shape[0]
    test_acc += acc
    step += 1

test_loss = test_loss / step
test_acc = test_acc / step
print(f'test loss:{test_loss}, test accuray:{test_acc}')
