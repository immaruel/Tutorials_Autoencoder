## 오토인코더로 이미지 특징 추출하기

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import os
import numpy as np
import time
import copy 
from PIL import Image
from matplotlib import cm

hyper_param_epoch = 20
hyper_param_batch = 4

# dataloader 설정
class CustomImageDataset(Dataset):
    def read_data_set(self):

        all_img_files = []
        all_labels = []

        class_names = os.walk(self.data_set_path).__next__()[1] 

        for index, class_name in enumerate(class_names):
            label = index
            img_dir = os.path.join(self.data_set_path, class_name)
            img_files = os.walk(img_dir).__next__()[2]

            for img_file in img_files:
                img_file = os.path.join(img_dir, img_file)
                img = Image.open(img_file)
                if img is not None:
                    all_img_files.append(img_file)
                    all_labels.append(label) # label은 0 또는 1로 저장됨 (2개의 이미지폴더가 존재하므로)

        return all_img_files, all_labels, len(all_img_files), len(class_names)

    def __init__(self, data_set_path, transforms=None):
        self.data_set_path = data_set_path
        self.image_files_path, self.labels, self.length, self.num_classes = self.read_data_set()
        self.transforms = transforms

    def __getitem__(self, index):
        image = Image.open(self.image_files_path[index])
        image = image.convert("RGB") # Tesor는 RGB이므로 변환

        if self.transforms is not None:
            image = self.transforms(image)

        return {'image': image, 'label': self.labels[index]}

    def __len__(self):
        return self.length

transforms_train = transforms.Compose([transforms.Resize((28, 28)),
                                       transforms.RandomRotation(10.),
                                       transforms.ToTensor()])

transforms_test = transforms.Compose([transforms.Resize((28, 28)),
                                      transforms.ToTensor()])

train_data_set = CustomImageDataset(data_set_path="animal/data/train", transforms=transforms_train) # 딕셔너리임{'image': ~, 'label':~}
train_loader = DataLoader(train_data_set, batch_size=hyper_param_batch, shuffle=True)

test_data_set = CustomImageDataset(data_set_path="animal/data/test", transforms=transforms_test)
test_loader = DataLoader(test_data_set, batch_size=hyper_param_batch, shuffle=True)

#print(train_data_set['image'])
#print(train_data_set[2]['image'])

data = []
for idx, data in enumerate(train_loader):
    image_x = data['image']
    label_x = data['label']

# print(image_x[:])
# # print(type(image_x))
#print(image_x[0].shape)
print(image_x.shape)
print(label_x.shape)
#print(len(image_x))
#print(sizeof(image_x))
# # input data reshape

view_data = image_x[:].view(-1,28*28*3)# : -> bath_size = 4 고정, view는 reshape와 동일

view_data = view_data.type(torch.FloatTensor)/255 # 전처리 과정 : 압축시키기


if not (train_data_set.num_classes == test_data_set.num_classes):
   print("error: Numbers of class in training set and test set are not equal")
   exit()

num_classes = train_data_set.num_classes
#print(num_classes)  # 2 출력됨

# model
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(28*28*3, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 12),
            nn.Tanh(),
            nn.Linear(12, 3),   # 입력의 특징을 3차원으로 압축합니다
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.Tanh(),
            nn.Linear(12, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 28*28*3),
            nn.Sigmoid(),       # 픽셀당 0과 1 사이로 값을 출력합니다
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = Autoencoder().to(device)
optimizer = optim.Adam(model.parameters(), lr = 0.005)
criterion = nn.MSELoss()


# 하이퍼파라미터
EPOCH = 10
BATCH_SIZE = 64
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

# train
def train(model,train_loader,optimizer):
    model.train()
    for i_batch, item in enumerate(train_loader):
        #print(item['image'].shape)
        x = item['image'].view(4,-1).to(device) # 인코딩 결과
        y = x.view(4,-1).to(device) # 기존 x 를 카피 -> x와 비교하여 loss 계산 시 사용
        label = item['label'].to(device)

        # 순전파
        encoded, decoded = model(x)
        print(decoded.shape)
        print(y.shape)
        loss = criterion(decoded, y) # 출력단(decoded 결과)은 자율학습에서 출력을 입력에 근사 ,학습을 마치고 나며 의미가 있는 hidden layer만 남기고 출력단은 버린다

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i_batch + 1) % hyper_param_batch == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'
                  .format(epoch, hyper_param_epoch, loss.item()))



EPOCHS = 10
for epoch in range(1, EPOCHS + 1):
    train(model, train_loader, optimizer)
    test_x = view_data.to(device)
    encoded_data, decoded_data = model(test_x)
    f, a = plt.subplots(2, 5, figsize = (10, 4))
    print("[Epoch {}]".format(epoch))
#     for idx in range(5):
#         view_data = view_data[idx].view(4, 3, 28, 28).numpy()
#         img = view_data#np.reshape(view_data.data.numpy(), (28, 28))
#         a[0][idx].imshow(img, cmap = "gray")
#         a[0][idx].set_xticks(())
#         a[0][idx].set_yticks(())
        
#     for idx in range(5):
#         decoded_data = decoded_data[idx].view(4, 3, 28, 28).numpy()
#         img = decoded_data#np.reshape(decoded_data.to("cpu").data.numpy(), (28, 28))
#         a[1][idx].imshow(img, cmap = "gray")
#         a[1][idx].set_xticks(())
#         a[1][idx].set_yticks(())
#     plt.show()