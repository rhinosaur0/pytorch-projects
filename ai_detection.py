import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
# import matplotlib.pyplot as plt 
# uncomment the above to visualize the images

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform = None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for label, subdir in enumerate(['FAKE', 'REAL']): # iterates through the fake subdir and the real subdir
            subdir_path = os.path.join(self.root_dir, subdir)
            for filename in os.listdir(subdir_path)[:3000]:
                self.image_paths.append(os.path.join(subdir_path, filename))
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

img_dir = '/content/train'
train_data = CustomImageDataset(root_dir = img_dir, transform = transform)
train_loader = DataLoader(
    train_data,
    batch_size = 32,
    shuffle = True
)

img_dir = '/content/test'
test_data = CustomImageDataset(root_dir = img_dir, transform = transform)
test_loader = DataLoader(
    train_data,
    batch_size = 32,
    shuffle = True
)

print(len(train_loader), len(test_loader))

import requests
from pathlib import Path

if Path('helper_functions.py').is_file():
    print('Skipping download')
else:
    print('Downloading helper_functions.py')
    request = requests.get('https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py')
    with open('helper_functions.py', 'wb') as f:
        f.write(request.content)

from helper_functions import accuracy_fn

class AIDetection(nn.Module):
    def __init__(self, input_features, hidden_units, output_features):
        super().__init__()
        self.chain_layer_1 = nn.Sequential(
            nn.Conv2d(input_features, hidden_units, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )

        self.chain_layer_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features = hidden_units * 56 * 56, out_features = 1)
        )

    def forward(self, x):
        return self.classifier(self.chain_layer_2(self.chain_layer_1(x)))

model_0 = AIDetection(3, 16, 1)
print(model_0)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model_0.parameters(), lr = 0.1)

epochs = 3

for epoch in range(epochs):
    model_0.train()
    train_loss = 0
    acc = 0
    for img, label in train_loader:
        y_pred = model_0(img)

        loss = loss_fn(y_pred.squeeze(), torch.tensor(label, dtype = torch.float))
        acc += accuracy_fn(torch.round(y_pred), torch.tensor(label))
        train_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss /= len(train_loader)
    acc /= len(train_loader)

    model_0.eval()
    with torch.inference_mode():
        test_loss = 0
        test_acc = 0
        for img, label in test_loader:
            test_pred = model_0(img)

            loss = loss_fn(test_pred.squeeze(), torch.tensor(label, dtype = torch.float))
            test_acc += accuracy_fn(torch.round(y_pred), torch.tensor(label))
            test_loss += loss
        test_loss /= len(test_loader)
        test_acc /= len(test_loader)

    
    print(f"Loss for this epoch: {train_loss} | Train Accuracy: {acc} | Test Loss: {test_loss} | Test Accuracy: {test_acc}")
