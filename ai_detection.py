import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from PIL import Image
from tqdm.auto import tqdm
# import matplotlib.pyplot as plt 
# uncomment the above to visualize the images

train_transform_trivial_augment = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.TrivialAugmentWide(num_magnitude_bins = 31),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

train_data_augmented = datasets.ImageFolder(train_dir, transform = train_transform_trivial_augment)
test_data_simple = datasets.ImageFolder(test_dir, transform = test_transform)

indices_train = torch.randperm(len(train_data_augmented))[:10000]
sampled_train_data = Subset(train_data_augmented, indices_train)
indices_test = torch.randperm(len(test_data_simple))[:2000]
sampled_test_data = Subset(test_data_simple, indices_test)

batch_size = 32
num_workers = os.cpu_count()

torch.manual_seed(42)
train_augment_loader = DataLoader(
    sampled_train_data,
    batch_size = batch_size,
    num_workers = num_workers,
    shuffle = True
)

test_dataloader = DataLoader(
    sampled_test_data,
    batch_size = batch_size,
    num_workers = num_workers,
    shuffle = False
)

class TinyVGG(nn.Module):
    def __init__(self, input_shape, hidden_units, output_shape):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(input_shape, hidden_units, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_units * 16 * 16, output_shape)
        )

    def forward(self, x):
        return self.classifier(self.conv_block_2(self.conv_block_1(x)))

def train_step(model, dataloader, loss_fn, optimizer):
    model.train()

    train_loss, train_acc = 0, 0
    for batch, (img, label) in enumerate(dataloader):
        img, label = img.to(device), label.to(device)

        label_logits = model(img).squeeze()

        loss = loss_fn(label_logits, label.float())
        train_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        label_pred_class = torch.round(torch.sigmoid(label_logits))
        train_acc += (label_pred_class == label).sum().item() / len(label_logits)

    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

def test_step(model, dataloader, loss_fn, optimizer):
    model.eval()

    test_loss, test_acc = 0, 0

    with torch.inference_mode():
        for batch, (img, label) in enumerate(dataloader):
            img, label = img.to(device), label.to(device)

            label_logits = model(img).squeeze()
            loss = loss_fn(label_logits, label.float())
            test_loss += loss

            test_pred_labels = torch.round(torch.sigmoid(label_logits))
            test_acc += (test_pred_labels == label).sum().item() / len(label_logits)

    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc



def train(model, train_dataloader, test_dataloader, acc_fn, optimizer, epochs = 5):

    results = {
        "train_loss" : [],
        "train_acc" : [],
        "test_loss" : [],
        "test_acc" : []
    }

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model, train_dataloader, loss_fn, optimizer)
        test_loss, test_acc = test_step(model, test_dataloader, loss_fn, optimizer)

        print(
            f"Epoch: {epoch + 1} |"
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        results["train_loss"].append(train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss)
        results["train_acc"].append(train_acc.item() if isinstance(train_acc, torch.Tensor) else train_acc)
        results["test_loss"].append(test_loss.item() if isinstance(test_loss, torch.Tensor) else test_loss)
        results["test_acc"].append(test_acc.item() if isinstance(test_acc, torch.Tensor) else test_acc)

    return results

model_1 = TinyVGG(3, 10, 1)

torch.manual_seed(42)
torch.cuda.manual_seed(42)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(params=model_1.parameters(), lr=0.001)

start_time = timer()
final_results = train(model_1, train_augment_loader, test_dataloader, loss_fn, optimizer, NUM_EPOCHS)
end_time = timer()
print(f'Total training time: {end_time - start_time:.3f} seconds')

