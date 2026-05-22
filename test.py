import torch
print(torch.__version__, torch.cuda.is_available())

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.datasets import OxfordIIITPet
from torchvision import transforms

import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self, num_classes=37):
        super().__init__()
        self.features = nn.Sequential(
            #CNN 1
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            #CNN 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            #CNN 3
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            #CNN 4
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            #CNN 5
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        logits = self.head(x)
        return logits  # return raw logits

#determine the contraints for the image to be calculated
IMG_SIZE = 128

train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.ToTensor(),
])

test_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

test_set = datasets.OxfordIIITPet(root="data", split="test",
                                  target_types="category",
                                  transform=test_tf, download=True)

test_loader = DataLoader(test_set, batch_size=32)

import torch.nn as nn
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Training on device:", device)

model = NeuralNetwork().to(device)
if os.path.exists("model.pth"):
    model.load_state_dict(torch.load("model.pth", weights_only=True))
    print("Loaded saved model weights.")
else:
    print("No saved model found — using a fresh, untrained model.")
print(model)

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    classes = test_set.classes
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        p_name = [classes[p] for p in preds]
        t_name = [classes[t] for t in labels]
        total += labels.size(0)
        print(f'prediction:{p_name[0]}, actual:{t_name[0]}')
    return correct / total

test_acc = evaluate(model, test_loader)
print(f"Test accuracy:{test_acc}")
