import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================

DATA_DIR = "candidate_tiles"
MODEL_DIR = "models"

IMG_SIZE = 64
BATCH_SIZE = 32
NUM_EPOCHS = 5
LEARNING_RATE = 1e-3

NUM_WORKERS = 0
SEED = 42

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("DEVICE")
print("=" * 70)
print(device)


# ============================================================
# TRANSFORM
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ============================================================
# DATASET
# ============================================================

def main():
    dataset = datasets.ImageFolder(
        root=DATA_DIR,
        transform=train_transform,
    )

    classes = dataset.classes
    num_classes = len(classes)

    print("\n" + "=" * 70)
    print("DATASET")
    print("=" * 70)
    print("Classes:")
    for i, cls in enumerate(classes):
        print(f"{i}: {cls}")
    print("\nTotal images:", len(dataset))

    # ============================================================
    # DATALOADER
    # ============================================================
    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    # ============================================================
    # CNN
    # ============================================================
    class BasicCNN(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(128, num_classes),
            )

        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x

    model = BasicCNN(num_classes=num_classes).to(device)

    print("\n" + "=" * 70)
    print("MODEL")
    print("=" * 70)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        for images, labels in progress:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            progress.set_postfix(loss=f"{loss.item():.4f}")

        epoch_loss = running_loss / total
        epoch_accuracy = correct / total
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        print(f"Loss     : {epoch_loss:.4f}")
        print(f"Accuracy : {epoch_accuracy * 100:.2f}%")
        print(f"LR       : {optimizer.param_groups[0]['lr']:.6f}")
        scheduler.step()

    model_path = os.path.join(MODEL_DIR, "basic_cnn.pth")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "classes": classes,
            "img_size": IMG_SIZE,
        },
        model_path,
    )

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print("Model saved to:")
    print(model_path)


if __name__ == "__main__":
    main()

