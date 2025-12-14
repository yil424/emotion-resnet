import os
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from PIL import Image

from vit_emotion import EmotionViT


#  Dataset & Dataloaders
class EmotionSubset(Dataset):
    def __init__(self, base_dataset, indices, transform=None):
        self.base_dataset = base_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        orig_idx = self.indices[idx]
        img_path, label = self.base_dataset.samples[orig_idx]  # (path, class_idx)

        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        return img, label


def build_dataloaders(
    data_dir: str,
    batch_size: int = 64,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> Tuple[DataLoader, DataLoader, DataLoader, list]:

    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])

    eval_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    base_dataset = ImageFolder(root=data_dir)
    class_names = base_dataset.classes
    print(f"[ViT] Total images: {len(base_dataset)}")
    print(f"[ViT] Classes: {class_names}")

    n_total = len(base_dataset)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val

    print(f"[ViT] Split sizes -> train: {n_train}, val: {n_val}, test: {n_test}")

    torch.manual_seed(42)
    indices = torch.randperm(n_total).tolist()

    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    train_dataset = EmotionSubset(base_dataset, train_indices, transform=train_transform)
    val_dataset = EmotionSubset(base_dataset, val_indices, transform=eval_transform)
    test_dataset = EmotionSubset(base_dataset, test_indices, transform=eval_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=0, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=0, pin_memory=True
    )

    return train_loader, val_loader, test_loader, class_names



#  Train / Eval Helpers
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)      # [B, num_classes]
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def eval_one_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc



#  Main Training Script
def main():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(project_dir, "Data")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[ViT] Using device:", device)

    batch_size = 64
    num_epochs = 20
    learning_rate = 3e-4

    train_loader, val_loader, test_loader, class_names = build_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size
    )

    model = EmotionViT(
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=len(class_names),
        embed_dim=256,
        depth=6,
        num_heads=4,
        mlp_ratio=4.0,
        drop_rate=0.1,
        attn_drop_rate=0.0,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)

    best_val_acc = 0.0
    best_model_path = os.path.join(project_dir, "vit_emotion_best.pth")

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = eval_one_epoch(
            model, val_loader, criterion, device
        )

        print(
            f"Epoch [{epoch}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}  "
            f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> New best ViT model saved with val_acc = {best_val_acc:.4f}")

    print(f"[ViT] Training finished. Best val_acc = {best_val_acc:.4f}")


    #  Evaluate on test set
    print("\n[ViT] Evaluating on test set using best model...")

    best_model = EmotionViT(
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=len(class_names),
        embed_dim=256,
        depth=6,
        num_heads=4,
        mlp_ratio=4.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
    ).to(device)

    state_dict = torch.load(best_model_path, map_location=device)
    best_model.load_state_dict(state_dict)

    test_loss, test_acc = eval_one_epoch(
        best_model, test_loader, criterion, device
    )
    print(f"[ViT] Test Loss: {test_loss:.4f}  Test Acc: {test_acc:.4f}")


if __name__ == "__main__":
    main()
