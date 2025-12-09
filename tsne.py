import os
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


#Dataset / Dataloader
class EmotionSubset(Dataset):
    def __init__(self, base_dataset, indices, transform=None):
        self.base_dataset = base_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        orig_idx = self.indices[idx]
        img_path, label = self.base_dataset.samples[orig_idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def build_dataloaders_for_tsne(
    data_dir: str,
    batch_size: int = 64,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
):
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
    print(f"[TSNE] Total images: {len(base_dataset)}")
    print(f"[TSNE] Classes: {class_names}")

    n_total = len(base_dataset)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val

    torch.manual_seed(42)
    indices = torch.randperm(n_total).tolist()
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    test_dataset = EmotionSubset(base_dataset, test_indices, transform=eval_transform)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    return test_loader, class_names


#ResNet18 + SE

class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ResNet18SE(nn.Module):
    def __init__(self, num_classes: int = 5, pretrained: bool = False):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.se = SEBlock(channels=512, reduction=16)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.se(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def extract_features(self, x):
        """
        提取倒数第二层的 512 维 embedding（在 fc 之前）
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.se(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)   # [B, 512]
        return x


# Extract test features + t-SNE

def main():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(project_dir, "Data")
    best_model_path = os.path.join(project_dir, "resnet18_se_best.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    test_loader, class_names = build_dataloaders_for_tsne(
        data_dir=data_dir,
        batch_size=64
    )

    num_classes = len(class_names)

    model = ResNet18SE(num_classes=num_classes, pretrained=False).to(device)
    state_dict = torch.load(best_model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    all_features = []
    all_labels = []

    max_samples = 2000
    collected = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            feats = model.extract_features(images)  # [B, 512]

            all_features.append(feats.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            collected += labels.size(0)
            if collected >= max_samples:
                break

    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    if features.shape[0] > max_samples:
        features = features[:max_samples]
        labels = labels[:max_samples]

    print("Collected features shape:", features.shape)  # [N, 512]

    # t-SNE to 2D
    print("Running t-SNE (this may take a while)...")
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, init="pca", random_state=42)
    features_2d = tsne.fit_transform(features)  # [N, 2]

    #Scatter plot
    plt.figure(figsize=(8, 6))
    num_classes = len(class_names)
    colors = ['red', 'green', 'blue', 'orange', 'purple']

    for i, cls_name in enumerate(class_names):
        idx = labels == i
        plt.scatter(
            features_2d[idx, 0],
            features_2d[idx, 1],
            s=5,
            alpha=0.6,
            label=cls_name,
            c=colors[i]
        )

    plt.legend()
    plt.title("t-SNE of ResNet18-SE penultimate features (test set)")
    plt.tight_layout()

    out_path = os.path.join(project_dir, "resnet18_se_tsne.png")
    plt.savefig(out_path, dpi=300)
    print(f"t-SNE plot saved to: {out_path}")


if __name__ == "__main__":
    main()
