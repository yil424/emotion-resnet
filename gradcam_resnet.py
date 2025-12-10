import os
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt

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
        return img, label, img_path


def build_test_loader(
    data_dir: str,
    batch_size: int = 1,
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
    print(f"[GradCAM] Total images: {len(base_dataset)}")
    print(f"[GradCAM] Classes: {class_names}")

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


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module, device):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
        self.device = device

        self.activations = None
        self.gradients = None

        #forward / backward hook
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: int = None):
        self.model.zero_grad()

        output = self.model(input_tensor)  # [1, num_classes]

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        score = output[:, class_idx]
        score.backward(retain_graph=True)

        grads = self.gradients        # d score / d feature map
        activations = self.activations

        weights = grads.mean(dim=(2, 3), keepdim=True)

        cam = (weights * activations).sum(dim=1)  # [B, H', W']
        cam = torch.relu(cam)

        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        cam = F.interpolate(
            cam.unsqueeze(1),                 # [B, 1, H', W']
            size=(input_tensor.size(2), input_tensor.size(3)),
            mode="bilinear",
            align_corners=False
        )
        cam = cam.squeeze().cpu().numpy()    # [H, W]

        return cam, class_idx


def denormalize(img_tensor: torch.Tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img_tensor.cpu() * std + mean
    img = torch.clamp(img, 0.0, 1.0)
    img_np = img.permute(1, 2, 0).numpy()   # [H, W, 3]
    return img_np


def save_gradcam_figure(
    img_tensor: torch.Tensor,
    cam: np.ndarray,
    class_name: str,
    pred_class_name: str,
    save_path: str
):
    img_np = denormalize(img_tensor.squeeze(0))  # [H, W, 3]

    plt.figure(figsize=(6, 3))

    plt.subplot(1, 2, 1)
    plt.imshow(img_np)
    plt.title(f"Original ({class_name})")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(img_np)
    plt.imshow(cam, cmap="jet", alpha=0.4)
    plt.title(f"Grad-CAM (pred: {pred_class_name})")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved Grad-CAM to {save_path}")


def main():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(project_dir, "Data")
    best_model_path = os.path.join(project_dir, "resnet18_se_best.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    test_loader, class_names = build_test_loader(
        data_dir=data_dir,
        batch_size=1
    )
    num_classes = len(class_names)

    model = ResNet18SE(num_classes=num_classes, pretrained=False).to(device)
    state_dict = torch.load(best_model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    target_layer = model.layer4
    gradcam = GradCAM(model, target_layer, device)

    num_per_class = 3
    saved_count = {i: 0 for i in range(num_classes)}

    out_dir = os.path.join(project_dir, "gradcam_resnet_se")
    os.makedirs(out_dir, exist_ok=True)

    for images, labels, paths in test_loader:
        label = labels.item()
        if saved_count[label] >= num_per_class:
            continue

        images = images.to(device)

        logits = model(images)
        pred = logits.argmax(dim=1).item()

        cam, class_idx = gradcam.generate(images, class_idx=pred)

        class_name = class_names[label]
        pred_class_name = class_names[pred]

        filename = os.path.basename(paths[0])
        save_name = f"gradcam_{class_name}_pred_{pred_class_name}_{filename}.png"
        save_path = os.path.join(out_dir, save_name)

        save_gradcam_figure(images, cam, class_name, pred_class_name, save_path)

        saved_count[label] += 1

        if all(saved_count[i] >= num_per_class for i in range(num_classes)):
            break

    print("Done. Per-class saved counts:", saved_count)


if __name__ == "__main__":
    main()
