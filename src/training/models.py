import torch
import torch.nn as nn
from torchvision import models


class SmallCNN(nn.Module):
    def __init__(self, num_classes):
        super(SmallCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=2)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # new conv layer
        self.pool3 = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

        # Compute the flattened size after conv/pool layers
        dummy_input = torch.zeros(1, 3, 128, 128)
        out = self.pool1(self.relu(self.conv1(dummy_input)))
        out = self.pool2(self.relu(self.conv2(out)))
        out = self.pool3(self.relu(self.conv3(out)))
        self.flattened_size = out.view(1, -1).size(1)

        self.fc1 = nn.Linear(self.flattened_size, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        out = self.pool1(self.relu(self.conv1(x)))
        out = self.pool2(self.relu(self.conv2(out)))
        out = self.pool3(self.relu(self.conv3(out)))
        out = out.view(out.size(0), -1)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out


class PretrainedResNet(nn.Module):
    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super(PretrainedResNet, self).__init__()
        self.model = models.resnet18(pretrained=pretrained)

        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)


class SmallViT(nn.Module):
    def __init__(
        self,
        num_classes: int = 8,
        image_size: int = 128,
        patch_size: int = 16,
        dim: int = 128,
        depth: int = 16,
        heads: int = 32,
        mlp_dim: int = 256,
        channels: int = 3,
        dropout: float = 0.1,
    ):
        super(SmallViT, self).__init__()
        assert image_size % patch_size == 0, "Image dimensions must be divisible by the patch size."
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size * patch_size

        self.patch_size = patch_size
        self.patch_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        self.dropout = nn.Dropout(dropout)

        # First encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout, activation="gelu", batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        B, C, H, W = x.shape
        p = self.patch_size
        x = x.unfold(2, p, p).unfold(3, p, p)
        x = x.contiguous().view(B, C, -1, p, p)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.contiguous().view(B, -1, C * p * p)
        x = self.patch_embedding(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding
        x = self.dropout(x)

        x = self.transformer(x)
        x = self.transformer(x)
        x = self.transformer(x)
        x = self.norm(x)
        cls_output = x[:, 0]
        return self.head(cls_output)


MODEL_REGISTRY = {
    "pretrained_resnet": PretrainedResNet,
    "small_cnn": SmallCNN,
    "small_vit": SmallViT,
    # add more models here as needed
}
