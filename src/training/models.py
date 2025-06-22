import torch
import torch.nn as nn
from torchvision import models


class SmallCNN(nn.Module):
    def __init__(self, num_classes):
        super(SmallCNN, self).__init__()
        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # 64 -> 32
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)  # 64 -> 32
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 128 -> 64
        self.conv4 = nn.Conv2d(64, 96, kernel_size=3, padding=1)  # 128 -> 64
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Conv2d(96, 128, kernel_size=3, padding=1)  # 256 -> 128
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)  # 256 -> 128
        self.pool3 = nn.MaxPool2d(2, 2)

        # Dynamically compute flattened size
        dummy_input = torch.zeros(1, 3, 128, 128)
        out = self._forward_conv(dummy_input)
        self.flattened_size = out.view(1, -1).size(1)

        self.fc1 = nn.Linear(self.flattened_size, 512)  # 512 -> 256
        self.fc2 = nn.Linear(512, num_classes)

    def _forward_conv(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool1(x)

        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.pool2(x)

        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.pool3(x)

        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class PretrainedResNet(nn.Module):
    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super(PretrainedResNet, self).__init__()
        self.model = models.resnet18(pretrained=pretrained)

        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)


class PretrainedVGG16(nn.Module):
    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super(PretrainedVGG16, self).__init__()
        self.model = models.vgg16(pretrained=pretrained)

        in_features = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)


class PretrainedEfficientNet(nn.Module):
    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super(PretrainedEfficientNet, self).__init__()
        self.model = models.efficientnet_b0(pretrained=pretrained)

        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)


class PretrainedConvnext(nn.Module):
    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super(PretrainedConvnext, self).__init__()
        self.model = models.convnext_small(pretrained=pretrained)

        in_features = self.model.classifier[2].in_features
        self.model.classifier[2] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)


class SmallViT(nn.Module):
    def __init__(
        self,
        num_classes: int = 8,
        image_size: int = 128,
        patch_size: int = 16,
        dim: int = 256,  # doubled
        depth: int = 24,  # increased
        heads: int = 32,  # keep or increase to 48 for even more params
        mlp_dim: int = 512,  # doubled
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
        x = self.norm(x)
        cls_output = x[:, 0]
        return self.head(cls_output)


MODEL_REGISTRY = {
    "pretrained_resnet": PretrainedResNet,
    "small_cnn": SmallCNN,
    "small_vit": SmallViT,
    "pretrained_vgg16": PretrainedVGG16,
    "pretrained_efficientnet": PretrainedEfficientNet,
    "pretrained_convnext": PretrainedConvnext,
    # add more models here as needed
}
