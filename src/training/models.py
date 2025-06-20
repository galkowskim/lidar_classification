import torch.nn as nn
from torchvision import models


class PretrainedResNet(nn.Module):
    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super(PretrainedResNet, self).__init__()
        self.model = models.resnet18(pretrained=pretrained)

        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)


MODEL_REGISTRY = {
    "pretrained_resnet": PretrainedResNet,
    # Add other models here
}
