import torchvision.models as models
import torch.nn as nn

#mobilenet3 = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
mobilenet3 = models.mobilenet_v3_small(weights=None)
#resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
resnet18 = models.resnet18(weights=None)
resnet34 = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

class ClassifieurMovie(nn.Module):
    def __init__(self, num_classes=10):
        super(ClassifieurMovie, self).__init__()
        self.model = mobilenet3
        # Remplacer la dernière couche pour la classification binaire
        self.model.classifier = nn.Linear(576, num_classes)
        
    def forward(self, x):
        x = self.model(x)
        return x

class ClassifieurResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ClassifieurResNet18, self).__init__()
        self.model = resnet18
        # Remplacer la dernière couche pour la classification binaire
        self.model.fc = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.model(x)
        return x
    
class ClassifieurResNet34(nn.Module):
    def __init__(self, num_classes=10):
        super(ClassifieurResNet34, self).__init__()
        self.model = resnet34
        # Remplacer la dernière couche pour la classification binaire
        self.model.fc = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.model(x)
        return x