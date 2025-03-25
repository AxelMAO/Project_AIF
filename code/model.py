import torchvision.models as models
import torch.nn as nn

mobinet = models.mobilenet_v3_small(pretrained=True)

class Classifieur(nn.Module):
    def __init__(self, num_layers):
        super(Classifieur, self).__init__()
        self.model = model
        self.fc = nn.Linear(1000, 2)
        
    def forward(self, x):
        x = self.model(x)
        x = self.fc(x)
        return x
