from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn
import torch

class Backbone(nn.Module):
    def __init__(self,embedding_size=512):
        super().__init__()

        base_model=resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.features=nn.Sequential(*list(base_model.children())[:-1])

        self.embedding=nn.Linear(2048,embedding_size)
        self.bn=nn.BatchNorm1d(embedding_size)

    def forward(self,x):
        x=self.features(x)
        x=torch.flatten(x,1)
        x=self.embedding(x)
        x=self.bn(x)
        return x
    
if __name__=='__main__':
    model=Backbone()

    dummy=torch.randn(8,3,112,112)
    out=model(dummy)
    print(out.shape)


