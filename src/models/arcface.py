import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbone import Backbone


class ArcFaceLoss(nn.Module):
    def __init__(self,num_classes,embedding_dim=512,s=64.0,m=0.1):
        super().__init__()
        self.num_classes=num_classes
        self.embedding_dim=embedding_dim
        self.s=s
        self.m=m

        self.weight=nn.Parameter(torch.empty(num_classes,embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        
        import math
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self,embeddings,labels):
        # Cast to float32 to prevent AMP-related exploding gradients/NaNs
        embeddings = embeddings.float()
        self.weight.data = self.weight.data.float()

        embeddings = F.normalize(embeddings, dim=1)
        weights = F.normalize(self.weight, dim=1)

        cosine = F.linear(embeddings, weights)
        # Use 1e-3 for clamping, because 1e-7 evaluates to 0.0 in float16!
        # This prevents the collapse even if embeddings remain in float16.
        cosine = cosine.clamp(-1.0 + 1e-3, 1.0 - 1e-3)

        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))

        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        loss = F.cross_entropy(output, labels)
        return loss

if __name__=='__main__':
    model=Backbone()
    arcface_loss=ArcFaceLoss(num_classes=10537)

    images=torch.randn(8,3,112,112)
    labels=torch.randint(0,10537,(8,))
    embeddings=model(images)
    loss=arcface_loss(embeddings,labels)
    print(f'Embeddings shape: {embeddings.shape}')
    print(f'ArcFace Loss: {loss.item()}')