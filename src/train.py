import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from data.dataset import CasiaDataset, get_transforms
from models.backbone import Backbone
from models.arcface import ArcFaceLoss

def train():
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset=CasiaDataset(
        root_dir='data/processed/',
        txt_file='data/casia-webface.txt',
        transform=get_transforms(train=True)
    )

    loader=DataLoader(dataset,batch_size=256,shuffle=True,num_workers=6,pin_memory=True)

    backbone=Backbone().to(device)
    arcface_loss=ArcFaceLoss(num_classes=dataset.num_classes).to(device)

    optimizer = torch.optim.SGD(list(backbone.parameters()) + list(arcface_loss.parameters()),lr=0.001,momentum=0.9,weight_decay=5e-4)

    scheduler=StepLR(optimizer,step_size=10,gamma=0.1)

    scaler=torch.cuda.amp.GradScaler()

    start_epoch=0
    num_epochs=25

    checkpoint_path = None # 'checkpoints/epoch_15.pth'
    best_loss=float('inf')
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Resuming from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path,map_location=device)

        backbone.load_state_dict(checkpoint['backbone'])
        arcface_loss.load_state_dict(checkpoint['arcface_loss'])
        optimizer.load_state_dict(checkpoint['optimizer'])

        start_epoch = checkpoint['epoch']  # already saved as epoch+1 in your code

        scheduler.last_epoch = start_epoch - 1

        import math

        arcface_loss.m = 0.5
        arcface_loss.cos_m = math.cos(0.5)
        arcface_loss.sin_m = math.sin(0.5)
        arcface_loss.th = math.cos(math.pi - 0.5)
        arcface_loss.mm = math.sin(math.pi - 0.5) * 0.5

        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.03
        
    
    
    for epoch in range(start_epoch,num_epochs):
        backbone.train()
        arcface_loss.train()

        running_loss=0.0
        
        pbar=tqdm(loader,desc=f'Epoch {epoch+1}/{num_epochs}')

        for images,labels in pbar:
            images=images.to(device)
            labels=labels.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                embeddings=backbone(images)
                loss=arcface_loss(embeddings,labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss+=loss.item()

            pbar.set_postfix(loss=loss.item())

        avg_loss=running_loss/len(loader)
        print(f'Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}')

        print("\n=== Validating embeddings ===")
        backbone.eval()
        with torch.no_grad():
            from PIL import Image
            sample_paths = [
                ('data/processed/casia-webface/000000/00000001.jpg', 0),
                ('data/processed/casia-webface/000001/00000016.jpg', 1),
                ('data/processed/casia-webface/000002/00000275.jpg', 2)
            ]
            embeddings = []
            for path, label in sample_paths:
                img = Image.open(path).convert('RGB')
                img_t = get_transforms(train=False)(img).unsqueeze(0).to(device)
                emb = backbone(img_t)[0]
                embeddings.append(emb)
                print(f"Class {label} embedding (first 10): {emb[:10]}")
            
            # Check pairwise similarities
            for i in range(3):
                for j in range(i+1, 3):
                    sim = F.cosine_similarity(embeddings[i].unsqueeze(0), embeddings[j].unsqueeze(0)).item()
                    print(f"Similarity between class {i} and {j}: {sim:.4f}")
        backbone.train()
        print("=== Validation done ===\n")

        scheduler.step()
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            os.makedirs('checkpoints', exist_ok=True)
            torch.save({
                'epoch': epoch+1,
                'backbone': backbone.state_dict(),
                'arcface_loss': arcface_loss.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, f'checkpoints/best.pth')
        
        if(epoch+1)%5==0:
            os.makedirs('checkpoints',exist_ok=True)
            torch.save({
                'epoch': epoch+1,
                'backbone': backbone.state_dict(),
                'arcface_loss': arcface_loss.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, f'checkpoints/epoch_{epoch+1}.pth')
if __name__=='__main__':
    train()