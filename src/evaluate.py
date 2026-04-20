import os
import csv
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import numpy as np

from models.backbone import Backbone
from data.dataset import get_transforms

def load_image(path,transform):
    image=Image.open(path).convert('RGB')
    return transform(image)

def get_embedding(model,image,device):
    image=image.unsqueeze(0).to(device)
    with torch.no_grad():
        embedding=model(image)
    return embedding.squeeze(0)

def evaluate():
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model=Backbone().to(device)
    checkpoint=torch.load('checkpoints/best.pth',map_location=device)
    model.load_state_dict(checkpoint['backbone'])
    model.eval()

    transform=get_transforms(train=False)

    lfw_root="data/processed/lfw"
    match_csv="data/processed/lfw/matchpairsDevTest.csv"
    mismatch_csv="data/processed/lfw/mismatchpairsDevTest.csv"

    scores=[]
    labels=[]

    with open(match_csv,'r') as f:
        reader=csv.reader(f)
        next(reader)  # Skip header
        
        for i,row in enumerate(tqdm(reader,desc="Match Pairs")):
            name,img1,img2=row
            path1=os.path.join(lfw_root,name,f"{name}_{int(img1):04d}.jpg")
            path2=os.path.join(lfw_root,name,f"{name}_{int(img2):04d}.jpg")

            if not os.path.exists(path1) or not os.path.exists(path2):
                continue

            img1=load_image(path1,transform)
            img2=load_image(path2,transform)

            emb1=F.normalize(get_embedding(model,img1,device),dim=0)
            emb2=F.normalize(get_embedding(model,img2,device),dim=0)

            sim=F.cosine_similarity(emb1.unsqueeze(0),emb2.unsqueeze(0)).item()
            scores.append(sim)
            labels.append(1)
    
    with open(mismatch_csv,'r') as f:
        reader=csv.reader(f)
        next(reader)  # Skip header
        
        for i,row in enumerate(tqdm(reader,desc="Mismatch Pairs")):
            name1,img1,name2,img2=row

            path1=os.path.join(lfw_root,name1,f"{name1}_{int(img1):04d}.jpg")
            path2=os.path.join(lfw_root,name2,f"{name2}_{int(img2):04d}.jpg")

            if not os.path.exists(path1) or not os.path.exists(path2):
                continue

            img1=load_image(path1,transform)
            img2=load_image(path2,transform)

            emb1=F.normalize(get_embedding(model,img1,device),dim=0)
            emb2=F.normalize(get_embedding(model,img2,device),dim=0)

            sim=F.cosine_similarity(emb1.unsqueeze(0),emb2.unsqueeze(0)).item()
            scores.append(sim)
            labels.append(0)
    
    scores=np.array(scores)
    labels=np.array(labels)

    best_acc=0
    best_thresh=0

    for thresh in np.arange(0.0,1.0,0.005):
        preds=(scores>thresh).astype(int)
        acc=(preds==labels).mean()

        if acc>best_acc:
            best_acc=acc
            best_thresh=thresh
    
    print(f"Best Accuracy: {best_acc:.4f} at Threshold: {best_thresh:.2f}")

    try:
        from sklearn.metrics import roc_auc_score
        auc=roc_auc_score(labels,scores)
        print(f"ROC-AUC: {auc:.4f}")
    except:
        print("sklearn not installed, skipping ROC-AUC calculation")

if __name__=='__main__':
    evaluate()
