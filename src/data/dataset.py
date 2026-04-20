import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class CasiaDataset(Dataset):
    def __init__(self,root_dir,txt_file,transform=None):
        self.root_dir=root_dir
        self.transform=transform
        self.samples=[]
        missing_count=0
        with open(txt_file,'r') as f:
            for line in f:
                parts=line.strip().split()
                if len(parts)<2:
                    continue
                label=int(parts[0])
                rel_path=parts[1]
                img_path=os.path.join(root_dir,rel_path)
                if(os.path.exists(img_path)):
                    self.samples.append((img_path,label))
                else:
                    missing_count+=1
        print(f"Missing images: {missing_count}")
        
        unique_labels=sorted(set(label for _,label in self.samples))

        label_map={old:new for new,old in enumerate(unique_labels)}

        self.samples=[(img_path,label_map[label]) for img_path,label in self.samples]
        
        self.num_classes=len(unique_labels)
        labels = [label for _, label in self.samples]
        print("Min label:", min(labels))
        print("Max label:", max(labels))
        print("Num classes:", self.num_classes)

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self,idx):
        img_path,label=self.samples[idx]
        image=Image.open(img_path).convert('RGB')
        if self.transform:
            image=self.transform(image)
        return image,label
    
class CustomToTensor:
    def __call__(self, pic):
        import torch
        channels = len(pic.getbands())
        img_tensor = torch.frombuffer(bytearray(pic.tobytes()), dtype=torch.uint8)
        img_tensor = img_tensor.reshape(pic.height, pic.width, channels)
        return img_tensor.permute(2, 0, 1).float() / 255.0

def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2,contrast=0.2),
            CustomToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
        ])
    else:
        return transforms.Compose([
            CustomToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
        ])

if __name__=='__main__':
    dataset=CasiaDataset(root_dir='data/processed/',txt_file='data/casia-webface.txt',transform=get_transforms(train=True))
    print(f'Number of samples: {len(dataset)}')
    print(f'Number of classes: {dataset.num_classes}')
    from torch.utils.data import DataLoader
    dataloader=DataLoader(dataset,batch_size=128,shuffle=True)

    imgs,labels=next(iter(dataloader))
    print(f'Batch of images shape: {imgs.shape}')
    print(f'Batch of labels shape: {labels.shape}')