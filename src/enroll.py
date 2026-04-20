import os
import cv2
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN

from models.backbone import Backbone
from data.dataset import get_transforms


def enroll():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Backbone().to(device)
    checkpoint = torch.load('checkpoints/best.pth', map_location=device)
    model.load_state_dict(checkpoint['backbone'])
    model.eval()

    transform = get_transforms(train=False)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if not ret:
        print("Failed to open webcam")
        return

    embeddings_list = []

    print("Press SPACE to capture (10 images needed)")
    print("Press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow('Enrollment', frame)
        key = cv2.waitKey(1)

        if key == ord('q'):
            break

        if key == 32:
            # Detect face with OpenCV
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) == 0:
                print("No face detected. Try again.")
                continue
            
            # Get the largest face
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            
            # Crop and resize to 112x112
            face_crop = frame[y:y+h, x:x+w]
            face_resized = cv2.resize(face_crop, (112, 112))
            
            # Convert to RGB and normalize manually (CHANGED THIS PART)
            rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            
            # Convert to tensor and normalize to [-1, 1]
            face_tensor = torch.tensor(rgb.tolist(), dtype=torch.uint8).permute(2, 0, 1).float() / 255.0
            face_tensor = (face_tensor - 0.5) / 0.5  # Normalize to [-1, 1]
            face_tensor = face_tensor.unsqueeze(0).to(device)
            
            # Get embedding
            with torch.no_grad():
                emb = model(face_tensor)
            
            emb = torch.nn.functional.normalize(emb, dim=1)
            embeddings_list.append(emb.squeeze(0))
            
            print(f"Captured {len(embeddings_list)}/10")
            
            if len(embeddings_list) == 10:
                print("Capture complete!")
                break
    if len(embeddings_list) > 0:
        embeddings = torch.stack(embeddings_list)
        mean_embedding = torch.mean(embeddings, dim=0)

        name = input("Enter your name: ")

        os.makedirs('data/register', exist_ok=True)

        save_path = f'data/register/{name}.npy'
        np.save(save_path, mean_embedding.cpu().numpy())

        print(f"Saved embedding to {save_path}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    enroll()