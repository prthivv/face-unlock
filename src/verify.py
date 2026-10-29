import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F

from models.backbone import Backbone
from face_utils import get_face_tensor

def verify():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Load your best checkpoint
    model = Backbone().to(device)
    checkpoint = torch.load('checkpoints/best.pth', map_location=device)
    model.load_state_dict(checkpoint['backbone'])
    model.eval()

    # 2. Load all enrolled templates from data/register/
    templates = {}
    register_dir = 'data/register'
    if os.path.exists(register_dir):
        for npy_file in os.listdir(register_dir):
            if npy_file.endswith('.npy'):
                name = npy_file.replace('.npy', '')
                embedding = np.load(os.path.join(register_dir, npy_file))
                # Ensure the embedding tensor is on the same device
                templates[name] = torch.tensor(embedding.tolist(), dtype=torch.float32).to(device)
    
    if not templates:
        print("No enrolled templates found. Please run enroll.py first to register a face.")
        return

    # 4. Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open webcam")
        return

    print("Verification started. Press Q to quit.")

    # 5. Continuous verification loop
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        face_tensor,box,prob = get_face_tensor(frame)

        font = cv2.FONT_HERSHEY_SIMPLEX

        if face_tensor is not None:

            face_tensor = face_tensor.to(device)
            
            # Get embedding
            with torch.no_grad():
                emb = model(face_tensor)
            
            # Normalize embedding
            emb = F.normalize(emb, dim=1)
            emb = emb.squeeze(0)

            # Compare embedding against all enrolled templates using cosine similarity
            max_similarity = -1.0
            best_match_name = "UNKNOWN"

            for name, template_emb in templates.items():
                sim = F.cosine_similarity(emb.unsqueeze(0), template_emb.unsqueeze(0)).item()
                if sim > max_similarity:
                    max_similarity = sim
                    best_match_name = name

            # Determine text and color based on the 0.6 threshold
            if max_similarity > 0.6:
                display_text = f"UNLOCKED: {best_match_name} ({max_similarity:.2f})"
                color = (0, 255, 0)  # Green for unlocked
            else:
                display_text = f"UNKNOWN ({max_similarity:.2f})"
                color = (0, 0, 255)  # Red for unknown
            
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, display_text, (x1, max(y1 - 10, 20)), font, 0.8, color, 2, cv2.LINE_AA)

        else:
            cv2.putText(frame, "No face detected", (10,30), font, 0.8, (0,0,255), 2, cv2.LINE_AA)


        cv2.imshow('Live Verification', frame)
        
        # Press 'q' to exit the loop
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    verify()
