import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F

from models.backbone import Backbone

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

    # 3. Initialize the face cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

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

        # Detect face with OpenCV
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            # Get the largest face
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            
            # Crop and resize to 112x112
            face_crop = frame[y:y+h, x:x+w]
            face_resized = cv2.resize(face_crop, (112, 112))
            
            # Convert to RGB and normalize manually
            rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            
            # Convert to tensor and normalize to [-1, 1] using the exact same robust method as enroll.py
            face_tensor = torch.tensor(rgb.tolist(), dtype=torch.uint8).permute(2, 0, 1).float() / 255.0
            face_tensor = (face_tensor - 0.5) / 0.5  # Normalize to [-1, 1]
            face_tensor = face_tensor.unsqueeze(0).to(device)
            
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

            # Draw a bounding box around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

            # 6. Overlay the result on the video frame using cv2.putText()
            font = cv2.FONT_HERSHEY_SIMPLEX
            # Place the text slightly above the bounding box
            cv2.putText(frame, display_text, (x, y - 10), font, 0.8, color, 2, cv2.LINE_AA)

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
