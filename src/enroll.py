import os
import cv2
import torch
import numpy as np

from models.backbone import Backbone
from face_utils import get_face_tensor


def enroll():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Backbone().to(device)
    checkpoint = torch.load('checkpoints/best.pth', map_location=device)
    model.load_state_dict(checkpoint['backbone'])
    model.eval()


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

        face_tensor,box,prob=get_face_tensor(frame)

        display = frame.copy()

        if box is not None:
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display, f"Face {prob:.2f}", (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(display, "No face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.putText(display, f"Captured {len(embeddings_list)}/10",
                    (10, display.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


        cv2.imshow('Enrollment', display)
        key = cv2.waitKey(1)

        if key == ord('q'):
            break

        if key == 32:

            if face_tensor is None:
                print("No face detected. Try again.")
                continue
            face_tensor = face_tensor.to(device)
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