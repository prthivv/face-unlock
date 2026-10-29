import cv2
import torch
from PIL import Image
from facenet_pytorch import MTCNN
from facenet_pytorch.models.utils.detect_face import extract_face

IMAGE_SIZE = 112
MARGIN = 14

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mtcnn = MTCNN(
    image_size=IMAGE_SIZE,
    margin=MARGIN,
    min_face_size=40,
    thresholds=[0.5,0.6,0.6],
    device=device,
)

clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))

def enhance_for_detection(frame_bgr):
    # Boost local contrast so the detector can find faces in dim light
    lab = cv2.cvtColor(frame_bgr,cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    lab = cv2.merge((l, a ,b))

    return cv2.cvtColor(lab,cv2.COLOR_LAB2RGB)

def get_face_tensor(frame_bgr):
    # Returns face tensor for the largest face or None. Face tensor is normalised

    enhanced_rgb = enhance_for_detection(frame_bgr)

    boxes, probs = mtcnn.detect(Image.fromarray(enhanced_rgb))

    if boxes is None:
        return None, None, None

    box = boxes[0]
    prob = probs[0]

    original_rgb = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    face = extract_face(original_rgb,box, image_size=IMAGE_SIZE,margin=MARGIN)

    face = face / 255.0
    face = (face - 0.5) / 0.5
    return face.unsqueeze(0), box, prob

    