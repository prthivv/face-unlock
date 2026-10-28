import os
from pathlib import Path
from PIL import Image
from facenet_pytorch import MTCNN
import numpy as np
import tqdm

mtcnn = MTCNN(
    image_size=112,
    margin=14,
    min_face_size=40,
    keep_all=False,
    device='cpu'
)


def process_dataset(input_dir, output_dir):
    """
    Detect and align the face in every image under input_dir.
    Expects input_dir/<identity>/<image>.jpg and writes aligned
    112x112 crops to output_dir/<identity>/<image>.jpg.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    failed_count = 0

    for identity_folder in tqdm.tqdm(list(input_path.iterdir()), desc="Identities"):
        if not identity_folder.is_dir():
            continue

        out_identity = output_path / identity_folder.name
        out_identity.mkdir(parents=True, exist_ok=True)

        for img_file in identity_folder.glob("*.jpg"):
            try:
                img = Image.open(img_file).convert("RGB")
                face = mtcnn(img)

                if face is None:
                    failed_count += 1
                    continue

                face_np = ((face.permute(1, 2, 0) + 1) / 2 * 255).numpy().astype("uint8")
                Image.fromarray(face_np).save(out_identity / img_file.name)

            except Exception:
                failed_count += 1

    print(f"Processing completed. Failed to detect faces in {failed_count} images.")


if __name__ == "__main__":
    process_dataset("data/raw/casia-webface/casia-webface", "data/processed/casia-webface")
    process_dataset("data/raw/lfw/lfw", "data/processed/lfw")
