import cv2
from pathlib import Path

from config import REELS_DIR, OUTPUT_DIR
from ocr_utils import extract_text_from_frame
from filtering import is_redundant

def process_video(video_path: Path):
    print(f"Processing {video_path.name}...")
    cap = cv2.VideoCapture(str(video_path))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    frame_id = 0
    collected_text = []
    last_captions = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        text = extract_text_from_frame(frame)

        if text and not is_redundant(text, last_captions[-30:]):
            print(f"[{frame_id}] {text}")
            collected_text.append((frame_id, text))
            last_captions.append((frame_id, text))

        frame_id += 1

    cap.release()

    with open(OUTPUT_DIR / f"{video_path.stem}_captions.txt", "w") as f:
        for fid, t in collected_text:
            f.write(f"[{fid}] {t}\n")

if __name__ == "__main__":
    for mp4 in REELS_DIR.glob("*.mp4"):
        process_video(mp4)
