import os
import cv2
import re
import pytesseract
import difflib
from difflib import SequenceMatcher
from pathlib import Path

# Path settings
SCRIPT_DIR = Path(__file__).parent.resolve()
REELS_DIR = SCRIPT_DIR.parent / "reels"
OUTPUT_DIR = SCRIPT_DIR.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# OCR config
pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"

def clean_text(text):
    lines = text.splitlines()
    cleaned = []

    for line in lines:
        line = line.strip()

        # Skip empty or known junk lines
        if not line:
            continue
        if re.search(r"CRUSH\s+LIVE\s+POKER", line, re.IGNORECASE):
            continue
        if re.search(r"[|()/\\@#%^&*_=+\[\]{}<>]", line):  # junk characters / HUD overlays
            continue
        if re.search(r"[a-z]", line):  # exclude lines containing lowercase letters
            continue
        if re.fullmatch(r"[A-Z]{1,3}", line):  # short noise like "J", "SB"
            continue
        if len(line.split()) == 1 and line.isupper() and not line.startswith("$"):
            continue
        if not re.fullmatch(r"[A-Z0-9$,! '\-]+", line):  # allow only valid caption characters
            continue
        if len(line) <= 2:
            continue

        cleaned.append(line)

    return " ".join(cleaned)


def extract_text_from_frame(frame):
    h, w, _ = frame.shape

    # Crop bottom-center region for captions (tweak these as needed)
    y_start = int(h * 0.58)     # start at 58% height
    y_end = int(h * 0.7)      # end at 70% height (bottom 58% to 30%)
    x_start = int(w*0.07)
    x_end = int(w*0.93)

    cropped = frame[y_start:y_end, x_start:x_end]

    # cv2.imshow("Cropped Caption Region", cropped)
    # cv2.waitKey(0)  # Display for a brief moment
    # cv2.destroyAllWindows()

    # Convert to grayscale and threshold
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    # Run OCR
    raw_text = pytesseract.image_to_string(thresh, config='--oem 3 --psm 6')

    cleaned_text = clean_text(raw_text)

    return cleaned_text.strip()

def is_similar(a: str, b: str, threshold: float = 0.88) -> bool:
    #old code for redundant texts
    return SequenceMatcher(None, a, b).ratio() > threshold

def is_redundant(new_text, recent_texts, min_overlap_ratio=0.8):
    for _, prev in recent_texts:
        # Normalize
        a = new_text.strip().upper()
        b = prev.strip().upper()

        # Check if one is contained in the other
        if a in b or b in a:
            return True

        # Check token overlap (e.g., 4/5 tokens match)
        tokens_a = set(a.split())
        tokens_b = set(b.split())
        overlap = tokens_a & tokens_b
        if len(overlap) / max(len(tokens_a), 1) > min_overlap_ratio:
            return True

    return False


def process_video(video_path):
    print(f"Processing {video_path.name}...")
    cap = cv2.VideoCapture(str(video_path))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(frame_rate * 0.1)

    frame_id = 0
    collected_text = []

    last_captions = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # if frame_id % frame_interval == 0:

        text = extract_text_from_frame(frame)
        
        if text:
            if not is_redundant(text, last_captions[-30:]):
                print(f"[{frame_id}] {text}")
                collected_text.append((frame_id, text))
                last_captions.append((frame_id, text))

        frame_id += 1

    cap.release()

    #save results
    with open(OUTPUT_DIR / f"{video_path.stem}_captions.txt", "w") as f:
        for fid, t in collected_text:
            f.write(f"[{fid}] {t}\n")

if __name__ == "__main__":
    for mp4 in REELS_DIR.glob("*.mp4"):
        process_video(mp4)
