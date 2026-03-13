# scripts/ocr_utils.py
import re
import cv2
import numpy as np
import pytesseract
from pathlib import Path
from config import TESSERACT_PATH, CROP_REGION

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

def detect_speaker_color(caption_crop) -> str:
    """
    Analyzes the caption image region to determine if the speaker is Bart (white text) or the Caller (yellow text).
    Assumes `caption_crop` is the cropped image region containing the subtitle text.

    Returns:
        "Bart" or "Caller"
    """

    hsv = cv2.cvtColor(caption_crop, cv2.COLOR_BGR2HSV)

    bright_mask = hsv[:, :, 2] > 180 

    hue = hsv[:, :, 0][bright_mask]
    sat = hsv[:, :, 1][bright_mask]

    if len(hue) == 0:
        return "Unknown"

    avg_hue = np.mean(hue)
    avg_sat = np.mean(sat)

    if avg_hue > 20 and avg_hue < 40 and avg_sat > 100:
        return "Caller"
    else:
        return "Bart"


def clean_text(text: str) -> str:
    lines = text.splitlines()
    cleaned = []

    for line in lines:
        line = line.strip()

        if not line:
            continue
        if re.search(r"CRUSH\s+LIVE\s+POKER", line, re.IGNORECASE):
            continue
        if re.search(r"[|()/\\@#%^&*_=+\[\]{}<>]", line):
            continue
        if re.search(r"[a-z]", line):
            continue
        if re.fullmatch(r"[A-Z]{1,3}", line):
            continue
        if len(line.split()) == 1 and line.isupper() and not line.startswith("$"):
            continue
        if not re.fullmatch(r"[A-Z0-9$,! '\-]+", line):
            continue
        if len(line) <= 2:
            continue

        cleaned.append(line)

    return " ".join(cleaned)


def extract_text_from_frame(frame):
    h, w, _ = frame.shape
    y_start, y_end = int(h * CROP_REGION["y1"]), int(h * CROP_REGION["y2"])
    x_start, x_end = int(w * CROP_REGION["x1"]), int(w * CROP_REGION["x2"])

    cropped = frame[y_start:y_end, x_start:x_end]
    speaker = detect_speaker_color(cropped)

    cv2.imshow("Cropped Caption Region", cropped)
    cv2.waitKey(1)  # Display for a brief moment
    cv2.destroyAllWindows()
    
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    raw_text = pytesseract.image_to_string(thresh, config='--oem 3 --psm 6')
    return speaker + ": " + clean_text(raw_text).strip()
