from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
REELS_DIR = SCRIPT_DIR.parent / "reels"
OUTPUT_DIR = SCRIPT_DIR.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

TESSERACT_PATH = "/opt/homebrew/bin/tesseract"

CROP_REGION = {
    "y1": 0.58,  # vertical crop (start %)
    "y2": 0.70,  # vertical crop (end %)
    "x1": 0.07,  # horizontal crop (start %)
    "x2": 0.93,  # horizontal crop (end %)
}
