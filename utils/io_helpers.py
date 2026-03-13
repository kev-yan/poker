import json
from pathlib import Path

def load_data(path: Path):
    with open(path, 'r') as file:
        return json.load(file)
    
def save_data(path: Path, data):
    with open(path, 'w') as file:
        json.dump(data, file, indent=2)