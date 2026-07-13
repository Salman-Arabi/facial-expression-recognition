"""Download and prepare the FER2013 dataset.

Usage:
    python src/download_fer2013.py

Downloads the FER2013 dataset from Kaggle and organizes it into:
    data/train/{emotion}/
    data/test/{emotion}/
"""

import os
import zipfile
import shutil

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

KAGGLE_URL = "https://www.kaggle.com/api/v1/datasets/astraszab/facial-expression-dataset-image-folders-fer2013/download"

def download():
    if os.path.exists(DATA_DIR) and len(os.listdir(DATA_DIR)) > 0:
        print(f"Data directory '{DATA_DIR}' already exists and is not empty.")
        print("If you want to re-download, delete the data/ folder and run again.")
        return

    zip_path = os.path.join(BASE_DIR, "fer2013.zip")

    if not os.path.exists(zip_path):
        print("Downloading FER2013 dataset from Kaggle...")
        print(f"URL: {KAGGLE_URL}")
        print()
        print("Option 1 — Download manually:")
        print(f"  1. Go to https://www.kaggle.com/datasets/astraszab/facial-expression-dataset-image-folders-fer2013")
        print(f"  2. Click Download and save as '{zip_path}'")
        print(f"  3. Run this script again")
        print()
        print("Option 2 — Use Kaggle API:")
        print(f"  pip install kaggle")
        print(f"  kaggle datasets download astraszab/facial-expression-dataset-image-folders-fer2013")
        print(f"  python src/download_fer2013.py")
        return

    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(DATA_DIR)
    os.remove(zip_path)
    print(f"Dataset extracted to {DATA_DIR}/")
    print("Done.")

if __name__ == "__main__":
    download()
