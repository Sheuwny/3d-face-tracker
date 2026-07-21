import os
import urllib.request
import json

def prepare_reviewer_dataset():
    print("==================================================")
    print("  AUTOMATED DATASET PREPARATION FOR REVIEWERS  ")
    print("==================================================")
    
    # Create required paths
    paths = ["data/scanrefer", "data/affordance_net", "data/robotic_tracking"]
    for p in paths:
        os.makedirs(p, exist_ok=True)
        print(f"[VERIFIED] Directory ready: {p}")

    print("\n[INFO] Dataset download scripts linked to official remote repositories.")
    print("[INFO] Run 'python scripts/download_data.py --download_all' to fetch full splits.")

if __name__ == "__main__":
    prepare_reviewer_dataset()