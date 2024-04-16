"""
This script is used to download the images from the Fitzpatrick17k dataset.
"""

import sys
import os
sys.path.extend([
    "./",
])

import requests
from PIL import Image
from io import BytesIO
import pandas as pd
import os
from tqdm import tqdm

from src.utils import util_path

def load_image(url):
    # Some websites may block requests that don't appear to come from a web browser to prevent scraping. We can mimic a browser request by setting a User-Agent header in the script.
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    try:
        # print(f"Attempting to load image from URL: {url}")
        response = requests.get(url, headers=headers)
        if response.status_code == 404:
            print(f"404 Not Found: {url}")
            return None
        img = Image.open(BytesIO(response.content))
        return img
    except Exception as e:
        print(f"Failed to load image from {url}. Error: {e}")
        return None
def save_image(img, file_path):
    try:
        img.save(file_path, format='PNG')
        # print(f"Image saved to {file_path}")
    except Exception as e:
        print(f"Failed to save image. Error: {e}")

# Main
if __name__ == "__main__":

    dataset_name = 'fitzpatrick17'
    data_dir = os.path.join('data', dataset_name)
    data_dir_raw =  os.path.join(data_dir, 'raw')
    data_dir_interim = os.path.join(data_dir, 'interim')

    # Create directories
    util_path.create_dir(data_dir_raw)
    util_path.create_dir(data_dir_interim)

    # Load annotation csv file
    df: object = pd.read_csv(os.path.join(data_dir, 'fitzpatrick17k.csv'))

    # Download images
    # Use tqdm
    for index in tqdm(range(len(df))):
        # Extract the corrensponding image url
        url = df['url'][index]
        # Extract the corrensponding md5 hash
        md5hash = df['md5hash'][index]
        img = load_image(url)
        if img is not None:
            file_path = os.path.join(data_dir_raw, f"img__{md5hash}.png")
            save_image(img, file_path)