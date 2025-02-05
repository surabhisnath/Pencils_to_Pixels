import cv2
import numpy as np
import os
from PIL import Image
from pathlib import Path
import pandas as pd
from tqdm import tqdm

col = "ink_density"
creators = ["children", "adults", "AI"]

for creator in tqdm(creators):
    df = pd.read_csv(f"../../csvs/{creator}_product.csv")
    df[col] = None
    for index, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing rows for {creator}"):
        image = cv2.imread("../" + row["filepath"], cv2.IMREAD_GRAYSCALE)
        _, image = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY_INV)
        ink_density = np.sum(image == 255) / image.size * 100
        df.at[index, col] = ink_density
    df.to_csv(f"../../csvs/{creator}_product.csv", index=False)