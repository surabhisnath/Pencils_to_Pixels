import cv2
import numpy as np
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt
from PIL import Image
import matplotlib.pyplot as plt
import os
from scipy.stats import f_oneway, kruskal, shapiro
import scikit_posthocs as sp
from pathlib import Path
import pandas as pd
from tqdm import tqdm

col = "ink_fraction_inside_mask"
creators = ["children", "adults", "AI"]
stim_to_mask = {"G": (67, 84, 255, 220), "I": (116, 70, 165, 225), "R": (115, 82, 161, 220)}

for creator in tqdm(creators):
    df = pd.read_csv(f"../../csvs/{creator}_product.csv")
    df[col] = None
    for index, row in df.iterrows():
        stim = row["stimuli"]
        image = cv2.imread("../" + row["filepath"], cv2.IMREAD_GRAYSCALE)
        _, image = cv2.threshold(image, 250, 255, cv2.THRESH_BINARY_INV)
        ink_density = np.sum(image == 255)
        x, y, w, h = stim_to_mask[stim]
        mask = np.zeros_like(image, dtype=np.uint8)
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, thickness=-1)
        black_inside = np.sum((image == 255) & (mask == 255))
        black_outside = np.sum((image == 255) & (mask == 0))
        assert black_inside + black_outside == ink_density        
        inside_frac = black_inside/ink_density
        df.at[index, col] = inside_frac
    df.to_csv(f"../../csvs/{creator}_product.csv", index=False)