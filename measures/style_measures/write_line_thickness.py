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

col = "line_thickness"
creators = ["children", "adults", "AI"]

for creator in tqdm(creators):
    df = pd.read_csv(f"../../csvs/{creator}_product.csv")
    df[col] = None
    for index, row in df.iterrows():
        image = cv2.imread("../" + row["filepath"], cv2.IMREAD_GRAYSCALE)
        _, binary = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        skeleton = cv2.ximgproc.thinning(binary)
        distance_transform = distance_transform_edt(binary)
        thickness_values = (distance_transform[skeleton > 0] * 2)
        average_thickness = thickness_values.mean()
        df.at[index, col] = average_thickness
    df.to_csv(f"../../csvs/{creator}_product.csv", index=False)