import cv2
import numpy as np
import os
from PIL import Image
from pathlib import Path
import pandas as pd
from tqdm import tqdm

col = "number_of_lines"
creators = ["children", "adults", "AI"]

for creator in creators:
    df = pd.read_csv(f"../../csvs/{creator}_product.csv")
    df[col] = None
    for index, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing rows for {creator}"):
        image = cv2.imread("../" + row["filepath"], cv2.IMREAD_GRAYSCALE)
        _, binary = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        skeleton = cv2.ximgproc.thinning(binary)
        lines = cv2.HoughLinesP(skeleton, rho=0.5, theta=np.pi/180, threshold=30, minLineLength=20, maxLineGap=50)
        df.at[index, col] = len(lines) if lines is not None else 0

        # # optionally save the lines
        # output_image = cv2.imread("../" + row["filepath"])
        # if lines is not None:
        #     for line in lines:
        #         x1, y1, x2, y2 = line[0]
        #         cv2.line(output_image, (x1, y1), (x2, y2), (0, 255, 0), 1)  # Draw line in green
        
        # savepath = f"lines/{os.path.dirname(row["filepath"][8:])}/"
        # os.makedirs(os.path.dirname(savepath), exist_ok=True)
        # cv2.imwrite(savepath + os.path.basename(row["filepath"]), output_image)

    df.to_csv(f"../../csvs/{creator}_product.csv", index=False)