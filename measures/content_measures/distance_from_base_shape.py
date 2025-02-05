import cv2
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import pickle as pk
from scipy.spatial.distance import cosine

modelnames = ["dino", "clip"]

for modelname in modelnames:
    col = f"dist_from_base_shape_{modelname}"
    creators = ["children", "adults", "AI"]

    base_embeddings = pk.load(open(f"../../saved/image_embeddings/base_embeddings_{modelname}.pk", "rb"))

    for creator in creators:
        embeddings = pk.load(open(f"../../saved/image_embeddings/embeddings_{creator}_{modelname}.pk", "rb"))
        
        df = pd.read_csv(f"../../csvs/{creator}_product.csv")
        df[col] = None
        for index, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing rows for {creator}"):
            stim = row["stimuli"]

            image_embedding = embeddings[row["filepath"]]
            base_embedding = base_embeddings[stim]
            dist_from_base_shape = cosine(base_embedding, image_embedding)
            df.at[index, col] = dist_from_base_shape

        df.to_csv(f"../../csvs/{creator}_product.csv", index=False)