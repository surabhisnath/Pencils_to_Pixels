import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F
from transformers import AutoProcessor, AutoImageProcessor, Dinov2Model, CLIPModel
import torch
from datasets import load_dataset
import pickle as pk

def get_embeddings(images, modelname, keys):
    if modelname == "dino":
        processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        model = Dinov2Model.from_pretrained("facebook/dinov2-base").cuda()
    elif modelname == "clip":
        processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").cuda()

    inputs = processor(images=images, return_tensors="pt")
    with torch.no_grad():
        if modelname == "dino":
            embeddings = model(**{x:v.cuda() for x,v in inputs.items()}).pooler_output
        elif modelname == "clip":
            embeddings = model.get_image_features(**{x:v.cuda() for x,v in inputs.items()})
        print(embeddings.shape)
        return dict(zip(keys, embeddings.cpu().numpy()))

modelname = "clip"
creators = ["children", "adults", "AI"]

base_embeddings = {}
stimuli = ["G", "I", "R"]
base_embeddings = get_embeddings([Image.open(p) for p in [f"../../stimuli/originals/stimuli_{stim}.png" for stim in stimuli]], modelname, stimuli)
pk.dump(base_embeddings, open(f"../../saved/image_embeddings/base_embeddings_{modelname}.pk", "wb"))

for creator in ["children", "adults", "AI"]:
    filename = f"embeddings_{creator}_{modelname}"
    product = pd.read_csv(f"../../csvs/{creator}_product.csv")
    imagepaths = product["filepath"].tolist()
    images = [Image.open("../" + p) for p in imagepaths]
    embeddings = get_embeddings(images, modelname, imagepaths)
    pk.dump(embeddings, open(f"../../saved/image_embeddings/{filename}.pk", "wb"))