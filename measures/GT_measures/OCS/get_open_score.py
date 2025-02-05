# Load model directly
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
from PIL import Image
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

dist = pd.read_csv('./score_norm_distribution.csv', dtype=float)

def get_percentile(score):
    return dist[dist['score_norm'] <= score].iloc[-1, 0]

def inverse_scale(logits):
    # undo the min-max scaling that was done from the JRT range to 0-1
    scaler_params = {'min': -3.024, 'max': 3.164, 'range': 6.188}
    return logits * (scaler_params['range']) + scaler_params['min']

image_processor = AutoImageProcessor.from_pretrained("POrg/ocsai-d-web")
model = AutoModelForImageClassification.from_pretrained("POrg/ocsai-d-web")


creators = ["children", "adults", "AI/prompt_1", "AI/prompt_2", "AI/prompt_3"]

for creator in creators:
    for stim in ["G", "I", "R"]:

        image_folder = f"../../../data/{creator}/stimuli_{stim}/"
        
        rows = []
        for imagename in tqdm(sorted(os.listdir(image_folder))):
            image = Image.open(image_folder + imagename)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            inputs = image_processor(image, return_tensors="pt")

            with torch.no_grad():
                prediction = model(**inputs)
                score = prediction.logits[0].detach().numpy()[0]
                score = min(max(score, 0), 1)
            
            # print("Originiality:", np.round(score, 2))
            # print("Percentile:", get_percentile(score))
            rows.append({"creator": creator.replace("_", "").replace("/", "_"), "stimuli": stim, "image": imagename, "filepath": image_folder[image_folder.find("../data/"):] + imagename, "originality_ocs": np.round(score, 2), "percentile_ocs": get_percentile(score)})

        df = pd.DataFrame(rows)
        df.to_csv(f"output/{creator.replace("_", "").replace("/", "_")}_{stim}.csv", index=False)