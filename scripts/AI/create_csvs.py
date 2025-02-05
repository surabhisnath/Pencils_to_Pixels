import pandas as pd
import numpy as np
import os
from tqdm import tqdm 

keys = ["pid", "creator", "age", "stimuli", "inverted", "filename", "filepath", "blip_caption", "blip_caption_correct", "gpt_caption", "gpt_caption_correct", "gpt_caption_confidence", "originality_audra", "originality_ocs", "percentile_ocs", "category1", "category2", "category3", "confidence", "creativity"]


age = np.nan
inverted = False
category1, category2, category3, confidence, creativity, blip_caption_correct, gpt_caption_correct = ["", "", "", "", "", "", ""]

rows = []


for prompt in ["1", "2", "3"]:
    creator = f"AI_prompt{prompt}"

    for i, stimuli in enumerate(["G", "I", "R"]):

        image_folder = f"../../data/AI/prompt_{prompt}/stimuli_{stimuli}/"  # Replace with the path to your local image
        
        blipcsv = pd.read_csv(f"../../measures/content_measures/blip_captions/output/AI_prompt{prompt}_{stimuli}.csv")
        blipdict = dict(zip(blipcsv["filepath"], blipcsv["blip_caption"]))

        gptcsv = pd.read_csv(f"../../measures/content_measures/gpt_captions/output/AI_prompt{prompt}_{stimuli}.csv")
        gptdict = dict(zip(gptcsv["filepath"], gptcsv["gpt_caption"]))

        audracsv = pd.read_csv(f"../../measures/GT_measures/AuDrA/output/AI_prompt{prompt}_{stimuli}.csv")
        audradict = dict(zip(audracsv["filenames"], audracsv["predictions"]))

        ocscsv = pd.read_csv(f"../../measures/GT_measures/OCS/output/AI_prompt{prompt}_{stimuli}.csv")
        ocsdictori = dict(zip(ocscsv["filepath"], ocscsv["originality_ocs"]))
        ocsdictper = dict(zip(ocscsv["filepath"], ocscsv["percentile_ocs"]))

        
        for imagename in tqdm(sorted(os.listdir(image_folder))):
            
            pid = 0     # arbitrary
            filename = imagename
            filepath = image_folder[image_folder.find("../data"):] + imagename


            blip_caption = blipdict[filepath]
            
            gptsplit = gptdict[filepath].split("[")
            gpt_caption = gptsplit[0][:-2]
            gpt_caption_confidence = gptsplit[1][0]

            originality_audra = audradict[filename]
            originality_ocs = ocsdictori[filepath]
            percentile_ocs = ocsdictper[filepath]


            values = [pid, creator, age, stimuli, inverted, filename, filepath, blip_caption, blip_caption_correct, gpt_caption, gpt_caption_correct, gpt_caption_confidence, originality_audra, originality_ocs, percentile_ocs, category1, category2, category3, confidence, creativity]
            row = dict(zip(keys, values))
            rows.append(row)

df = pd.DataFrame(rows)
df.to_csv(f"../../csvs/AI_product.csv", index=False)