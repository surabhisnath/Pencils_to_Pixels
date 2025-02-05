import pandas as pd
import numpy as np
import os
from tqdm import tqdm 

keys1 = ["pid", "creator", "age", "stimuli", "inverted", "filename", "filepath", "blip_caption", "blip_caption_correct", "gpt_caption", "gpt_caption_correct", "gpt_caption_confidence", "originality_audra", "originality_ocs", "percentile_ocs", "category1", "category2", "category3", "confidence", "creativity"]
keys2 = ["pid", "creator", "age", "filenames", "filepaths", "flexibility_manual", "flexibility_categories"]

creator = "adults"
age = np.nan
inverted = False
category1, category2, category3, confidence, creativity, blip_caption_correct, gpt_caption_correct = ["", "", "", "", "", "", ""]
flexibility_manual, flexibility_categories = ["", ""]

rows1 = []

pid_to_filenames = {}
pid_to_filepaths = {}

for i, stimuli in enumerate(["G", "I", "R"]):

    image_folder = f"../../data/adults/stimuli_{stimuli}/"  # Replace with the path to your local image
    
    blipcsv = pd.read_csv(f"../../measures/content_measures/blip_captions/output/adults_{stimuli}.csv")
    blipdict = dict(zip(blipcsv["filepath"], blipcsv["blip_caption"]))

    gptcsv = pd.read_csv(f"../../measures/content_measures/gpt_captions/output/adults_{stimuli}.csv")
    gptdict = dict(zip(gptcsv["filepath"], gptcsv["gpt_caption"]))

    audracsv = pd.read_csv(f"../../measures/GT_measures/AuDrA/output/adults_{stimuli}.csv")
    audradict = dict(zip(audracsv["filenames"], audracsv["predictions"]))

    ocscsv = pd.read_csv(f"../../measures/GT_measures/OCS/output/adults_{stimuli}.csv")
    ocsdictori = dict(zip(ocscsv["filepath"], ocscsv["originality_ocs"]))
    ocsdictper = dict(zip(ocscsv["filepath"], ocscsv["percentile_ocs"]))

    
    for imagename in tqdm(sorted(os.listdir(image_folder))):
        
        pid = imagename.split("__")[0]
        filename = imagename
        filepath = image_folder[3:] + imagename

        if pid in pid_to_filenames:
            pid_to_filenames[pid].append(filename)
        else:
            pid_to_filenames[pid] = [filename]

        if pid in pid_to_filepaths:
            pid_to_filepaths[pid].append(filepath)
        else:
            pid_to_filepaths[pid] = [filepath]


        blip_caption = blipdict[filepath]
        
        gptsplit = gptdict[filepath].split("[")
        gpt_caption = gptsplit[0][:-2]
        gpt_caption_confidence = gptsplit[1][0]

        originality_audra = audradict[filename]
        originality_ocs = ocsdictori[filepath]
        percentile_ocs = ocsdictper[filepath]

        values = [pid, creator, age, stimuli, inverted, filename, filepath, blip_caption, blip_caption_correct, gpt_caption, gpt_caption_correct, gpt_caption_confidence, originality_audra, originality_ocs, percentile_ocs, category1, category2, category3, confidence, creativity]
        row = dict(zip(keys1, values))
        rows1.append(row)

df = pd.DataFrame(rows1)
df.to_csv(f"../../csvs/{creator}_product.csv", index=False)

rows2 = []
for p in tqdm(pid_to_filenames):
    if len(pid_to_filenames[p]) != 3 or len(pid_to_filepaths[p]) != 3:
        print(p, len(pid_to_filenames[p]), len(pid_to_filepaths[p]))
    
    filenames = ", ".join(pid_to_filenames[p])
    filepaths = ", ".join(pid_to_filepaths[p])
    values = [p, creator, age, filenames, filepaths, flexibility_manual, flexibility_categories]
    row = dict(zip(keys2, values))
    rows2.append(row)

df = pd.DataFrame(rows2)
df.to_csv(f"../../csvs/{creator}_process.csv", index=False)