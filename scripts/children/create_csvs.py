import pandas as pd
import numpy as np
import os
from tqdm import tqdm 

keys1 = ["pid", "creator", "age", "stimuli", "inverted", "filename", "filepath", "blip_caption", "blip_caption_correct", "gpt_caption", "gpt_caption_correct", "gpt_caption_confidence", "originality_audra", "originality_ocs", "percentile_ocs", "category1", "category2", "category3", "confidence", "creativity"]
keys2 = ["pid", "creator", "age", "filenames", "filepaths", "flexibility_manual", "flexibility_categories"]

product = pd.read_csv("../../csvs/product.csv")
process = pd.read_csv("../../csvs/process.csv")

blipcsv = pd.concat([pd.read_csv("../../measures/content_measures/blip_captions/output/children_G.csv"), pd.read_csv("../../measures/content_measures/blip_captions/output/children_I.csv"), pd.read_csv("../../measures/content_measures/blip_captions/output/children_R.csv")], ignore_index=True)
blipdict = dict(zip(blipcsv["filepath"], blipcsv["blip_caption"]))

gptcsv = pd.concat([pd.read_csv("../../measures/content_measures/gpt_captions/output/children_G.csv"), pd.read_csv("../../measures/content_measures/gpt_captions/output/children_I.csv"), pd.read_csv("../../measures/content_measures/gpt_captions/output/children_R.csv")], ignore_index=True)
gptcsv['gpt_caption_only'] = gptcsv['gpt_caption'].apply(lambda x: x[:x.find('[') - 2])
gptcsv['gpt_caption_confidence'] = gptcsv['gpt_caption'].apply(lambda x: x[x.find('[') + 1])
gptdict1 = dict(zip(gptcsv["filepath"], gptcsv["gpt_caption_only"]))
gptdict2 = dict(zip(gptcsv["filepath"], gptcsv["gpt_caption_confidence"]))

audracsv1 = pd.read_csv(f"../../measures/GT_measures/AuDrA/output/children_G.csv")
audracsv1["filepath"] = "../data/children/stimuli_G/" + audracsv1["filenames"]
audracsv2 = pd.read_csv(f"../../measures/GT_measures/AuDrA/output/children_I.csv")
audracsv2["filepath"] = "../data/children/stimuli_I/" + audracsv2["filenames"]
audracsv3 = pd.read_csv(f"../../measures/GT_measures/AuDrA/output/children_R.csv")
audracsv3["filepath"] = "../data/children/stimuli_R/" + audracsv3["filenames"]
audracsv = pd.concat([audracsv1, audracsv2, audracsv3])
audradict = dict(zip(audracsv["filepath"], audracsv["predictions"]))

ocscsv1 = pd.read_csv(f"../../measures/GT_measures/OCS/output/children_G.csv")
ocscsv2 = pd.read_csv(f"../../measures/GT_measures/OCS/output/children_I.csv")
ocscsv3 = pd.read_csv(f"../../measures/GT_measures/OCS/output/children_R.csv")

ocscsv = pd.concat([ocscsv1, ocscsv2, ocscsv3])
ocsdictori = dict(zip(ocscsv["filepath"], ocscsv["originality_ocs"]))
ocsdictper = dict(zip(ocscsv["filepath"], ocscsv["percentile_ocs"]))

product["creator"] = "children"
product["filename"] = product['ID'].astype(str) + '.png'
product["filepath"] = '../data/children/stimuli_' + product['stim'].astype(str) + '/' + product['ID'].astype(str) + '.png'
product["blip_caption"] = product["filepath"].map(blipdict)
product["blip_caption_correct"] = ""
product["gpt_caption"] = product["filepath"].map(gptdict1)
product["gpt_caption_correct"] = ""
product["gpt_caption_confidence"] = product["filepath"].map(gptdict2)
product["originality_audra"] = product["filepath"].map(audradict)
product["originality_ocs"] = product["filepath"].map(ocsdictori)
product["percentile_ocs"] = product["filepath"].map(ocsdictper)

product = product.rename(columns={"ID": "pid", "stim": "stimuli"})
product = product[keys1]

pid_to_filenames = product.groupby("pid")["filename"].agg(lambda x: ", ".join(x)).to_dict()
pid_to_filepaths = product.groupby("pid")["filepath"].agg(lambda x: ", ".join(x)).to_dict()

process = process.rename(columns={"ID": "pid", "flexibility (manually)": "flexibility_manual", "flexibility (categories)": "flexibility_categories"})
process["creator"] = "children"
process["filenames"] = process["pid"].map(pid_to_filenames)
process["filepaths"] = process["pid"].map(pid_to_filepaths)
process = process[keys2]

product.to_csv("../../csvs/children_product.csv", index = False)
process.to_csv("../../csvs/children_process.csv", index = False)